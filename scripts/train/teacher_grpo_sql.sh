#!/bin/bash
# Stage 2 (teacher GRPO), text-to-SQL: RL-train the SQL teacher from the stage-1
# cold-start model, the way teacher_grpo_math.sh does for math. Same presets, GPU
# split, optimizer, clip and batch; only the environment differs: the model queries a
# SQLite database through the sql_executor tool (train/src/generate_with_sql.py) and is
# rewarded +1 / -1 on whether its final query's result equals the gold query's.
#
# Data (scripts/data/prepare_sql.py):
#   python3 scripts/data/prepare_sql.py --out /workspace/data/sql
#   -> /workspace/data/sql/train.jsonl and /workspace/data/sql/databases/<db_id>/<db_id>.sqlite
#
# Usage:
#   MODEL=qwen3-4b \
#   HF_CHECKPOINT=/workspace/ckpt/sft_hf REF_LOAD=/workspace/ckpt/sft_torch_dist   (the defaults) \
#   PROMPT_DATA=/workspace/data/sql/train.jsonl \
#   SQL_DB_ROOT=/workspace/data/sql/databases \
#   SAVE_DIR=/workspace/runs/grpo_sql \
#       bash scripts/train/teacher_grpo_sql.sh
#
# No in-training eval; accuracy is measured separately with scripts/eval/sql.sh.
# Budgets are the search environment's: 9216 tokens per trajectory, 4096 new tokens
# per turn, 8 turns, 6 queries. A schema prompt plus six queries with 20-row results
# fits comfortably.

set -ex

export PYTHONUNBUFFERED=1
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
RELEASE_DIR="${REPO_DIR}/train"
SRC_DIR="${RELEASE_DIR}/src"
SLIME_DIR="${REPO_DIR}/third_party/slime"

# ===========================================================================
# KEY SETTINGS — what you most likely need to change.
# ===========================================================================

# --- Model preset (config + parallelism + rollout engine / memory) ----------
MODEL=${MODEL:-qwen3-4b}
case "${MODEL}" in
    qwen3-4b)
        _MODEL_CONFIG="qwen3-4B-Instruct-2507.sh"
        _TP=2; _EP=1; _OFFLOAD=0; _ENGINE=2; _MEM=0.6; _MAXTOK=16400; _NROLLOUT=500
        _KVTOK=147456; _WGIB=8 ;;
    *)
        echo "Unknown MODEL='${MODEL}'. Use qwen3-4b" >&2
        exit 1 ;;
esac
MODEL_CONFIG=${MODEL_CONFIG:-"${REPO_DIR}/scripts/models/${_MODEL_CONFIG}"}

# --- Checkpoints ------------------------------------------------------------
# The cold-start SFT checkpoint, the same files on every machine since 2026-10-16;
# the directory check below fails loudly if a machine lacks them.
HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/ckpt/sft_hf}
REF_LOAD=${REF_LOAD:-/workspace/ckpt/sft_torch_dist}
for _ckpt_var in HF_CHECKPOINT REF_LOAD; do
   if [ ! -d "${!_ckpt_var}" ]; then
      echo "${_ckpt_var}=${!_ckpt_var} is not a directory." >&2
      exit 1
   fi
done
if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
   echo "HF_CHECKPOINT=${HF_CHECKPOINT} has no config.json, so it is not an HF checkpoint dir." >&2
   exit 1
fi

SAVE_DIR=${SAVE_DIR:?"set SAVE_DIR to the teacher output dir"}

# --- Data -------------------------------------------------------------------
PROMPT_DATA=${PROMPT_DATA:?"set PROMPT_DATA to the SQL prompt jsonl (prepare_sql.py's train.jsonl)"}
SQL_DB_ROOT=${SQL_DB_ROOT:?"set SQL_DB_ROOT to prepare_sql.py's databases/ directory"}
if [ ! -f "${PROMPT_DATA}" ]; then
   echo "PROMPT_DATA=${PROMPT_DATA} does not exist." >&2
   exit 1
fi
if [ ! -d "${SQL_DB_ROOT}" ] || [ -z "$(ls -A "${SQL_DB_ROOT}" 2>/dev/null)" ]; then
   echo "SQL_DB_ROOT=${SQL_DB_ROOT} is not a non-empty directory of <db_id>/<db_id>.sqlite." >&2
   exit 1
fi

# --- Anything already on the GPUs (see teacher_grpo_math.sh) -----------------
if [ -z "${RETRIEVER_RESERVE_GIB:-}" ]; then
   _GPU_USED_MAX_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -1)
   RETRIEVER_RESERVE_GIB=$(awk -v used="${_GPU_USED_MAX_MIB:-0}" \
                               'BEGIN { printf "%d", (used < 256) ? 0 : int((used + 512) / 1024 + 0.999) }')
   echo "[gpu] most-used GPU before launch: ${_GPU_USED_MAX_MIB:-unknown} MiB -> reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (override with RETRIEVER_RESERVE_GIB)"
else
   echo "[gpu] reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (RETRIEVER_RESERVE_GIB given)"
fi

_GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "${_GPU_MEM_MIB}" ] && [ "${_GPU_MEM_MIB}" -ge 49152 ] 2>/dev/null; then
   _MEM_BASE=0.85
else
   _MEM_BASE=0.78
fi
_MEM_ASYNC=$(awk -v base="${_MEM_BASE}" -v gib="${RETRIEVER_RESERVE_GIB}" \
                 -v mib="${_GPU_MEM_MIB:-40960}" 'BEGIN {
   f = base - (gib * 1024.0) / mib;
   if (f < 0.30) f = 0.30;
   printf "%.3f", f;
}')
echo "[gpu] memory.total=${_GPU_MEM_MIB:-unknown} MiB, base ${_MEM_BASE}, minus ${RETRIEVER_RESERVE_GIB} GiB resident -> async sglang-mem-fraction-static ${_MEM_ASYNC}"

# --- GPUs / parallelism (defaults from the MODEL preset) --------------------
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-${_TP}}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-${_EP}}
ETP_SIZE=${ETP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-${_MAXTOK}}
ASYNC=${ASYNC:-1}
if [ "${ASYNC}" = "1" ]; then
   ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-2}
   ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-6}
   ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
   SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-${_MEM_ASYNC}}
   OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-1}
   if [ $((ACTOR_NUM_GPUS + ROLLOUT_NUM_GPUS)) -ne "${NUM_GPUS}" ]; then
      echo "ACTOR_NUM_GPUS(${ACTOR_NUM_GPUS}) + ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS}) must equal NUM_GPUS(${NUM_GPUS})." >&2
      exit 1
   fi
fi
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-${_OFFLOAD}}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${_ENGINE}}

_MODEL_PAR=$((TP_SIZE * PP_SIZE * CP_SIZE))
if [ $((${ACTOR_NUM_GPUS:-${NUM_GPUS}} % _MODEL_PAR)) -ne 0 ]; then
   echo "[preflight] MODEL=${MODEL} needs TP x PP x CP = ${_MODEL_PAR} GPUs per replica; the training pool is ${ACTOR_NUM_GPUS:-${NUM_GPUS}} GPU(s)." >&2
   exit 1
fi
if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "[preflight] ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} is not a multiple of ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}." >&2
   exit 1
fi
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-${_MEM}}

# ===========================================================================
# ADVANCED SETTINGS — usually fine as-is.
# ===========================================================================

NUM_ROLLOUT=${NUM_ROLLOUT:-${_NROLLOUT}}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
# The search environment's budget, not math's 16384: a SQL trajectory is a schema
# prompt plus a handful of short queries and 20-row results.
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-9216}

# Rollout concurrency per engine, derived from the KV pool (teacher_grpo_math.sh).
if [ -z "${SGLANG_SERVER_CONCURRENCY:-}" ]; then
   SGLANG_SERVER_CONCURRENCY=$(awk -v mib="${_GPU_MEM_MIB:-40960}" \
                                   -v frac="${SGLANG_MEM_FRACTION_STATIC}" \
                                   -v gpus="${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
                                   -v wgib="${_WGIB}" \
                                   -v ctx="${ROLLOUT_MAX_CONTEXT_LEN}" \
                                   -v kv="${_KVTOK}" 'BEGIN {
      pool = frac * mib * 1048576 * gpus - wgib * 1073741824;
      n = 0.8 * pool / (ctx * kv);
      if (n < 4) n = 4;
      printf "%d", n;
   }')
fi
echo "[gpu] sglang-server-concurrency ${SGLANG_SERVER_CONCURRENCY} per engine"
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
LR=${LR:-1e-6}
LOAD=${LOAD:-${SAVE_DIR}}
ROLLOUT_DEBUG_DIR=${ROLLOUT_DEBUG_DIR:-${SAVE_DIR}/rollout_debug}

MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# --- SQL environment (train/src/sql_tools.py) -------------------------------
export SQL_DB_ROOT
export SQL_MAX_TOOL_CALLS=${SQL_MAX_TOOL_CALLS:-6}
export SQL_MAX_TURNS=${SQL_MAX_TURNS:-8}
export SQL_MAX_TOKENS_PER_TURN=${SQL_MAX_TOKENS_PER_TURN:-4096}
export SQL_EXEC_TIMEOUT=${SQL_EXEC_TIMEOUT:-5}
export SQL_SCORE_TIMEOUT=${SQL_SCORE_TIMEOUT:-30}
export SQL_MAX_RESULT_ROWS=${SQL_MAX_RESULT_ROWS:-20}
export SQL_MAX_RESULT_CHARS=${SQL_MAX_RESULT_CHARS:-4000}
export SQL_TOOL_CONCURRENCY=${SQL_TOOL_CONCURRENCY:-128}
export GENERATE_SAMPLE_TIMEOUT=${GENERATE_SAMPLE_TIMEOUT:-1200}

# ===========================================================================
# No user-editable settings below this line.
# ===========================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

mkdir -p "${SAVE_DIR}" "${ROLLOUT_DEBUG_DIR}"
export WANDB_DIR="${SAVE_DIR}"

source "${MODEL_CONFIG}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${LOAD}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
   # no --rollout-stop: generate_with_sql puts </tool_call> and </answer> in the stop list itself
)

EVAL_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size "${ETP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --override-opt_param-scheduler
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
if [ "${OPTIMIZER_OFFLOAD}" = "1" ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-router-request-timeout-secs "${ROUTER_REQUEST_TIMEOUT_SECS:-600}"
)

WANDB_ARGS=()
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-agentic-opd}
WANDB_GROUP=${WANDB_GROUP:-grpo-sql-${MODEL}}
DISABLE_WANDB_RANDOM_SUFFIX=${DISABLE_WANDB_RANDOM_SUFFIX:-1}
export WANDB_DIR=${WANDB_DIR:-${SAVE_DIR}}

if [[ "${USE_WANDB}" == "1" ]]; then
   WANDB_ARGS+=(--use-wandb)
fi
if [[ -n "${WANDB_PROJECT}" ]]; then
   WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
   WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_KEY:-}" ]]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi
if [[ "${DISABLE_WANDB_RANDOM_SUFFIX}" == "1" ]]; then
   WANDB_ARGS+=(--disable-wandb-random-suffix)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --log-multi-turn
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_sql.generate
   --custom-rm-path generate_with_sql.reward_func
)

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
ulimit -u 65536 2>/dev/null || true
echo "[teacher_grpo] ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SRC_DIR}:${RELEASE_DIR}:${REPO_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"SQL_DB_ROOT\": \"${SQL_DB_ROOT}\",
    \"SQL_MAX_TOOL_CALLS\": \"${SQL_MAX_TOOL_CALLS}\",
    \"SQL_MAX_TURNS\": \"${SQL_MAX_TURNS}\",
    \"SQL_MAX_TOKENS_PER_TURN\": \"${SQL_MAX_TOKENS_PER_TURN}\",
    \"SQL_EXEC_TIMEOUT\": \"${SQL_EXEC_TIMEOUT}\",
    \"SQL_SCORE_TIMEOUT\": \"${SQL_SCORE_TIMEOUT}\",
    \"SQL_MAX_RESULT_ROWS\": \"${SQL_MAX_RESULT_ROWS}\",
    \"SQL_MAX_RESULT_CHARS\": \"${SQL_MAX_RESULT_CHARS}\",
    \"SQL_TOOL_CONCURRENCY\": \"${SQL_TOOL_CONCURRENCY}\",
    \"GENERATE_SAMPLE_TIMEOUT\": \"${GENERATE_SAMPLE_TIMEOUT}\"
  }
}"

if [ "${ASYNC}" = "1" ]; then
   ENTRYPOINT=train_async.py
   PLACEMENT_ARGS=(
      --actor-num-nodes 1
      --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}"
      --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
      --use-rollout-logprobs
   )
else
   ENTRYPOINT=train.py
   PLACEMENT_ARGS=(
      --colocate
      --actor-num-nodes 1
      --actor-num-gpus-per-node "${NUM_GPUS}"
   )
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir="${SLIME_DIR}" \
   -- python3 "${ENTRYPOINT}" \
   "${PLACEMENT_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
