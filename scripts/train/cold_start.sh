#!/bin/bash
# Stage 1 (cold start): SFT the base model on tool-use traces so it learns the
# tool-call format before teacher RL / distillation. Same recipe for every
# environment — only the data (PROMPT_DATA) differs:
#   math   -> https://huggingface.co/datasets/baohao/math-cold-start
#   search -> https://huggingface.co/datasets/baohao/search-cold-start
#   both   -> https://huggingface.co/datasets/baohao/both-cold-start
#
#   hf download baohao/math-cold-start --repo-type dataset --local-dir /path/to/data
#
# The base checkpoint must also exist in Megatron torch_dist format:
#   python tools/convert_hf_to_torch_dist.py ... (see tools/)
#
# The model preset MODEL (qwen3-4b) sets the slime config and the parallelism:
#   qwen3-4b   -> qwen3-4B-Instruct-2507.sh      TP2  EP1  offload off
#
# Usage:
#   MODEL=qwen3-4b \
#   HF_CHECKPOINT=/path/to/Qwen3-4B-Instruct-2507 \
#   REF_LOAD=/path/to/Qwen3-4B-Instruct-2507/torch_dist \
#   PROMPT_DATA=/path/to/data/train.parquet \
#   SAVE_DIR=/path/to/output \
#       bash scripts/train/cold_start.sh
#
# Any preset value can still be overridden explicitly, e.g. TP_SIZE=4 MODEL_CONFIG=...

set -ex

export PYTHONUNBUFFERED=1
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_DIR}/third_party/slime"

# ===========================================================================
# KEY SETTINGS — what you most likely need to change.
# Every variable is env-overridable, e.g.: LR=1e-5 bash scripts/train/cold_start.sh
# ===========================================================================

# --- Model ------------------------------------------------------------------
# Base model preset. Selects the slime config, parallelism, and optimizer
# offload below.
MODEL=${MODEL:-qwen3-4b}
case "${MODEL}" in
    qwen3-4b)
        _MODEL_CONFIG="qwen3-4B-Instruct-2507.sh";       _TP=2; _EP=1; _OFFLOAD=0 ;;
    *)
        echo "Unknown MODEL='${MODEL}'. Use qwen3-4b" >&2
        echo "(or set MODEL_CONFIG/TP_SIZE/EP_SIZE/OPTIMIZER_OFFLOAD explicitly)." >&2
        exit 1 ;;
esac
# slime model-args file (from the preset; override to use another config).
# The rotary base is baked into the configs; export MODEL_ARGS_ROTARY_BASE
# before sourcing only for a config whose default is wrong.
MODEL_CONFIG=${MODEL_CONFIG:-"${REPO_DIR}/scripts/models/${_MODEL_CONFIG}"}
# Base HF checkpoint and its Megatron torch_dist conversion:
HF_CHECKPOINT=${HF_CHECKPOINT:?"set HF_CHECKPOINT to the base HF checkpoint dir"}
REF_LOAD=${REF_LOAD:?"set REF_LOAD to the base torch_dist checkpoint dir"}
# Both are read long after Ray is up, and the failure they cause does not name
# them: transformers only treats hf_checkpoint as a Hub repo id once the local
# path fails to resolve, so a missing directory arrives as
#   HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
# forty lines into a traceback, with the cluster already started. Check here.
for _ckpt_var in HF_CHECKPOINT REF_LOAD; do
   if [ ! -d "${!_ckpt_var}" ]; then
      echo "${_ckpt_var}=${!_ckpt_var} is not a directory." >&2
      echo "On a machine this run has not used before, check the persistent disk is" >&2
      echo "mounted and the name matches:  ls $(dirname "${!_ckpt_var}")" >&2
      exit 1
   fi
done
if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
   echo "HF_CHECKPOINT=${HF_CHECKPOINT} has no config.json, so it is not an HF checkpoint dir." >&2
   exit 1
fi


# --- Data / output ----------------------------------------------------------
# train.parquet from the cold-start dataset for your environment:
#   baohao/math-cold-start | baohao/search-cold-start | baohao/both-cold-start
PROMPT_DATA=${PROMPT_DATA:?"set PROMPT_DATA to the cold-start parquet (e.g. .../train.parquet)"}
SAVE_DIR=${SAVE_DIR:?"set SAVE_DIR to the output checkpoint dir"}

# --- Training ---------------------------------------------------------------
NUM_EPOCH=${NUM_EPOCH:-3}
LR=${LR:-5e-5}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

# --- GPUs / parallelism (defaults come from the MODEL preset above) ---------
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-${_TP}}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-${_EP}}
ETP_SIZE=${ETP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
# Offload the optimizer state to CPU (off for the 4B model).
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-${_OFFLOAD}}

# ===========================================================================
# ADVANCED SETTINGS — usually fine as-is.
# ===========================================================================

MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# ===========================================================================
# No user-editable settings below this line.
# ===========================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

mkdir -p "${SAVE_DIR}"
export WANDB_DIR="${SAVE_DIR}"

source "${MODEL_CONFIG}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${PROMPT_DATA}"
   --input-key messages
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

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

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)
if [ "${OPTIMIZER_OFFLOAD}" = "1" ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

WANDB_ARGS=()
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-agentic-opd}
WANDB_GROUP=${WANDB_GROUP:-coldstart-${MODEL}}
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
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir="${SLIME_DIR}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"
