#!/bin/bash
# Stage 2 (teacher GRPO), math (ReTool): RL-train the teacher, initialized from
# the stage-1 cold-start model. The on-policy rollouts saved here become the
# replayed-prefix pool for student distillation (stage 3).
#
# Before running, re-materialize the cold-start checkpoint into a clean
# HF + torch_dist pair (see README "Teacher GRPO"):
#   bash scripts/train/convert_torch_dist_to_hf.sh   # cold-start iter -> HF
#   bash scripts/train/convert_hf_to_torch_dist.sh   # HF -> torch_dist
# then point HF_CHECKPOINT / REF_LOAD at them.
#
# RL prompts:
#   hf download baohao/math-train --repo-type dataset --local-dir /path/to/rl_data
#
# Pick the base model with MODEL (default qwen3-4b), which sets the slime config
# and parallelism / optimizer offload automatically:
#   qwen3-4b   -> qwen3-4B-Instruct-2507.sh      TP2 EP1  2 GPUs/engine  mem 0.6
#
# Usage:
#   MODEL=qwen3-4b \
#   HF_CHECKPOINT=/workspace/ckpt/sft_hf REF_LOAD=/workspace/ckpt/sft_torch_dist   (the defaults) \
#   PROMPT_DATA=/path/to/rl_data/train.jsonl \
#   SAVE_DIR=/path/to/teacher_run \
#       bash scripts/train/teacher_grpo_math.sh
#
# No in-training eval; final accuracy is measured separately with
# scripts/eval/math.sh. Any preset value can be overridden explicitly, e.g. TP_SIZE=4.

set -ex

export PYTHONUNBUFFERED=1
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
# torch._inductor builds a cache path via getpass.getuser(); on container UIDs
# missing from /etc/passwd that crashes, so pin the cache dir.
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
# Each preset also sets the per-model RL knobs (rollout GPUs/engine, SGLang mem
# fraction, max tokens/gpu, weights-backuper). Any can be overridden explicitly.
# _KVTOK is bytes of KV cache per token for the whole model and _WGIB the bf16
# weights per engine; both feed the rollout-concurrency derivation further down.
# Qwen3-4B: 36 layers x 8 KV heads x 128 head_dim x 2 (K,V)
# x 2 bytes = 147456.
MODEL=${MODEL:-qwen3-4b}
case "${MODEL}" in
    qwen3-4b)
        _MODEL_CONFIG="qwen3-4B-Instruct-2507.sh"
        _TP=2; _EP=1; _OFFLOAD=0; _ENGINE=2; _MEM=0.6; _MAXTOK=16400; _NROLLOUT=500
        _KVTOK=147456; _WGIB=8 ;;
    *)
        echo "Unknown MODEL='${MODEL}'. Use qwen3-4b" >&2
        echo "(or set MODEL_CONFIG/TP_SIZE/EP_SIZE/OPTIMIZER_OFFLOAD explicitly)." >&2
        exit 1 ;;
esac
MODEL_CONFIG=${MODEL_CONFIG:-"${REPO_DIR}/scripts/models/${_MODEL_CONFIG}"}

# --- Checkpoints (the cold-start round-trip outputs) ------------------------
# The cold-start SFT checkpoint, the same files on every machine since 2026-10-16;
# the directory check below fails loudly if a machine lacks them.
HF_CHECKPOINT=${HF_CHECKPOINT:-/workspace/ckpt/sft_hf}
REF_LOAD=${REF_LOAD:-/workspace/ckpt/sft_torch_dist}
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

SAVE_DIR=${SAVE_DIR:?"set SAVE_DIR to the teacher output dir"}

# --- Data -------------------------------------------------------------------
PROMPT_DATA=${PROMPT_DATA:?"set PROMPT_DATA to the RL prompt jsonl (baohao/math-train train.jsonl)"}
# --- Anything already on the GPUs -------------------------------------------
# The search retriever normally lives on this same node, in a screen on the VM,
# and math trains on what is left. SGLang sizes its static KV pool as a fraction
# of *total* card memory, not free memory, so whatever is resident at launch has
# to be subtracted explicitly or the pool is sized as if the cards were empty and
# the run OOMs an hour in, not at startup. Measure it rather than guess: the
# most-used GPU right now, before anything of ours launches, plus half a GiB of
# slack, rounded up. A clean node reads 0, so nothing is wasted when nothing is
# up; a leftover engine from a crashed run is accounted for the same way.
if [ -z "${RETRIEVER_RESERVE_GIB:-}" ]; then
   _GPU_USED_MAX_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -1)
   RETRIEVER_RESERVE_GIB=$(awk -v used="${_GPU_USED_MAX_MIB:-0}" \
                               'BEGIN { printf "%d", (used < 256) ? 0 : int((used + 512) / 1024 + 0.999) }')
   echo "[gpu] most-used GPU before launch: ${_GPU_USED_MAX_MIB:-unknown} MiB -> reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (override with RETRIEVER_RESERVE_GIB)"
else
   echo "[gpu] reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (RETRIEVER_RESERVE_GIB given)"
fi

# --- SGLang static fraction, sized from the card and what is already on it ---
# Baseline first. 0.85 of a 40 GB A100 reserves ~33.6 GB for weights + KV and
# leaves only ~6 GB for CUDA graphs, prefill activations, NCCL buffers and
# fragmentation. That is not enough here, and it fails *late*: a rollout engine
# ran 9.5 hours and then died trying to allocate 1.56 GB with 1.51 GB free. On
# an 80 GB card the same fraction leaves ~12 GB and is fine. Detect the card,
# then subtract the resident share of the *total*, which is the denominator
# SGLang's fraction is measured against.
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
# ASYNC=1 runs train_async.py: rollout N+1 is generated while step N trains, so the
# straggler tail and the offload/reload transitions stop stalling the trainer. This
# forbids --colocate, so the GPUs are split into a dedicated training pool and a
# dedicated rollout pool instead of time-sharing all of them.
#
# Defaults below are tuned for the multi-turn tool workload, which is far more
# rollout-heavy than slime's own examples/fully_async reference (that one is
# single-turn DAPO math and splits 4/4). Measured wait_time_ratio here is 0.83-0.90,
# hence 6 rollout / 2 train. Check the two pools after ~10 steps and rebalance if
# one starves.
#
# Because the rollout is now one update behind the trainer, --use-rollout-logprobs
# makes the importance ratio use the log probs the engine actually sampled with,
# rather than ones recomputed from the newer weights (which would silently pin the
# ratio at 1 and leave the staleness uncorrected). Watch pg_clipfrac: if it climbs
# well above the sync run, the mismatch is eating the eps-clip budget and --use-tis
# (mutually exclusive with this flag) clips the correction separately instead.
ASYNC=${ASYNC:-1}
if [ "${ASYNC}" = "1" ]; then
   ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS:-2}
   ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-6}
   # 2 GPUs per engine (-> 3 engines on 6 GPUs). slime's examples/fully_async
   # reference uses 1, but that run is single-turn with an 8192 response cap. At
   # 16384 tokens TP=1 both halves each engine's KV cache and halves decode speed,
   # and decode speed is exactly what the router's request timeout measures.
   ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
   # The rollout GPUs are no longer shared with the trainer, so SGLang can keep a
   # much bigger KV cache -- this is what actually raises rollout concurrency.
   SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-${_MEM_ASYNC}}
   # Keeps the 2-GPU training pool small: the distributed optimizer cannot
   # shard at DP=1, so without this the fp32 Adam state alone is ~48 GB/GPU.
   OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-1}
   if [ $((ACTOR_NUM_GPUS + ROLLOUT_NUM_GPUS)) -ne "${NUM_GPUS}" ]; then
      echo "ACTOR_NUM_GPUS(${ACTOR_NUM_GPUS}) + ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS}) must equal NUM_GPUS(${NUM_GPUS})." >&2
      exit 1
   fi
fi

# NOTE: the ASYNC block above assigns these three first, so the ${VAR:-...} here is
# a no-op under ASYNC=1. The values below are the sync/colocate defaults.
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-${_OFFLOAD}}
# GPUs per SGLang rollout engine (rollout and training colocate on NUM_GPUS).
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${_ENGINE}}

# --- Preflight: the presets have to fit the pools they are launched into -----
# TP x PP x CP must divide the training pool, and the rollout pool must split
# into whole engines. A TP_SIZE that the default 2-GPU async training pool
# cannot hold would otherwise only fail in Megatron, after Ray, the retriever
# and the rollout engines had all come up.
_MODEL_PAR=$((TP_SIZE * PP_SIZE * CP_SIZE))
if [ $((${ACTOR_NUM_GPUS:-${NUM_GPUS}} % _MODEL_PAR)) -ne 0 ]; then
   echo "[preflight] MODEL=${MODEL} needs TP x PP x CP = ${_MODEL_PAR} GPUs per model replica, but the" >&2
   echo "[preflight] training pool is ${ACTOR_NUM_GPUS:-${NUM_GPUS}} GPU(s) (ACTOR_NUM_GPUS). Give it a multiple of ${_MODEL_PAR}, e.g." >&2
   echo "[preflight]   ACTOR_NUM_GPUS=${_MODEL_PAR} bash $(basename "$0")" >&2
   exit 1
fi
if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "[preflight] ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} is not a multiple of ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE};" >&2
   echo "[preflight] $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) rollout GPU(s) would sit idle. Adjust one of them." >&2
   exit 1
fi
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-${_MEM}}


# ===========================================================================
# ADVANCED SETTINGS — usually fine as-is.
# ===========================================================================

# Rollout / GRPO knobs.
NUM_ROLLOUT=${NUM_ROLLOUT:-${_NROLLOUT}}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-16384}
# Total budget for one trajectory: prompt + every turn's generation + every tool
# observation. Set explicitly -- generate_with_retool otherwise derives it from
# context_parallel_size * max_tokens_per_gpu, which couples a semantic limit to a
# GPU-memory knob. Must stay <= MAX_TOKENS_PER_GPU * CP_SIZE, or a full-length
# sequence lands in a microbatch that overruns the budget (the first-fit packer in
# slime/utils/data.py does not check, so it OOMs mid-run rather than failing fast).
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-16384}

# --- Rollout concurrency ----------------------------------------------------
# THIS is the knob that keeps the engines alive, not the memory fraction.
#
# slime fires rollout_batch_size x n_samples_per_prompt requests at once (512
# here) behind a semaphore of sglang_server_concurrency x num_engines
# (sglang_rollout.py:94). The default 512 makes that semaphore effectively
# unlimited: every request is submitted immediately. But an engine's KV pool only
# holds capacity/rollout_max_context_len full-length sequences. Submit many more
# than that and the engine sits permanently at "KV cache pool is full. Retract
# requests."; every retract is followed by a re-prefill burst whose transient
# allocation has to fit outside the static pool, and eventually one does not --
# which is how a 40 GB engine died after 1.5 hours of continuous retraction,
# *with* the memory fraction already lowered.
#
# So cap in-flight requests per engine at ~0.8 x (KV capacity / context len).
# Lowering the memory fraction is the wrong axis: it shrinks the KV pool and
# makes retraction worse. Derived rather than hardcoded so that a retriever on
# the cards, a different model, or a different context length all flow through:
#
#   pool bytes = frac x card bytes x GPUs-per-engine - bf16 weights
#   capacity   = pool bytes / (context length x KV bytes per token)
#
# The 40 (80 GB) and 16 (40 GB) this used to hardcode for Qwen3-4B at 16384 are
# what the formula gives to within rounding.
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
# slime's own default; kept derived so it cannot drift from the two values above.
# slime >=0.3.0 dropped --use-dynamic-global-batch-size: one optimizer step per
# rollout is now the unconditional default, and passing --num-steps-per-rollout is
# the only way to get more (it then asserts gbs == rollout_batch_size *
# n_samples_per_prompt // num_steps_per_rollout, which this value already satisfies).
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# NOTE: slime also saves at every epoch boundary and after the final rollout,
# regardless of this interval (slime/utils/misc.py: should_run_periodic_action).
# One Qwen3-4B checkpoint is ~56 GB (8 GB weights + 48 GB optimizer state).
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
LR=${LR:-1e-6}
# Stop each turn at the end of a tool call (ReTool multi-turn).
# Auto-resume: --load points at SAVE_DIR, so an interrupted run continues from
# its latest checkpoint (empty on the first run -> starts from HF_CHECKPOINT).
LOAD=${LOAD:-${SAVE_DIR}}
# Save raw rollouts (the replayed-prefix pool for distillation) here.
ROLLOUT_DEBUG_DIR=${ROLLOUT_DEBUG_DIR:-${SAVE_DIR}/rollout_debug}

MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
# ReTool Python tool sandbox.
export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-subprocess}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-600}
# 32, not 128: the sandbox runs each call through asyncio.to_thread, whose default
# executor caps at min(32, cpu+4). Extra permits only queue while still holding the
# sample's SGLang slot. This is also tool_sandbox.py's own default and the paper's.
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-128}
export TOOL_SANDBOX_MAX_TURNS=${TOOL_SANDBOX_MAX_TURNS:-16}
export TOOL_SANDBOX_MAX_TOOL_CALLS=${TOOL_SANDBOX_MAX_TOOL_CALLS:-16}
# Wall-clock cap for one sample's whole multi-turn rollout (generate_with_retool).
# Last line of defence only: recovery is supposed to come from
# ROUTER_REQUEST_TIMEOUT_SECS cutting a hung request so _post can retry, so this must
# stay comfortably ABOVE that value. See the note there for why both numbers grew.
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
   # no --rollout-stop: the ReTool rollout puts </tool_call> in the stop list itself
   # (generate_with_retool.py). The launcher flag never reached SGLang intact.
)

# No in-training eval — accuracy is measured separately with scripts/eval/math.sh.
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
   # Megatron derives lr_decay_steps from num_rollout
   # (model.py:204-207: train_iters = num_rollout * rollout_batch_size *
   # n_samples_per_prompt // global_batch_size, then x global_batch_size) and
   # OptimizerParamScheduler.load_state_dict asserts the reloaded value matches
   # the one in the checkpoint. Raising num_rollout and resuming therefore aborts
   # with "class input value N and checkpoint value M ... do not match", even
   # though under a constant schedule lr_decay_steps never reaches get_lr() and
   # the learning rate is identical either way. This flag tells Megatron to take
   # the scheduler's configuration from the arguments instead of the checkpoint,
   # which is what we mean: the LR is a constant we set here. Do NOT copy it to
   # cold_start.sh -- that one decays cosine, where the value is load-bearing.
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
   # Bound the prefill peak. Left unset, SGLang sizes the chunk generously and a
   # late-trajectory ReTool prefill (10-16k tokens of accumulated history) needs a
   # logits buffer of chunk x vocab_shard x 2 bytes. At TP=2 on Qwen3-4B that is
   # ~1.6 GB for an 11k-token chunk -- precisely the allocation that OOM'd the
   # 40 GB engines. 4096 caps it near 0.6 GB and costs a little prefill latency.
   --sglang-chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}"
   # Per-engine cap on in-flight requests; see the Rollout concurrency block.
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   # Bound how long a single /generate may hang. slime builds its HTTP client with
   # httpx.Timeout(None) and only retries on an exception, so without this a lost
   # response stalls the whole rollout. Must stay below GENERATE_SAMPLE_TIMEOUT, or
   # the sample gives up first and _post never sees the exception that triggers its
   # retry.
   #
   # This clock starts when the request is *submitted*, so queueing counts. With
   # rollout_batch_size 64 x 8 samples we fire 512 requests at roughly 150 concurrent
   # slots (16384-token sequences, ~147 KB of KV per token for Qwen3-4B), so the back
   # of the queue waits several full waves. 90s was calibrated against 256 requests at
   # 9200 tokens and is nowhere near enough here -- it made every wave time out, which
   # tripped the router's circuit breaker (cb_failure_threshold=10) and turned into a
   # sustained no_available_workers storm amplified by _post's own retries.
   --sglang-router-request-timeout-secs "${ROUTER_REQUEST_TIMEOUT_SECS:-600}"
   # NOTE: the router circuit breaker no longer needs a flag. slime >=0.3.0 sets
   # router_args.disable_circuit_breaker = True unconditionally in
   # slime/ray/rollout.py ("Timeouts are transient ... and do not indicate a dead
   # server"), which is exactly the behaviour the old flag bought us.
)

WANDB_ARGS=()
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-agentic-opd}
WANDB_GROUP=${WANDB_GROUP:-grpo-math-${MODEL}}
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
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# Raised before `ray start` so every Ray worker inherits it -- the sglang router runs
# inside the RolloutManager actor and holds one socket per in-flight request. With
# rollout_batch_size 64 x 8 samples that is up to 512 connections fanning out to the
# engines, on top of the tool sandbox's subprocess pipes. Past the limit the router
# fails with "error sending request for url", surfacing to slime as a 500 with
# code=call_upstream_request_error. The opd_* launchers already do this.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
ulimit -u 65536 2>/dev/null || true
echo "[teacher_grpo] ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Note: no PYTORCH_CUDA_ALLOC_CONF below. --colocate turns on offload_train and
# offload_rollout, both of which drive torch_memory_saver, and it refuses to
# initialize when expandable_segments is enabled.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SRC_DIR}:${RELEASE_DIR}:${REPO_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TOOL_SANDBOX_BACKEND\": \"${TOOL_SANDBOX_BACKEND}\",
    \"TOOL_SANDBOX_JUPYTER_TIMEOUT\": \"${TOOL_SANDBOX_JUPYTER_TIMEOUT}\",
    \"TOOL_SANDBOX_CONCURRENCY\": \"${TOOL_SANDBOX_CONCURRENCY}\",
    \"TOOL_SANDBOX_MAX_TURNS\": \"${TOOL_SANDBOX_MAX_TURNS}\",
    \"TOOL_SANDBOX_MAX_TOOL_CALLS\": \"${TOOL_SANDBOX_MAX_TOOL_CALLS}\",
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
