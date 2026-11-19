#!/bin/bash
# Stage 2 (teacher GRPO), search (Search-R1): RL-train the teacher, initialized
# from the stage-1 cold-start model. The on-policy rollouts saved here become the
# replayed-prefix pool for student distillation (stage 3).
#
# This single entry point (like scripts/eval/search.sh):
#   1. Starts the local dense retrieval servers (e5 + wiki-18) and waits for them.
#   2. Runs teacher GRPO with those retrievers as the search backend.
#   3. Shuts the retrievers down on exit.
#
# Prereq: re-materialize the cold-start checkpoint into a clean HF + torch_dist
# pair (see README "Teacher GRPO"): convert_torch_dist_to_hf.sh then
# convert_hf_to_torch_dist.sh; point HF_CHECKPOINT / REF_LOAD at them.
#
# RL prompts (NQ + HotpotQA):
#   hf download baohao/search-train --repo-type dataset --local-dir /path/to/rl_data
#
# DATA_DIR must hold e5_Flat.index and wiki-18.jsonl; get them with
# (see docs/search.md#retriever-index-and-corpus; needs ~132 GB):
#   python eval/search/local_dense_retriever/download.py --save_path $DATA_DIR
#   cat $DATA_DIR/part_* > $DATA_DIR/e5_Flat.index
#   gzip -d $DATA_DIR/wiki-18.jsonl.gz
#
# Pick the base model with MODEL (default qwen3-4b), which sets the slime config,
# parallelism and rollout engine size:
#   qwen3-4b -> qwen3-4B-Instruct-2507.sh  TP2  2 GPUs/engine
# The SGLang memory fraction and rollout concurrency are derived from the card
# and the retriever footprint rather than presets -- see the GPUs note below.
#
# Usage:
#   MODEL=qwen3-4b \
#   HF_CHECKPOINT=/workspace/ckpt/sft_hf REF_LOAD=/workspace/ckpt/sft_torch_dist   (the defaults) \
#   PROMPT_DATA=/path/to/rl_data/train.parquet \
#   DATA_DIR=/path/to/search_data \
#   SAVE_DIR=/path/to/teacher_run \
#       bash scripts/train/teacher_grpo_search.sh
#
# Already have retrievers running? Skip launching them:
#   START_RETRIEVER=0 bash ...
#   (SEARCH_URL defaults to http://127.0.0.1:8000/retrieve, which under --network
#   host is the retriever on the VM. Its per-GPU footprint is measured off
#   nvidia-smi at launch; RETRIEVER_RESERVE_GIB overrides that number.)
#
# GPUs (8x A100 40GB): under ASYNC=1 the trainer owns the first
# ACTOR_NUM_GPUS_PER_NODE cards and the rollout engines the rest
# (placement_group.py:117,127), and by default one retrieval server is sharded
# across the rollout cards only -- an index shard on a training card would
# compete with Megatron activations and the reloaded optimizer state, which no
# SGLang fraction can protect against.
#
# retrieval_server.py:212-218 clones the flat e5 index to GPU as fp16 with
# co.shard=True, so it costs about RETRIEVER_INDEX_GIB / (GPUs in the group) per
# GPU plus the encoder. SGLang sizes its static pool as a fraction of *total*
# card memory, not free memory, so that share is subtracted explicitly below
# rather than hardcoded into a fraction; the script prints every number it used.
# Check them against `nvidia-smi` on a warm run.
#
# No in-training eval; final accuracy is measured separately with
# scripts/eval/search.sh. Any preset value can be overridden, e.g. TP_SIZE=4.

set -ex

export PYTHONUNBUFFERED=1
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
RELEASE_DIR="${REPO_DIR}/train"
SRC_DIR="${RELEASE_DIR}/src"
SLIME_DIR="${REPO_DIR}/third_party/slime"
# TIS (truncated-importance-sampling) helper ships in the slime examples.
TIS_DIR="${SLIME_DIR}/examples"
MIS_CONFIG="${TIS_DIR}/train_infer_mismatch_helper/mis.yaml"

# ===========================================================================
# KEY SETTINGS — what you most likely need to change.
# ===========================================================================

# --- Model preset (config + parallelism + rollout engine / memory) ----------
# The preset carries the parallelism and rollout settings we used.
#
# _KVTOK is bytes of KV cache per token for the whole model and _WGIB is bf16
# weights per engine; both feed the rollout-concurrency derivation further down.
# Qwen3-4B: 36 layers x 8 KV heads x 128
# head_dim x 2 (K and V) x 2 bytes = 147456.
MODEL=${MODEL:-qwen3-4b}
case "${MODEL}" in
    qwen3-4b)
        _MODEL_CONFIG="qwen3-4B-Instruct-2507.sh"
        _TP=2; _EP=1; _OFFLOAD=0; _ENGINE=2; _MEM=0.6; _MAXTOK=9216
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
PROMPT_DATA=${PROMPT_DATA:?"set PROMPT_DATA to the RL prompt parquet (baohao/search-train)"}

# --- Retriever --------------------------------------------------------------
# Directory with e5_Flat.index and wiki-18.jsonl (see docs/search.md).
DATA_DIR=${DATA_DIR:-}
# Set to 0 if the retrievers are already running (then also set SEARCH_URL).
# Default off: the retriever is a long-lived service on the host here, shared by
# every arm and outliving any single run, so a script that starts its own would
# collide with it on the ports and on GPU memory. START_RETRIEVER=1 makes this
# script launch and own one.
START_RETRIEVER=${START_RETRIEVER:-0}
# Comma-separated retriever URLs; auto-derived from RETRIEVER_PORTS when empty.
SEARCH_URL=${SEARCH_URL:-}
# The retriever runs in its own conda env (faiss-gpu etc., see docs/search.md).
# Set RETRIEVER_CONDA_ENV="" to reuse the current environment instead.
RETRIEVER_CONDA_ENV=${RETRIEVER_CONDA_ENV-retriever}
RETRIEVER_PATH=${RETRIEVER_PATH:-intfloat/e5-base-v2}
SEARCH_R1_MAX_TURNS=${SEARCH_R1_MAX_TURNS:-8}
# Ceiling on ONE turn's generation, so a turn that never closes its tool call
# cannot run the whole context in a single /generate and time the router out.
SEARCH_R1_MAX_TOKENS_PER_TURN=${SEARCH_R1_MAX_TOKENS_PER_TURN:-4096}
# RETRIEVER_GPU_GROUPS / RETRIEVER_PORTS are derived after the ASYNC block below,
# since the default placement depends on which cards the rollout engines get.

# --- GPUs / parallelism (defaults from the MODEL preset) --------------------
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-${_TP}}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-${_EP}}
ETP_SIZE=${ETP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-${_MAXTOK}}
# --- ASYNC ------------------------------------------------------------------
# ASYNC=1 (the default) runs train_async.py: rollout N+1 is generated while step
# N trains, so the rollout GPUs keep decoding through the backward pass. This is
# one-step overlap, not fully async -- every batch is exactly one update stale
# and the per-rollout barrier stays (train_async.py:36 waits for the whole
# batch). train_async.py:11 forbids --colocate, so the pool is split into a
# dedicated training pool and a dedicated rollout pool instead of alternating
# between the two roles on all of them.
#
# --use-rollout-logprobs is required, not cosmetic: the policy ratio has to be
# formed against the logprobs of the weights that actually sampled the tokens,
# not ones recomputed from the newer weights (which would pin the ratio at 1 and
# leave the staleness uncorrected).
ASYNC=${ASYNC:-1}
if [ "${ASYNC}" = "1" ]; then
   ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-2}
   ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-$((NUM_GPUS - ACTOR_NUM_GPUS_PER_NODE))}
   # 2 training GPUs at TP=2 means DP=1, and the distributed optimizer cannot
   # shard at DP=1: the fp32 Adam state alone would be ~24 GB/GPU for a 4B model.
   OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-1}
   if [ $((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS)) -ne "${NUM_GPUS}" ]; then
      echo "ACTOR_NUM_GPUS_PER_NODE(${ACTOR_NUM_GPUS_PER_NODE}) + ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS})" >&2
      echo "must equal NUM_GPUS(${NUM_GPUS})." >&2
      exit 1
   fi
else
   ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}
   ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-${NUM_GPUS}}
fi

OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-${_OFFLOAD}}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${_ENGINE}}

# --- Preflight: the presets have to fit the pools they are launched into -----
# TP x PP x CP must divide the training pool, and the rollout pool must split
# into whole engines. A TP_SIZE that the default 2-GPU async training pool
# cannot hold would otherwise only fail in Megatron, after Ray, the retriever
# and the rollout engines had all come up.
_MODEL_PAR=$((TP_SIZE * PP_SIZE * CP_SIZE))
if [ $((${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}} % _MODEL_PAR)) -ne 0 ]; then
   echo "[preflight] MODEL=${MODEL} needs TP x PP x CP = ${_MODEL_PAR} GPUs per model replica, but the" >&2
   echo "[preflight] training pool is ${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}} GPU(s) (ACTOR_NUM_GPUS_PER_NODE). Give it a multiple of ${_MODEL_PAR}, e.g." >&2
   echo "[preflight]   ACTOR_NUM_GPUS_PER_NODE=${_MODEL_PAR} bash $(basename "$0")" >&2
   exit 1
fi
if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
   echo "[preflight] ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} is not a multiple of ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE};" >&2
   echo "[preflight] $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) rollout GPU(s) would sit idle. Adjust one of them." >&2
   exit 1
fi

# --- Retriever placement ----------------------------------------------------
# Default: ONE retrieval server, sharded across the rollout GPUs. Two decisions,
# both about memory rather than throughput:
#
#   * Rollout GPUs, not all 8, so no index shard lands on a training card (see
#     the GPUs note in the header). Under ASYNC=0 the pools are the same 8 cards
#     anyway, so the group is all of them.
#   * One group, not two. Sharding one index across N GPUs costs INDEX/N per
#     GPU; two servers across N/2 each replicate the index and cost twice that.
#     The old 2x4 default predates the async split.
#
# If retrieval latency rather than memory turns out to be the bottleneck -- one
# uvicorn worker serving search_concurrency=256 -- split it back and pay the
# extra INDEX/N per GPU:
#   RETRIEVER_GPU_GROUPS="2,3,4 5,6,7" RETRIEVER_PORTS="8000 8001" bash ...
#
# Built with a loop rather than `seq -s,`: BSD seq appends a trailing separator
# and GNU seq does not, and a trailing comma would both confuse the group-size
# count below and hand CUDA_VISIBLE_DEVICES an empty device.
_gpu_range() {
   local first=$1 last=$2 i out=""
   for ((i = first; i <= last; i++)); do
      out="${out:+${out},}${i}"
   done
   echo "${out}"
}
if [ "${ASYNC}" = "1" ]; then
   _ROLLOUT_GPU_LIST=$(_gpu_range "${ACTOR_NUM_GPUS_PER_NODE}" $((NUM_GPUS - 1)))
else
   _ROLLOUT_GPU_LIST=$(_gpu_range 0 $((NUM_GPUS - 1)))
fi
RETRIEVER_GPU_GROUPS=${RETRIEVER_GPU_GROUPS:-${_ROLLOUT_GPU_LIST}}
RETRIEVER_PORTS=${RETRIEVER_PORTS:-"8000"}

# --- How much room the retriever needs on each GPU it shares with SGLang ----
# The flat e5 index over wiki-18 is ~27 GiB once cloned to GPU as fp16
# (retrieval_server.py:213 co.useFloat16) and it is sharded, not replicated,
# across the group (retrieval_server.py:214 co.shard). So per GPU:
#
#     RETRIEVER_INDEX_GIB / (GPUs in the smallest group)  +  encoder + workspace
#
# docs/search.md quotes 5-7 GB per GPU, which is this formula for an 8-way
# shard. The encoder alone (e5-base-v2, 110M params) is only 1-2 GiB --
# that is the number to use if you move the index off the GPU by dropping
# --faiss_gpu, which we do not do here because a brute-force CPU scan over 21M
# passages is far too slow to keep the rollout fed.
#
# Override RETRIEVER_RESERVE_GIB to skip the derivation entirely.
RETRIEVER_INDEX_GIB=${RETRIEVER_INDEX_GIB:-27}
RETRIEVER_ENCODER_GIB=${RETRIEVER_ENCODER_GIB:-2}

_retriever_min_group_size() {
   local groups group n min=0
   read -r -a groups <<< "${RETRIEVER_GPU_GROUPS}"
   for group in "${groups[@]}"; do
      n=$(awk -F, '{print NF}' <<< "${group}")
      if [ "${min}" -eq 0 ] || [ "${n}" -lt "${min}" ]; then min=${n}; fi
   done
   echo "${min}"
}

if [ -n "${RETRIEVER_RESERVE_GIB:-}" ]; then
   echo "[retriever] reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (RETRIEVER_RESERVE_GIB given)"
elif [ "${START_RETRIEVER}" = "1" ] && [ "$(_retriever_min_group_size)" -gt 0 ]; then
   # We start it, so we know the sharding: index / N + encoder.
   _RGROUP_SIZE=$(_retriever_min_group_size)
   RETRIEVER_RESERVE_GIB=$(awk -v idx="${RETRIEVER_INDEX_GIB}" -v n="${_RGROUP_SIZE}" \
                               -v enc="${RETRIEVER_ENCODER_GIB}" \
                               'BEGIN { printf "%d", int(idx / n + enc + 0.999) }')
   echo "[retriever] will start it on groups='${RETRIEVER_GPU_GROUPS}' -> reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (index/${_RGROUP_SIZE} + encoder)"
else
   # Something else started it -- the default: a retriever in a screen on the
   # VM. Its footprint is already on the cards, so measure instead of guessing:
   # the most-used GPU right now, before anything of ours launches, plus half a
   # GiB for the encoder's activations under a query burst, rounded up. That is
   # GPU 0 in practice, which carries the encoder on top of its index shard. A
   # clean node reads 0, so nothing is wasted when no retriever is up; a
   # leftover engine from a crashed run is accounted for the same way -- it is
   # occupying the memory whether we like it or not.
   _GPU_USED_MAX_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -1)
   RETRIEVER_RESERVE_GIB=$(awk -v used="${_GPU_USED_MAX_MIB:-0}" \
                               'BEGIN { printf "%d", (used < 256) ? 0 : int((used + 512) / 1024 + 0.999) }')
   echo "[retriever] most-used GPU before launch: ${_GPU_USED_MAX_MIB:-unknown} MiB -> reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (override with RETRIEVER_RESERVE_GIB)"
fi

# --- SGLang static fraction, sized from the card AND the retriever -----------
# Baseline first. 0.85 of a 40 GB A100 reserves ~33.6 GB for weights + KV and
# leaves only ~6 GB for CUDA graphs, prefill activations, NCCL buffers and
# fragmentation. That is not enough, and it fails *late*: on the math runs a
# rollout engine ran 9.5 hours and then died trying to allocate 1.56 GB with
# 1.51 GB free. On an 80 GB card the same fraction leaves ~12 GB and is fine.
_GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${ASYNC}" != "1" ]; then
   # Colocate: the trainer time-shares these same cards, so the baseline is the
   # model preset's much lower fraction, not the async one.
   _MEM_BASE=${_MEM}
elif [ -n "${_GPU_MEM_MIB}" ] && [ "${_GPU_MEM_MIB}" -ge 49152 ] 2>/dev/null; then
   _MEM_BASE=0.85
else
   _MEM_BASE=0.78
fi
# Then subtract the retriever's share of the *total* card, which is the
# denominator SGLang's fraction is measured against.
_MEM_DERIVED=$(awk -v base="${_MEM_BASE}" -v gib="${RETRIEVER_RESERVE_GIB}" \
                 -v mib="${_GPU_MEM_MIB:-40960}" 'BEGIN {
   f = base - (gib * 1024.0) / mib;
   if (f < 0.30) f = 0.30;
   printf "%.3f", f;
}')
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-${_MEM_DERIVED}}
echo "[gpu] memory.total=${_GPU_MEM_MIB:-unknown} MiB, base ${_MEM_BASE}, minus ${RETRIEVER_RESERVE_GIB} GiB retriever"
echo "[gpu] -> sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}"

# ===========================================================================
# ADVANCED SETTINGS — usually fine as-is.
# ===========================================================================

# Rollout / GRPO knobs.
NUM_ROLLOUT=${NUM_ROLLOUT:-500}
# 64 x 8, the same shape teacher_grpo_math.sh uses, so a mixed math+search run
# draws the same number of prompts and the same GRPO group size in both
# environments. A group of 8 also gives the group-mean baseline half the
# variance a group of 4 does, which matters more here than in math because the
# exact-match reward is coarser.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
# Not a real generation cap -- the binding limit is ROLLOUT_MAX_CONTEXT_LEN, and
# the turn loop already clamps every turn to what is left of it
# (generate_with_search.py:511-517). This is what slime's
# raw_response_length/response_length_clip_ratio metric compares against, so it
# has to be the honest total-response budget: context minus a prompt of a few
# hundred tokens. At 4096 the metric would read ~1.0 from the first step, since
# eight turns of three retrieved passages alone run past 4000 tokens.
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
# Stated explicitly rather than left to default. generate_with_search.py:470-474
# falls back to context_parallel_size * max_tokens_per_gpu when this is unset,
# which is the same 9216 -- but it would silently track MAX_TOKENS_PER_GPU, and
# the concurrency derivation below needs the real per-request context bound.
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-9216}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
# NOTE: slime also saves at every epoch boundary and after the final rollout,
# regardless of this interval (slime/utils/misc.py: should_run_periodic_action).
# One Qwen3-4B checkpoint is ~56 GB (8 GB weights + 48 GB optimizer state).
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
LR=${LR:-1e-6}
# TIS (truncated importance sampling) corrects train/infer mismatch.
#
# Under ASYNC=1 the mismatch is already handled by --use-rollout-logprobs, which
# forms the ratio directly against the sampling weights' logprobs, and slime
# forbids both at once (arguments.py:1795-1796: "use_rollout_logprobs and use_tis
# cannot be set at the same time"). So TIS defaults off under async and on under
# sync, where --use-rollout-logprobs is not passed.
if [ "${ASYNC}" = "1" ]; then
   USE_TIS=${USE_TIS:-0}
else
   USE_TIS=${USE_TIS:-1}
fi
if [ "${ASYNC}" = "1" ] && [ "${USE_TIS}" = "1" ]; then
   echo "USE_TIS=1 cannot be combined with ASYNC=1: slime asserts that use_tis and" >&2
   echo "use_rollout_logprobs are mutually exclusive. Set USE_TIS=0 or ASYNC=0." >&2
   exit 1
fi
# Save raw rollouts (the replayed-prefix pool for distillation) here.
# Auto-resume: --load points at SAVE_DIR, so an interrupted run continues from
# its latest checkpoint (empty on the first run -> starts from HF_CHECKPOINT).
# teacher_grpo_math.sh has always done this; this script did not, so re-running
# the same command after a stop silently restarted training from step 0.
#
# Pass an iter_XXXXXXX path here and Megatron loads the weights but drops the
# optimizer state and restarts the rollout counter at 0; to resume one specific
# iteration, point LOAD at the parent directory and set CKPT_STEP.
LOAD=${LOAD:-${SAVE_DIR}}
CKPT_STEP=${CKPT_STEP:-}
ROLLOUT_DEBUG_DIR=${ROLLOUT_DEBUG_DIR:-${SAVE_DIR}/rollout_debug}

# --- Rollout concurrency ----------------------------------------------------
# THIS is the knob that keeps the engines alive, not the memory fraction.
#
# slime fires rollout_batch_size x n_samples_per_prompt requests at once behind a
# semaphore of sglang_server_concurrency x num_engines (sglang_rollout.py:94).
# The default 512 makes that semaphore effectively unlimited: every request is
# submitted immediately. But an engine's KV pool only holds
# capacity/rollout_max_context_len full-length sequences. Submit many more than
# that and the engine sits permanently at "KV cache pool is full. Retract
# requests."; every retract is followed by a re-prefill burst whose transient
# allocation has to fit outside the static pool, and eventually one does not --
# which is how a 40 GB engine died after 1.5 hours of continuous retraction on
# the math runs, *with* the memory fraction already lowered.
#
# So cap in-flight requests per engine at ~0.8 x (KV capacity / context len).
# Lowering the memory fraction is the wrong axis: it shrinks the KV pool and
# makes retraction worse -- and here the retriever has already taken a bite out
# of that pool, which is why this is derived rather than hardcoded.
#
#   pool bytes = frac x card bytes x GPUs-per-engine - bf16 weights
#   capacity   = pool bytes / (context length x KV bytes per token)
#
# The math scripts' hardcoded 40 (80 GB) and 16 (40 GB) are this formula at
# context 16384; it reproduces them to within rounding.
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

# Retriever knobs.
RETRIEVER_CONDA_SH=${RETRIEVER_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
INDEX_FILE=${INDEX_FILE:-${DATA_DIR}/e5_Flat.index}
CORPUS_FILE=${CORPUS_FILE:-${DATA_DIR}/wiki-18.jsonl}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_WAIT_HOST=${RETRIEVER_WAIT_HOST:-127.0.0.1}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}
RETRIEVER_WAIT_TIMEOUT=${RETRIEVER_WAIT_TIMEOUT:-1800}
LOG_DIR=${LOG_DIR:-${SAVE_DIR}/logs}
# Kill the retrievers we started when this script exits.
CLEANUP_ON_EXIT=${CLEANUP_ON_EXIT:-1}

MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# ===========================================================================
# No user-editable settings below this line.
# ===========================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

mkdir -p "${SAVE_DIR}" "${ROLLOUT_DEBUG_DIR}" "${LOG_DIR}"
export WANDB_DIR="${SAVE_DIR}"

# --- retriever lifecycle ----------------------------------------------------
PIDS=()

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ "${CLEANUP_ON_EXIT}" = "1" ]; then
        for pid in "${PIDS[@]:-}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                kill "${pid}" >/dev/null 2>&1 || true
            fi
        done
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

wait_for_tcp() {
    local host=$1 port=$2 timeout=$3 name=$4 pid=${5:-}
    local deadline=$((SECONDS + timeout))
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
            echo "${name} is reachable at ${host}:${port}"
            return 0
        fi
        if [ -n "${pid}" ] && ! kill -0 "${pid}" >/dev/null 2>&1; then
            echo "${name} process ${pid} exited before ${host}:${port} became reachable" >&2
            return 1
        fi
        sleep 5
    done
    echo "Timed out waiting for ${name} at ${host}:${port} after ${timeout}s" >&2
    return 1
}

join_retriever_urls() {
    local urls=() port
    for port in "$@"; do
        urls+=("http://${RETRIEVER_WAIT_HOST}:${port}/retrieve")
    done
    local IFS=,
    echo "${urls[*]}"
}

start_retrievers() {
    if [ ! -f "${INDEX_FILE}" ] || [ ! -f "${CORPUS_FILE}" ]; then
        echo "Retriever index/corpus not found (${INDEX_FILE}, ${CORPUS_FILE})." >&2
        echo "Set DATA_DIR to the dir holding e5_Flat.index and wiki-18.jsonl; see" >&2
        echo "docs/search.md for how to download and assemble them, or set" >&2
        echo "INDEX_FILE/CORPUS_FILE directly (or pass START_RETRIEVER=0)." >&2
        exit 1
    fi
    read -r -a gpu_groups <<< "${RETRIEVER_GPU_GROUPS}"
    read -r -a ports <<< "${RETRIEVER_PORTS}"
    if [ "${#gpu_groups[@]}" -ne "${#ports[@]}" ]; then
        echo "RETRIEVER_GPU_GROUPS and RETRIEVER_PORTS must have the same number of entries" >&2
        exit 1
    fi
    local retriever_pids=()
    for i in "${!gpu_groups[@]}"; do
        local gpus="${gpu_groups[$i]}" port="${ports[$i]}"
        local log_file="${LOG_DIR}/retrieval_server_gpus_${gpus//,/}_port_${port}.log"
        echo "Starting retrieval server on GPUs ${gpus}, port ${port}; log: ${log_file}"
        (
            if [ -n "${RETRIEVER_CONDA_ENV}" ]; then
                source "${RETRIEVER_CONDA_SH}"
                conda activate "${RETRIEVER_CONDA_ENV}"
                export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
            fi
            export CUDA_LAUNCH_BLOCKING=1
            exec env CUDA_VISIBLE_DEVICES="${gpus}" python "${REPO_DIR}/eval/search/local_dense_retriever/retrieval_server.py" \
                --index_path "${INDEX_FILE}" \
                --corpus_path "${CORPUS_FILE}" \
                --topk "${RETRIEVER_TOPK}" \
                --retriever_name "${RETRIEVER_NAME}" \
                --retriever_model "${RETRIEVER_PATH}" \
                --faiss_gpu \
                --port "${port}"
        ) >"${log_file}" 2>&1 &
        local pid=$!
        PIDS+=("${pid}")
        retriever_pids+=("${pid}")
    done
    for i in "${!ports[@]}"; do
        wait_for_tcp "${RETRIEVER_WAIT_HOST}" "${ports[$i]}" "${RETRIEVER_WAIT_TIMEOUT}" "retriever" "${retriever_pids[$i]}"
    done
    if [ -z "${SEARCH_URL}" ]; then
        SEARCH_URL="$(join_retriever_urls "${ports[@]}")"
    fi
}

if [ "${START_RETRIEVER}" = "1" ]; then
    start_retrievers
elif [ -z "${SEARCH_URL}" ]; then
    read -r -a ports <<< "${RETRIEVER_PORTS}"
    SEARCH_URL="$(join_retriever_urls "${ports[@]}")"
fi
echo "Using SEARCH_URL=${SEARCH_URL}"

# With the default off, "forgot to start the retriever" is the likely mistake --
# and it does not crash anything. generate_with_search catches the connection
# error and hands the model "[ERROR] ..." as if it were a search result, so the
# run trains to completion on transcripts that never retrieved a thing, with the
# reward sitting at the closed-book floor and nothing in the logs saying why.
# Ask the retriever one question here instead.
if [ "${START_RETRIEVER}" != "1" ]; then
   _retriever_ok=0
   for _url in ${SEARCH_URL//,/ }; do
      if curl -sf -m 10 -X POST "${_url}" -H 'Content-Type: application/json' \
              -d '{"queries":["who wrote pride and prejudice"],"topk":1,"return_scores":false}' \
              >/dev/null 2>&1; then
         _retriever_ok=1
      else
         echo "[retriever] not answering: ${_url}" >&2
      fi
   done
   if [ "${_retriever_ok}" != "1" ]; then
      echo "[retriever] nothing in SEARCH_URL answered a query. Start one with" >&2
      echo "[retriever]   bash scripts/eval/start_retriever.sh" >&2
      echo "[retriever] or pass START_RETRIEVER=1 to have this script launch its own." >&2
      exit 1
   fi
fi

source "${MODEL_CONFIG}"

_has_megatron_ckpt() { [ -n "${1:-}" ] && [ -f "${1}/latest_checkpointed_iteration.txt" ]; }
if [ -n "${LOAD}" ] && ! _has_megatron_ckpt "${LOAD}"; then
    echo "[grpo] LOAD=${LOAD} has no latest_checkpointed_iteration.txt -- starting from" >&2
    echo "[grpo] HF_CHECKPOINT instead of resuming. Expected on the first run." >&2
    LOAD=""
fi
if [ -n "${LOAD}" ]; then
    echo "[grpo] resuming from ${LOAD} (iteration $(cat "${LOAD}/latest_checkpointed_iteration.txt" 2>/dev/null))"
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)
if [ -n "${LOAD}" ]; then
   CKPT_ARGS+=(--load "${LOAD}")
   [ -n "${CKPT_STEP}" ] && CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
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

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
if [ "${USE_TIS}" = "1" ]; then
   GRPO_ARGS+=(--use-tis)
fi

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
   # late-turn Search-R1 prefill (question + up to SEARCH_R1_MAX_TURNS x topk
   # retrieved passages) needs a logits buffer of chunk x vocab_shard x 2 bytes.
   # 4096 caps it near 0.6 GB at TP=2 on Qwen3-4B and costs a little latency --
   # cheaper insurance here than in math, since the retriever has already taken
   # its cut of what is left outside the static pool.
   --sglang-chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}"
   # Per-engine cap on in-flight requests; see the Rollout concurrency block.
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   # Bound how long a single /generate may hang. slime builds its HTTP client
   # with httpx.Timeout(None) and only retries on an exception, so without this a
   # lost response stalls the whole rollout (the router default is 4 hours).
   #
   # 600, the same as the math arms. The clock starts when the request is
   # *submitted*, so queueing counts, and one sample issues up to
   # SEARCH_R1_MAX_TURNS sequential /generate calls -- roughly 4k calls per
   # rollout against a few dozen concurrent slots. At 90 the back of the queue
   # timed out routinely; the router dropped the connection, the engine logged
   # "Request is disconnected from the client", finished the decode anyway and
   # threw it away, and post() (max_retries=60, 1s backoff) paid for the whole
   # thing again.
   --sglang-router-request-timeout-secs "${ROUTER_REQUEST_TIMEOUT_SECS:-600}"
)

WANDB_ARGS=()
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-agentic-opd}
WANDB_GROUP=${WANDB_GROUP:-grpo-search-${MODEL}}
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
   # Rollout-shape logging (response lengths, clip ratio). data.py:436 only reads
   # loss_masks and round_number, both harmless when absent.
   --log-multi-turn
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
if [ "${USE_TIS}" = "1" ]; then
   CUSTOM_ARGS+=(
      --custom-config-path "${MIS_CONFIG}"
      --custom-tis-function-path train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
   )
fi

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# See teacher_grpo_math.sh: the sglang router holds one socket per in-flight request
# and runs inside a Ray actor, so the limit has to be raised before `ray start`.
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
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SRC_DIR}:${RELEASE_DIR}:${REPO_DIR}:${TIS_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"SEARCH_URL\": \"${SEARCH_URL}\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_R1_MAX_TURNS}\",
    \"SEARCH_R1_MAX_TOKENS_PER_TURN\": \"${SEARCH_R1_MAX_TOKENS_PER_TURN}\"
  }
}"

if [ "${ASYNC}" = "1" ]; then
   ENTRYPOINT=train_async.py
   PARALLEL_ARGS=(--use-rollout-logprobs)
else
   ENTRYPOINT=train.py
   PARALLEL_ARGS=(--colocate)
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir="${SLIME_DIR}" \
   -- python3 "${ENTRYPOINT}" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   "${PARALLEL_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
