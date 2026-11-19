#!/bin/bash
# The single entry point for search (Search-R1) evaluation:
#   1. Start local dense retrieval servers (e5 + wiki-18, Search-R1 style).
#   2. Convert Megatron torch_dist checkpoints to HF format (skipped for
#      checkpoints that are already HF, i.e. contain config.json).
#   3. Serve each HF checkpoint with SGLang.
#   4. Run eval/search/eval.py (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki,
#      Musique, Bamboogle — whatever is in DATASET).
#
# DATASET defaults to the merged test set (baohao/search-test), downloaded
# automatically; override with a local parquet/jsonl or another HF name.
#
# Usage:
#   # A ready HF checkpoint:
#   MODEL_PATH=/path/to/hf_model \
#   DATA_DIR=/path/to/search_data \
#       bash scripts/eval/search.sh
#
#   # Megatron torch_dist checkpoints from a training run (the retriever already up):
#   RUN_DIR=/workspace/runs/grpo_search bash scripts/eval/search.sh iter_0000199 iter_0000299
#
# Checkpoint arguments may be absolute directories or names relative to
# RUN_DIR (or set CHECKPOINTS="iter_A iter_B"). A directory containing
# config.json is treated as HF and served directly; otherwise it is
# converted into <checkpoint>/hf first.
#
# If one side is already running, skip it:
#   START_RETRIEVER=0 SEARCH_URL=http://127.0.0.1:8000/retrieve bash ...
#   START_SGLANG=0 PORT=30000 bash ...
#
# Required:
#   MODEL_PATH     HF checkpoint to evaluate (when no checkpoints are given).
#   DATA_DIR       Directory with e5_Flat.index and wiki-18.jsonl (when
#                  START_RETRIEVER=1). Get them with (see docs/search.md):
#                    python eval/search/local_dense_retriever/download.py --save_path $DATA_DIR
#                    cat $DATA_DIR/part_* > $DATA_DIR/e5_Flat.index
#                    gzip -d $DATA_DIR/wiki-18.jsonl.gz
#   ORIGIN_HF_DIR  Config/tokenizer donor for torch_dist -> HF conversion
#                  (torch_dist checkpoints only); defaults to the cold-start
#                  checkpoint, /workspace/ckpt/sft_hf.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ===========================================================================
# KEY SETTINGS — what you most likely need to change.
# Every variable is env-overridable, e.g.:
#   DP_SIZE=4 MODEL_PATH=... bash scripts/eval/search.sh
# ===========================================================================

# --- What to evaluate -------------------------------------------------------
# HF checkpoint to evaluate (used when no checkpoint args/CHECKPOINTS given):
MODEL_PATH=${MODEL_PATH:-}
# Test set: HF dataset name (downloaded automatically) or a local parquet/jsonl.
# Defaults to the merged test set: https://huggingface.co/datasets/baohao/search-test
DATASET=${DATASET:-baohao/search-test}
# Training run dir with iter_* checkpoints (for relative checkpoint names):
RUN_DIR=${RUN_DIR:-}
# Config/tokenizer donor for torch_dist -> HF conversion (torch_dist only): the
# cold-start checkpoint, the right donor for every arm.
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:-/workspace/ckpt/sft_hf}

# --- Retriever --------------------------------------------------------------
# Directory with e5_Flat.index and wiki-18.jsonl
# (see eval/search/local_dense_retriever/download.py):
DATA_DIR=${DATA_DIR:-}
# The retriever runs in its own conda env (faiss-gpu etc., see README).
# Set RETRIEVER_CONDA_ENV="" to reuse the current environment instead.
RETRIEVER_CONDA_ENV=${RETRIEVER_CONDA_ENV-retriever}
RETRIEVER_PATH=${RETRIEVER_PATH:-intfloat/e5-base-v2}
# One retrieval server per GPU group / port pair:
RETRIEVER_GPU_GROUPS=${RETRIEVER_GPU_GROUPS:-"0,1,2,3 4,5,6,7"}
# One server, matching scripts/eval/start_retriever.sh. The
# old two-port default made half of every round-robin hit a port nothing
# listens on; local_search fails over, so it cost a refused connection per
# search rather than a wrong answer -- and a hang, if the port is filtered
# rather than refused.
RETRIEVER_PORTS=${RETRIEVER_PORTS:-"8000"}

# --- GPUs / server ----------------------------------------------------------
# Total GPUs used = TP_SIZE * DP_SIZE; restrict with SGLANG_CUDA_VISIBLE_DEVICES.
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-8}
SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-}
PORT=${PORT:-30000}

# --- Outputs ----------------------------------------------------------------
OUTPUT_WAS_SET=0
SUMMARY_OUTPUT_WAS_SET=0
if [ -n "${OUTPUT+x}" ]; then OUTPUT_WAS_SET=1; fi
if [ -n "${SUMMARY_OUTPUT+x}" ]; then SUMMARY_OUTPUT_WAS_SET=1; fi
# Default: <checkpoint or EVAL_OUTPUT_ROOT>/eval_search_r1{,_summary}.{jsonl,json}
OUTPUT=${OUTPUT:-}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-}
LOG_DIR=${LOG_DIR:-/tmp/mrpd_eval_logs}

# ===========================================================================
# ADVANCED SETTINGS — usually fine as-is.
# ===========================================================================

# Eval knobs.
# 4 samples at TEMPERATURE=1.0, i.e. mean pass@1 -- one sample off a temperature-1
# policy is a coin flip per question and the noise swamps the difference between
# two training arms. Paired with MAX_EXAMPLES_PER_SOURCE below, which pays for it.
NUM_SAMPLES=${NUM_SAMPLES:-1}
# baohao/search-test merges seven QA sets ranging from 125 questions (Bamboogle)
# to over 14k (PopQA). Capping each source keeps the mix from being re-weighted
# toward whatever the big sets are, and past a couple of thousand questions the
# standard error on a per-source accuracy is already well under a point. Fixed
# seed, so every checkpoint and every arm is scored on the same questions.
# MAX_EXAMPLES_PER_SOURCE=0 evaluates everything.
MAX_EXAMPLES_PER_SOURCE=${MAX_EXAMPLES_PER_SOURCE:-2500}
SAMPLE_SEED=${SAMPLE_SEED:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
# Per-turn generation cap. Must match SEARCH_R1_MAX_TOKENS_PER_TURN in the
# search training scripts (same 4096 default): the policy is only ever
# allowed 4096 new tokens in one turn during training, so letting a turn run
# to 8192 here evaluates it off-distribution and pays double on the
# repetition loops that never terminate on their own.
MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN:-4096}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-9216}
MAX_TURNS=${MAX_TURNS:-8}
TOPK=${TOPK:-3}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-1200}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-600}
# 16 in flight per GPU on the 8 x 80 GB node, against a KV pool that holds about
# 47 sequences at MAX_CONTEXT_LEN. It was 32 per GPU (256): at that load SGLang's
# HTTP front end, which serialises every response's per-token log-probs, stalled
# for over 10 s now and then, and every /generate opened during a stall failed
# with "HTTP connect ... timed out after 10s", throwing the whole trace away
# (counted as a timeout); the retriever, one process serving every search, also
# queued past its 60 s timeout (search_errors). On 40 GB cards halve this again.
MAX_CONCURRENT=${MAX_CONCURRENT:-128}
LIMIT=${LIMIT:-}
DATA_SOURCE_FILTER=${DATA_SOURCE_FILTER:-}
RESUME=${RESUME:-1}
PRINT_TURNS=${PRINT_TURNS:-0}
DEBUG_TRACE=${DEBUG_TRACE:-0}

# Retriever knobs.
# Default off: the retriever is a long-lived service on the host here, shared by
# every arm and outliving any single run, so a script that starts its own would
# collide with it on the ports and on GPU memory. START_RETRIEVER=1 makes this
# script launch and own one.
START_RETRIEVER=${START_RETRIEVER:-0}
RETRIEVER_CONDA_SH=${RETRIEVER_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
INDEX_FILE=${INDEX_FILE:-${DATA_DIR}/e5_Flat.index}
CORPUS_FILE=${CORPUS_FILE:-${DATA_DIR}/wiki-18.jsonl}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_WAIT_HOST=${RETRIEVER_WAIT_HOST:-127.0.0.1}

# SGLang server knobs.
START_SGLANG=${START_SGLANG:-1}
HOST=${HOST:-0.0.0.0}
WAIT_HOST=${WAIT_HOST:-127.0.0.1}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-0}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-40}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}
# 0, not 1. Training disables prefix caching because the weights change every
# step and cached KV goes stale; here they are frozen, and a multi-turn search
# rollout is the ideal case for it -- turn k's prompt is turn k-1's context
# plus the retrieved passages.
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-0}
EXTRA_SGLANG_ARGS=${EXTRA_SGLANG_ARGS:-""}
# Same as math.sh: each of the DP_SIZE replicas draws a torch.distributed
# rendezvous port from the kernel's ephemeral range, releases it, and binds it
# again only once its process is up, so a port can be handed to something else in
# between. That replica dies with "address already in use" and takes the server
# down with it. It is a race, not a resource limit: a relaunch draws fresh ports.
SERVER_START_RETRIES=${SERVER_START_RETRIES:-3}
SERVER_RETRY_SLEEP=${SERVER_RETRY_SLEEP:-10}

# Conversion knobs.
FORCE_CONVERT=${FORCE_CONVERT:-0}
SKIP_CONVERT=${SKIP_CONVERT:-0}
ADD_MISSING_FROM_ORIGIN_HF=${ADD_MISSING_FROM_ORIGIN_HF:-0}
CONVERT_EXTRA_ARGS=${CONVERT_EXTRA_ARGS:-}

# Runtime / cleanup knobs.
SEARCH_BACKEND=${SEARCH_BACKEND:-local}
SEARCH_URL=${SEARCH_URL:-}
CLEANUP_ON_EXIT=${CLEANUP_ON_EXIT:-1}

# ===========================================================================
# No user-editable settings below this line.
# ===========================================================================

# Defensive env so SGLang/torch don't crash on missing pwd entries.
export USER=${USER:-user}
export LOGNAME=${LOGNAME:-$USER}
export HOME=${HOME:-/tmp}
# Make sure localhost traffic bypasses any corporate http(s) proxy.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,0.0.0.0,${HOST},${WAIT_HOST}"
export no_proxy="${NO_PROXY}"

mkdir -p "${LOG_DIR}"

PIDS=()
SGLANG_PID=""

if [ "$#" -gt 0 ]; then
    CHECKPOINT_LIST=("$@")
elif [ -n "${CHECKPOINTS:-}" ]; then
    # shellcheck disable=SC2206
    CHECKPOINT_LIST=(${CHECKPOINTS})
else
    CHECKPOINT_LIST=()
fi

if [ "${#CHECKPOINT_LIST[@]}" -eq 0 ] && [ -z "${MODEL_PATH}" ]; then
    echo "Set MODEL_PATH to an HF checkpoint, or pass checkpoint dirs as arguments / CHECKPOINTS." >&2
    exit 1
fi

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

resolve_checkpoint_dir() {
    local checkpoint=$1
    if [[ "${checkpoint}" = /* ]]; then
        printf '%s\n' "${checkpoint}"
    else
        if [ -z "${RUN_DIR}" ]; then
            echo "RUN_DIR must be set when checkpoint names are relative: ${checkpoint}" >&2
            exit 1
        fi
        printf '%s/%s\n' "${RUN_DIR}" "${checkpoint}"
    fi
}

convert_checkpoint() {
    local checkpoint_dir=$1
    local hf_dir=$2
    if [ "${SKIP_CONVERT}" = "1" ]; then
        echo "Skipping conversion because SKIP_CONVERT=1: ${checkpoint_dir}"
        return
    fi
    if [ "${FORCE_CONVERT}" != "1" ] && [ -f "${hf_dir}/config.json" ]; then
        echo "HF checkpoint already exists, skipping conversion: ${hf_dir}"
        return
    fi
    if [ ! -f "${ORIGIN_HF_DIR}/config.json" ]; then
        echo "ORIGIN_HF_DIR=${ORIGIN_HF_DIR} is not an HF checkpoint; torch_dist -> HF conversion needs one." >&2
        exit 1
    fi
    local cmd=(
        python "${REPO_ROOT}/tools/convert_torch_dist_to_hf.py"
        --input-dir "${checkpoint_dir}"
        --output-dir "${hf_dir}"
        --origin-hf-dir "${ORIGIN_HF_DIR}"
    )
    # The converter refuses to write into an existing directory unless forced.
    # We reach this point either because FORCE_CONVERT=1, or because hf/ has no
    # config.json -- i.e. a previous conversion died part-way. Both want --force.
    if [ -d "${hf_dir}" ]; then
        cmd+=(--force)
    fi
    if [ "${ADD_MISSING_FROM_ORIGIN_HF}" = "1" ]; then
        cmd+=(--add-missing-from-origin-hf)
    fi
    if [ -n "${CONVERT_EXTRA_ARGS}" ]; then
        # shellcheck disable=SC2206
        cmd+=(${CONVERT_EXTRA_ARGS})
    fi
    "${cmd[@]}"
}

stop_sglang() {
    if [ -n "${SGLANG_PID}" ]; then
        local old_pid="${SGLANG_PID}"
        kill "${old_pid}" >/dev/null 2>&1 || true
        wait "${old_pid}" >/dev/null 2>&1 || true
        SGLANG_PID=""
        local remaining_pids=()
        local pid
        for pid in "${PIDS[@]:-}"; do
            if [ "${pid}" != "${old_pid}" ]; then
                remaining_pids+=("${pid}")
            fi
        done
        PIDS=("${remaining_pids[@]:-}")
        sleep 5
    fi
}

wait_for_tcp() {
    local host=$1
    local port=$2
    local timeout=$3
    local name=$4
    local pid=${5:-}
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
        sleep 2
    done
    echo "Timed out waiting for ${name} at ${host}:${port} after ${timeout}s" >&2
    return 1
}

join_retriever_urls() {
    local urls=()
    local port
    for port in "$@"; do
        urls+=("http://${RETRIEVER_WAIT_HOST}:${port}/retrieve")
    done
    local IFS=,
    echo "${urls[*]}"
}

start_retrievers() {
    if [ -z "${DATA_DIR}" ] && { [ ! -f "${INDEX_FILE}" ] || [ ! -f "${CORPUS_FILE}" ]; }; then
        echo "Set DATA_DIR to the directory holding e5_Flat.index and wiki-18.jsonl" >&2
        echo "(see docs/search.md for how to download and assemble them, or set INDEX_FILE/CORPUS_FILE directly)" >&2
        exit 1
    fi
    read -r -a gpu_groups <<< "${RETRIEVER_GPU_GROUPS}"
    read -r -a ports <<< "${RETRIEVER_PORTS}"
    local retriever_pids=()
    if [ "${#gpu_groups[@]}" -ne "${#ports[@]}" ]; then
        echo "RETRIEVER_GPU_GROUPS and RETRIEVER_PORTS must have the same number of entries" >&2
        exit 1
    fi
    for i in "${!gpu_groups[@]}"; do
        local gpus="${gpu_groups[$i]}"
        local port="${ports[$i]}"
        local safe_gpus="${gpus//,/}"
        local log_file="${LOG_DIR}/retrieval_server_gpus_${safe_gpus}_port_${port}.log"
        echo "Starting retrieval server on GPUs ${gpus}, port ${port}; log: ${log_file}"
        (
            if [ -n "${RETRIEVER_CONDA_ENV}" ]; then
                source "${RETRIEVER_CONDA_SH}"
                conda activate "${RETRIEVER_CONDA_ENV}"
                export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
            fi
            export CUDA_LAUNCH_BLOCKING=1
            exec env CUDA_VISIBLE_DEVICES="${gpus}" python "${REPO_ROOT}/eval/search/local_dense_retriever/retrieval_server.py" \
                --index_path "${INDEX_FILE}" \
                --corpus_path "${CORPUS_FILE}" \
                --topk "${TOPK}" \
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
        wait_for_tcp "${RETRIEVER_WAIT_HOST}" "${ports[$i]}" "${SERVER_WAIT_TIMEOUT}" "retriever" "${retriever_pids[$i]}"
    done
    if [ -z "${SEARCH_URL}" ]; then
        SEARCH_URL="$(join_retriever_urls "${ports[@]}")"
    fi
}

# When one DP replica dies, SGLang kills the others with SIGQUIT and the parent
# fails with "Rank 0 scheduler died during initialization (exit code: -3)". That
# is collateral: the replica that actually failed logged its own error earlier,
# usually outside a tail -n 80. So print the first error lines as well as the tail.
report_server_failure() {
    local server_log="$1"
    echo "--- first errors in ${server_log} ---"
    grep -n -i -m 15 -E 'error|traceback|assert|out of memory|address already in use|failed|no such|cannot|refus' \
        "${server_log}" 2>/dev/null || echo "(no error lines matched)"
    echo "--- last 80 lines of ${server_log} ---"
    tail -n 80 "${server_log}" 2>/dev/null || true
    echo "--- end of ${server_log} ---"
    echo "If the tail shows exit code -3 or -9, read the first-errors block above:"
    echo "the dying worker's own error comes first, everything after it is cleanup."
}

start_sglang() {
    local model_path=$1
    local log_suffix=${2:-model}
    local server_log="${LOG_DIR}/sglang_${log_suffix}_${PORT}.log"
    stop_sglang
    local cmd=(
        python3 -m sglang.launch_server
        --model-path "${model_path}"
        --host "${HOST}"
        --port "${PORT}"
        --tp "${TP_SIZE}"
        --dp "${DP_SIZE}"
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
        --max-running-requests "${MAX_RUNNING_REQUESTS}"
        --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}"
        --trust-remote-code
    )
    if [ "${ENABLE_DP_ATTENTION}" = "1" ]; then
        cmd+=(--enable-dp-attention)
    fi
    if [ "${DISABLE_RADIX_CACHE}" = "1" ]; then
        cmd+=(--disable-radix-cache)
    fi
    if [ -n "${EXTRA_SGLANG_ARGS}" ]; then
        # shellcheck disable=SC2206
        cmd+=(${EXTRA_SGLANG_ARGS})
    fi
    echo "Starting SGLang for ${model_path} on ${HOST}:${PORT}; log: ${server_log}"
    (
        # SGLang spawns TP workers via multiprocessing "spawn"; each child
        # chdirs into the parent cwd, so move somewhere always accessible.
        cd /tmp
        if [ -n "${SGLANG_CUDA_VISIBLE_DEVICES}" ]; then
            export CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES}"
        fi
        exec "${cmd[@]}"
    ) >"${server_log}" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    SGLANG_PID="${pid}"
    if ! wait_for_tcp "${WAIT_HOST}" "${PORT}" "${SERVER_WAIT_TIMEOUT}" "SGLang" "${pid}"; then
        report_server_failure "${server_log}"
        return 1
    fi
}

run_eval() {
    local model_path=$1
    local output_path=$2
    local summary_output_path=$3
    local extra=()
    if [ -n "${LIMIT}" ]; then
        extra+=(--limit "${LIMIT}")
    fi
    if [ -n "${DATA_SOURCE_FILTER}" ]; then
        extra+=(--data-source-filter "${DATA_SOURCE_FILTER}")
    fi
    if [ "${PRINT_TURNS}" = "1" ]; then
        extra+=(--print-turns)
    fi
    if [ "${DEBUG_TRACE}" = "1" ]; then
        extra+=(--debug-trace)
    fi
    if [ "${RESUME}" = "1" ]; then
        extra+=(--resume)
    fi
    python "${REPO_ROOT}/eval/search/eval.py" \
        --model-path "${model_path}" \
        --dataset "${DATASET}" \
        --num-samples "${NUM_SAMPLES}" \
        --max-examples-per-source "${MAX_EXAMPLES_PER_SOURCE}" \
        --sample-seed "${SAMPLE_SEED}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-tokens-per-turn "${MAX_TOKENS_PER_TURN}" \
        --max-context-len "${MAX_CONTEXT_LEN}" \
        --max-turns "${MAX_TURNS}" \
        --topk "${TOPK}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --host "${WAIT_HOST}" \
        --port "${PORT}" \
        --server-wait-timeout "${SERVER_WAIT_TIMEOUT}" \
        --max-concurrent "${MAX_CONCURRENT}" \
        --search-backend "${SEARCH_BACKEND}" \
        --search-url "${SEARCH_URL}" \
        --output "${output_path}" \
        --summary-output "${summary_output_path}" \
        --sample-timeout "${SAMPLE_TIMEOUT}" \
        "${extra[@]}"
}

run_one_model() {
    local model_path=$1
    local output_root=$2
    local name=$3
    local output_path="${OUTPUT}"
    local summary_output_path="${SUMMARY_OUTPUT}"
    if [ "${OUTPUT_WAS_SET}" != "1" ]; then
        output_path="${output_root}/eval_search_r1.jsonl"
    fi
    if [ "${SUMMARY_OUTPUT_WAS_SET}" != "1" ]; then
        summary_output_path="${output_root}/eval_search_r1_summary.json"
    fi
    mkdir -p "$(dirname -- "${output_path}")" "$(dirname -- "${summary_output_path}")"

    if [ "${START_SGLANG}" = "1" ]; then
        local attempt=1
        until start_sglang "${model_path}" "${name}"; do
            if [ "${attempt}" -ge "${SERVER_START_RETRIES}" ]; then
                echo "SGLang failed to start ${attempt} times for ${name}; giving up. If every" >&2
                echo "attempt died on the same port, something outside this script holds it --" >&2
                echo "a random port in the traceback is a losable race, a fixed one is not." >&2
                exit 1
            fi
            attempt=$((attempt + 1))
            echo "Retrying the SGLang launch for ${name}" \
                 "(attempt ${attempt}/${SERVER_START_RETRIES}) in ${SERVER_RETRY_SLEEP}s..."
            sleep "${SERVER_RETRY_SLEEP}"
        done
    else
        wait_for_tcp "${WAIT_HOST}" "${PORT}" "${SERVER_WAIT_TIMEOUT}" "SGLang"
    fi
    echo "Using SEARCH_URL=${SEARCH_URL}"
    run_eval "${model_path}" "${output_path}" "${summary_output_path}"
}

if [ "${START_RETRIEVER}" = "1" ]; then
    start_retrievers
elif [ -z "${SEARCH_URL}" ]; then
    read -r -a ports <<< "${RETRIEVER_PORTS}"
    SEARCH_URL="$(join_retriever_urls "${ports[@]}")"
fi

if [ "${#CHECKPOINT_LIST[@]}" -gt 0 ]; then
    for checkpoint in "${CHECKPOINT_LIST[@]}"; do
        CHECKPOINT_DIR="$(resolve_checkpoint_dir "${checkpoint}")"
        CHECKPOINT_NAME="$(basename "${CHECKPOINT_DIR}")"
        # <root>/<checkpoint name>/, as math.sh does; one directory per checkpoint,
        # or two checkpoints evaluated into the same root overwrite (and --resume
        # continues) each other's eval_search_r1.jsonl.
        if [ -n "${EVAL_OUTPUT_ROOT}" ]; then
            OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}/${CHECKPOINT_NAME}"
        else
            OUTPUT_ROOT="${CHECKPOINT_DIR}"
        fi
        if [ ! -d "${CHECKPOINT_DIR}" ]; then
            echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
            exit 1
        fi
        if [ -f "${CHECKPOINT_DIR}/config.json" ]; then
            # Already an HF checkpoint: serve it directly.
            HF_DIR="${CHECKPOINT_DIR}"
        else
            HF_DIR="${CHECKPOINT_DIR}/hf"
            convert_checkpoint "${CHECKPOINT_DIR}" "${HF_DIR}"
        fi
        run_one_model "${HF_DIR}" "${OUTPUT_ROOT}" "${CHECKPOINT_NAME}"
    done
else
    if [ -n "${EVAL_OUTPUT_ROOT}" ]; then
        run_one_model "${MODEL_PATH}" "${EVAL_OUTPUT_ROOT}/$(basename "${MODEL_PATH}")" "$(basename "${MODEL_PATH}")"
    else
        run_one_model "${MODEL_PATH}" "${MODEL_PATH}" "$(basename "${MODEL_PATH}")"
    fi
fi
