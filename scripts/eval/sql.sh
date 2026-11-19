#!/bin/bash
# The single entry point for text-to-SQL evaluation, math.sh's shape:
#   1. Convert Megatron torch_dist checkpoints to HF format (skipped for
#      checkpoints that are already HF, i.e. contain config.json).
#   2. Serve each HF checkpoint with SGLang.
#   3. Run eval/sql/eval.py over every SQL set scripts/data/prepare_sql.py wrote
#      (spider_dev, spider_test when the archive has it, bird_minidev).
#
# Usage:
#   RUN_DIR=/workspace/runs/grpo_sql bash scripts/eval/sql.sh iter_0000199
#
#   DATASETS="spider_dev" RUN_DIR=... bash scripts/eval/sql.sh iter_0000199   # one set only
#   bash scripts/eval/sql.sh /path/to/hf_model                                # an HF dir, no conversion
#
# SQL_DATA_DIR (prepare_sql.py's --out, default /workspace/data/sql; the databases
# under SQL_DB_ROOT, default <SQL_DATA_DIR>/databases) and ORIGIN_HF_DIR (the
# cold-start HF checkpoint, default /workspace/ckpt/sft_hf) are per machine, not per run.
#
# Scores are mean pass@1 over NUM_SAMPLES traces at TEMPERATURE (4 at 1.0 by default,
# the math eval's convention for its larger sets; NUM_SAMPLES=1 for a quick look).
# Outputs land beside the checkpoint (<checkpoint>/<dataset>/{results.jsonl,summary.json})
# or under EVAL_OUTPUT_ROOT/<checkpoint name>/<dataset>/.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ===========================================================================
# KEY SETTINGS
# ===========================================================================
RUN_DIR=${RUN_DIR:-}
# Config/tokenizer donor for torch_dist -> HF conversion: the cold-start checkpoint.
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:-/workspace/ckpt/sft_hf}

# prepare_sql.py's output: <SQL_DATA_DIR>/<dataset>.jsonl and the databases.
SQL_DATA_DIR=${SQL_DATA_DIR:-/workspace/data/sql}
export SQL_DB_ROOT=${SQL_DB_ROOT:-${SQL_DATA_DIR}/databases}
# Every eval set prepare_sql.py wrote, unless DATASETS says otherwise.
if [ -z "${DATASETS:-}" ]; then
    DATASETS=""
    for _ds in spider_dev spider_test bird_minidev; do
        if [ -f "${SQL_DATA_DIR}/${_ds}.jsonl" ]; then DATASETS="${DATASETS} ${_ds}"; fi
    done
fi
NUM_SAMPLES=${NUM_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}

TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-8}
SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}
PORT=${PORT:-30000}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-}

# ===========================================================================
# ADVANCED SETTINGS
# ===========================================================================
FORCE_CONVERT=${FORCE_CONVERT:-0}
SKIP_CONVERT=${SKIP_CONVERT:-0}
ADD_MISSING_FROM_ORIGIN_HF=${ADD_MISSING_FROM_ORIGIN_HF:-0}
CONVERT_EXTRA_ARGS=${CONVERT_EXTRA_ARGS:-""}

BIND_HOST=${BIND_HOST:-0.0.0.0}
EVAL_HOST=${EVAL_HOST:-127.0.0.1}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-0}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-128}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-900}
SERVER_START_RETRIES=${SERVER_START_RETRIES:-3}
SERVER_RETRY_SLEEP=${SERVER_RETRY_SLEEP:-10}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-/tmp}
EXTRA_SGLANG_ARGS=${EXTRA_SGLANG_ARGS:-""}

# Training's budget (teacher_grpo_sql.sh): 9216 per trajectory; the per-turn cap is
# SQL_MAX_TOKENS_PER_TURN (4096) below, MAX_NEW_TOKENS only bounds it from above.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-9216}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-600}
MAX_CONCURRENT=${MAX_CONCURRENT:-128}
RESUME=${RESUME:-1}
PRINT_TURNS=${PRINT_TURNS:-0}
DEBUG_TRACE=${DEBUG_TRACE:-0}

# The SQL environment's knobs (train/src/sql_tools.py), the training defaults.
export SQL_MAX_TOOL_CALLS=${SQL_MAX_TOOL_CALLS:-6}
export SQL_MAX_TURNS=${SQL_MAX_TURNS:-8}
export SQL_MAX_TOKENS_PER_TURN=${SQL_MAX_TOKENS_PER_TURN:-4096}
export SQL_EXEC_TIMEOUT=${SQL_EXEC_TIMEOUT:-5}
export SQL_SCORE_TIMEOUT=${SQL_SCORE_TIMEOUT:-30}
export SQL_MAX_RESULT_ROWS=${SQL_MAX_RESULT_ROWS:-20}
export SQL_MAX_RESULT_CHARS=${SQL_MAX_RESULT_CHARS:-4000}
export SQL_TOOL_CONCURRENCY=${SQL_TOOL_CONCURRENCY:-128}

# ===========================================================================
# No user-editable settings below this line.
# ===========================================================================
if [ "$#" -gt 0 ]; then
    CHECKPOINT_LIST=("$@")
elif [ -n "${CHECKPOINTS:-}" ]; then
    # shellcheck disable=SC2206
    CHECKPOINT_LIST=(${CHECKPOINTS})
else
    echo "No checkpoints given. Pass them as arguments or set CHECKPOINTS." >&2
    exit 1
fi
if [ ! -d "${SQL_DATA_DIR}" ] || [ -z "${DATASETS// /}" ]; then
    echo "No eval set under SQL_DATA_DIR=${SQL_DATA_DIR}; set it to prepare_sql.py's --out directory (spider_dev.jsonl, ...)." >&2
    exit 1
fi
# shellcheck disable=SC2206
DATASET_LIST=(${DATASETS})
for dataset in "${DATASET_LIST[@]}"; do
    if [ ! -f "${SQL_DATA_DIR}/${dataset}.jsonl" ]; then
        echo "${SQL_DATA_DIR}/${dataset}.jsonl does not exist (run scripts/data/prepare_sql.py without --no-bird)." >&2
        exit 1
    fi
done
if [ ! -d "${SQL_DB_ROOT}" ]; then
    echo "SQL_DB_ROOT=${SQL_DB_ROOT} is not a directory." >&2
    exit 1
fi

export USER=${USER:-user}
export LOGNAME=${LOGNAME:-$USER}
export HOME=${HOME:-/tmp}
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,0.0.0.0,${BIND_HOST},${EVAL_HOST}"
export no_proxy="${NO_PROXY}"

SGLANG_PID=""
SUMMARY_PATHS=()

resolve_checkpoint_dir() {
    local checkpoint="$1"
    if [[ "${checkpoint}" = /* ]]; then
        printf '%s\n' "${checkpoint}"
    else
        if [ -z "${RUN_DIR}" ]; then
            echo "RUN_DIR must be set to resolve relative checkpoint '${checkpoint}'" >&2
            exit 1
        fi
        printf '%s/%s\n' "${RUN_DIR}" "${checkpoint}"
    fi
}

stop_sglang_server() {
    if [ -n "${SGLANG_PID:-}" ]; then
        kill "${SGLANG_PID}" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
        SGLANG_PID=""
        sleep 5
    fi
}

print_summary_table() {
    set +x
    if [ "${#SUMMARY_PATHS[@]}" -eq 0 ]; then
        echo
        echo "No results were produced: no eval got as far as writing a summary.json."
        return 0
    fi
    python3 - "${SUMMARY_PATHS[@]}" <<'PYEOF' || true
import json, os, sys
rows = []
for path in sys.argv[1:]:
    label = os.path.basename(os.path.dirname(os.path.dirname(path)))
    dataset = os.path.basename(os.path.dirname(path))
    try:
        with open(path) as fh:
            s = json.load(fh)
    except (OSError, ValueError):
        rows.append((label, dataset, None, None, None))
        continue
    rows.append((label, dataset, s.get("accuracy"), (s.get("timeout_stats") or {}).get("timeout_rate"), s.get("num_samples")))
w = max(max(len(r[0]) for r in rows), len("checkpoint"))
d = max(max(len(r[1]) for r in rows), len("benchmark"))
print(); print("=" * (w + d + 26))
print(f"{'checkpoint':<{w}}  {'benchmark':<{d}}  {'acc':>7} {'timeout':>8} {'n':>3}")
print("-" * (w + d + 26))
for label, dataset, acc, timeout, n in rows:
    if acc is None:
        print(f"{label:<{w}}  {dataset:<{d}}  {'--':>7} {'--':>8} {'--':>3}  (no summary.json)")
    else:
        print(f"{label:<{w}}  {dataset:<{d}}  {acc:>7.4f} {timeout:>7.1%} {n:>3}")
print("=" * (w + d + 26))
print("acc is mean pass@1 (execution match); timed-out traces are dropped from a")
print("question's denominator, so a non-zero timeout column makes it optimistic.")
PYEOF
}

on_exit() {
    stop_sglang_server
    print_summary_table
}
trap on_exit EXIT

convert_checkpoint() {
    local checkpoint_dir="$1"
    local hf_dir="$2"
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

report_server_failure() {
    local server_log="$1"
    echo "--- first errors in ${server_log} ---"
    grep -n -i -m 15 -E 'error|traceback|assert|out of memory|address already in use|failed|no such|cannot|refus' \
        "${server_log}" 2>/dev/null || echo "(no error lines matched)"
    echo "--- last 80 lines of ${server_log} ---"
    tail -n 80 "${server_log}" 2>/dev/null || true
    echo "--- end of ${server_log} ---"
}

start_sglang_server() {
    local model_path="$1"
    local checkpoint_name="$2"
    local server_log="${SERVER_LOG_DIR}/sglang_eval_sql_${checkpoint_name}_${PORT}.log"
    stop_sglang_server
    mkdir -p "${SERVER_LOG_DIR}"
    CMD=(
        python3 -m sglang.launch_server
        --model-path "${model_path}"
        --host "${BIND_HOST}"
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
        CMD+=(--enable-dp-attention)
    fi
    if [ "${DISABLE_RADIX_CACHE}" = "1" ]; then
        CMD+=(--disable-radix-cache)
    fi
    if [ -n "${EXTRA_SGLANG_ARGS}" ]; then
        # shellcheck disable=SC2206
        CMD+=(${EXTRA_SGLANG_ARGS})
    fi
    echo "Launching SGLang for ${model_path}. Log: ${server_log}"
    (
        if [ -n "${SGLANG_CUDA_VISIBLE_DEVICES}" ]; then
            export CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES}"
        fi
        cd /tmp
        exec "${CMD[@]}"
    ) > "${server_log}" 2>&1 &
    SGLANG_PID=$!
    local deadline=$((SECONDS + SERVER_WAIT_TIMEOUT))
    until curl -sf --noproxy '*' "http://${EVAL_HOST}:${PORT}/health_generate" > /dev/null; do
        if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
            echo "SGLang server exited early."
            report_server_failure "${server_log}"
            return 1
        fi
        if [ "${SECONDS}" -ge "${deadline}" ]; then
            echo "Timed out waiting for SGLang server on ${EVAL_HOST}:${PORT}."
            report_server_failure "${server_log}"
            return 1
        fi
        echo "Waiting for SGLang server on ${EVAL_HOST}:${PORT}..."
        tail -n 10 "${server_log}" || true
        sleep 5
    done
    curl --noproxy '*' "http://${EVAL_HOST}:${PORT}/get_model_info" || true
    echo "SGLang server is ready on ${EVAL_HOST}:${PORT}"
}

run_eval_dataset() {
    local model_path="$1"
    local output_root="$2"
    local dataset="$3"
    local output_dir_ds="${output_root}/${dataset}"
    mkdir -p "${output_dir_ds}"
    EVAL_CMD=(
        python "${REPO_ROOT}/eval/sql/eval.py"
        --model-path "${model_path}"
        --dataset "${SQL_DATA_DIR}/${dataset}.jsonl"
        --dataset-name "${dataset}"
        --num-samples "${NUM_SAMPLES}"
        --max-new-tokens "${MAX_NEW_TOKENS}"
        --max-context-len "${MAX_CONTEXT_LEN}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --host "${EVAL_HOST}"
        --port "${PORT}"
        --server-wait-timeout "${SERVER_WAIT_TIMEOUT}"
        --max-concurrent "${MAX_CONCURRENT}"
        --output "${output_dir_ds}/results.jsonl"
        --summary-output "${output_dir_ds}/summary.json"
        --sample-timeout "${SAMPLE_TIMEOUT}"
    )
    if [ "${PRINT_TURNS}" = "1" ]; then
        EVAL_CMD+=(--print-turns)
    fi
    if [ "${DEBUG_TRACE}" = "1" ]; then
        EVAL_CMD+=(--debug-trace)
    fi
    if [ "${RESUME}" = "1" ]; then
        EVAL_CMD+=(--resume)
    fi
    "${EVAL_CMD[@]}"
}

for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    CHECKPOINT_DIR="$(resolve_checkpoint_dir "${checkpoint}")"
    CHECKPOINT_NAME="$(basename "${CHECKPOINT_DIR}")"
    if [ -n "${EVAL_OUTPUT_ROOT}" ]; then
        OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}/${CHECKPOINT_NAME}"
    else
        OUTPUT_ROOT="${CHECKPOINT_DIR}"
    fi
    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}"
        exit 1
    fi
    if [ -f "${CHECKPOINT_DIR}/config.json" ]; then
        HF_DIR="${CHECKPOINT_DIR}"
    else
        HF_DIR="${CHECKPOINT_DIR}/hf"
        convert_checkpoint "${CHECKPOINT_DIR}" "${HF_DIR}"
    fi
    server_attempt=1
    until start_sglang_server "${HF_DIR}" "${CHECKPOINT_NAME}"; do
        if [ "${server_attempt}" -ge "${SERVER_START_RETRIES}" ]; then
            echo "SGLang server failed to start ${server_attempt} times for ${CHECKPOINT_NAME}; giving up." >&2
            exit 1
        fi
        server_attempt=$((server_attempt + 1))
        echo "Retrying SGLang server launch for ${CHECKPOINT_NAME} (attempt ${server_attempt}/${SERVER_START_RETRIES}) in ${SERVER_RETRY_SLEEP}s..."
        sleep "${SERVER_RETRY_SLEEP}"
    done
    for dataset in "${DATASET_LIST[@]}"; do
        run_eval_dataset "${HF_DIR}" "${OUTPUT_ROOT}" "${dataset}"
        SUMMARY_PATHS+=("${OUTPUT_ROOT}/${dataset}/summary.json")
    done
    stop_sglang_server
done
