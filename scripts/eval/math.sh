#!/bin/bash
# The single entry point for math (ReTool) evaluation:
#   1. Convert Megatron torch_dist checkpoints to HF format (skipped for
#      checkpoints that are already HF, i.e. contain config.json).
#   2. Serve each HF checkpoint with SGLang.
#   3. Run eval/math/eval.py over multiple benchmarks
#      (aime24 aime25 amc23 minerva_math olympiadbench math500 by default).
#
# Usage:
#   # Megatron torch_dist checkpoints from a training run:
#   RUN_DIR=/workspace/runs/grpo_math bash scripts/eval/math.sh iter_0000199 iter_0000299
#
#   CHECKPOINTS="iter_0000199 iter_0000299" RUN_DIR=/workspace/runs/grpo_math \
#       bash scripts/eval/math.sh
#
#   # A ready HF checkpoint (no conversion, no RUN_DIR needed):
#   DATASETS="aime24" bash scripts/eval/math.sh /path/to/hf_model
#
# Checkpoint arguments may be absolute directories or names relative to
# RUN_DIR. A directory containing config.json is treated as HF and served
# directly; otherwise it is converted into <checkpoint>/hf first.
#
# For torch_dist checkpoints:
#   RUN_DIR        Training run directory containing iter_* checkpoints
#                  (not needed if all checkpoints are absolute paths).
#   ORIGIN_HF_DIR  Original HF checkpoint of the same architecture, the
#                  config/tokenizer donor for torch_dist -> HF conversion;
#                  defaults to the cold-start checkpoint, /workspace/ckpt/sft_hf.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ===========================================================================
# KEY SETTINGS — what you most likely need to change.
# Every variable is env-overridable, e.g.:
#   DP_SIZE=4 DATASETS="aime24" bash scripts/eval/math.sh ...
# ===========================================================================

# --- What to evaluate -------------------------------------------------------
# Checkpoints are given as CLI arguments or via CHECKPOINTS="iter_A iter_B".
# Training run dir with iter_* checkpoints (for relative checkpoint names):
RUN_DIR=${RUN_DIR:-}
# Config/tokenizer donor for torch_dist -> HF conversion (torch_dist only): the
# cold-start checkpoint, the right donor for every arm.
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:-/workspace/ckpt/sft_hf}

# --- Benchmarks -------------------------------------------------------------
# HF namespace or local directory holding the benchmark datasets, passed to
# eval.py as "${DATA_ROOT}/${dataset}" (e.g. baohao/aime24):
DATA_ROOT=${DATA_ROOT:-baohao}
DATASETS=${DATASETS:-"aime24 aime25 amc23 minerva_math olympiadbench math500"}
# Samples per prompt: harder/smaller benchmarks get more samples.
NUM_SAMPLES_HARD=${NUM_SAMPLES_HARD:-8}    # aime24, aime25, amc23
NUM_SAMPLES_OTHER=${NUM_SAMPLES_OTHER:-4}  # everything else
# Sampling, not greedy: the numbers above are mean pass@1, so eval.py's own
# default of 0.0 would make every sample of a prompt identical.
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}

# --- GPUs / server ----------------------------------------------------------
# Total GPUs used = TP_SIZE * DP_SIZE; restrict with SGLANG_CUDA_VISIBLE_DEVICES.
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-8}
SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}
PORT=${PORT:-30000}

# --- Outputs ----------------------------------------------------------------
# Where per-benchmark outputs go. Unset (the default) puts them beside each
# checkpoint, at <checkpoint>/<dataset>/{results.jsonl,summary.json}. When set,
# results go to <root>/<checkpoint name>/<dataset>/ -- the checkpoint name has to
# stay in the path, or evaluating several checkpoints in one invocation would
# write them all to the same files.
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-}

# ===========================================================================
# ADVANCED SETTINGS — usually fine as-is.
# ===========================================================================

# Conversion knobs.
FORCE_CONVERT=${FORCE_CONVERT:-0}
SKIP_CONVERT=${SKIP_CONVERT:-0}
ADD_MISSING_FROM_ORIGIN_HF=${ADD_MISSING_FROM_ORIGIN_HF:-0}
CONVERT_EXTRA_ARGS=${CONVERT_EXTRA_ARGS:-""}

# SGLang server knobs.
BIND_HOST=${BIND_HOST:-0.0.0.0}
EVAL_HOST=${EVAL_HOST:-127.0.0.1}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-0}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-128}
SCHEDULE_CONSERVATIVENESS=${SCHEDULE_CONSERVATIVENESS:-0.3}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-900}
# Launching DP_SIZE replicas at once means DP_SIZE torch.distributed rendezvous
# ports drawn at random, and SGLang draws them from inside the kernel's ephemeral
# range (net.ipv4.ip_local_port_range, 32768-60999 by default) -- so a port can be
# handed to something else between SGLang checking it and binding it. One replica
# then dies with "DistNetworkError: ... address already in use" and takes the
# server down with it. It is a race, not a resource limit: a relaunch draws fresh
# ports and normally succeeds.
SERVER_START_RETRIES=${SERVER_START_RETRIES:-3}
SERVER_RETRY_SLEEP=${SERVER_RETRY_SLEEP:-10}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-/tmp}
EXTRA_SGLANG_ARGS=${EXTRA_SGLANG_ARGS:-""}

# Eval knobs.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-1200}
MAX_CONCURRENT=${MAX_CONCURRENT:-128}
RESUME=${RESUME:-1}
PRINT_TURNS=${PRINT_TURNS:-1}
DEBUG_TRACE=${DEBUG_TRACE:-1}

# ===========================================================================
# No user-editable settings below this line.
# ===========================================================================

# If no positional checkpoints are supplied, split CHECKPOINTS by whitespace.
if [ "$#" -gt 0 ]; then
    CHECKPOINT_LIST=("$@")
elif [ -n "${CHECKPOINTS:-}" ]; then
    # shellcheck disable=SC2206
    CHECKPOINT_LIST=(${CHECKPOINTS})
else
    echo "No checkpoints given. Pass them as arguments or set CHECKPOINTS." >&2
    exit 1
fi

# shellcheck disable=SC2206
DATASET_LIST=(${DATASETS})

# Defensive env so SGLang/torch don't crash on missing pwd entries
# (e.g. container UIDs absent from /etc/passwd).
export USER=${USER:-user}
export LOGNAME=${LOGNAME:-$USER}
export HOME=${HOME:-/tmp}

# Python tool sandbox used by the ReTool rollouts during eval.
export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-subprocess}
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-128}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-600}

# Make sure localhost traffic bypasses any corporate http(s) proxy.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,0.0.0.0,${BIND_HOST},${EVAL_HOST}"
export no_proxy="${NO_PROXY}"

SGLANG_PID=""
# Every summary.json this invocation produces, tabulated on the way out.
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
# Each benchmark prints its own summary.json and nothing aggregates them, so with
# six benchmarks per checkpoint the numbers are unfindable in the scrollback --
# especially under PRINT_TURNS=1, which echoes every assistant turn of every
# trace. Print one table of everything this invocation produced. It runs from the
# EXIT trap so a crash on the third checkpoint still reports the first two.
print_summary_table() {
    set +x
    if [ "${#SUMMARY_PATHS[@]}" -eq 0 ]; then
        echo
        echo "No benchmark results were produced: no eval got as far as writing a"
        echo "summary.json. The failure is above -- typically the SGLang server, since"
        echo "nothing downstream of it ran."
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
    rows.append((
        label,
        dataset,
        s.get("accuracy"),
        (s.get("timeout_stats") or {}).get("timeout_rate"),
        s.get("num_samples"),
    ))

if not rows:
    sys.exit(0)

w = max(max(len(r[0]) for r in rows), len("checkpoint"))
d = max(max(len(r[1]) for r in rows), len("benchmark"))
print()
print("=" * (w + d + 26))
print(f"{'checkpoint':<{w}}  {'benchmark':<{d}}  {'acc':>7} {'timeout':>8} {'n':>3}")
print("-" * (w + d + 26))
seen = []
for label, dataset, acc, timeout, n in rows:
    if label not in seen:
        seen.append(label)
    if acc is None:
        print(f"{label:<{w}}  {dataset:<{d}}  {'--':>7} {'--':>8} {'--':>3}  (no summary.json)")
    else:
        print(f"{label:<{w}}  {dataset:<{d}}  {acc:>7.4f} {timeout:>7.1%} {n:>3}")
for label in seen:
    vals = [r[2] for r in rows if r[0] == label and r[2] is not None]
    if len(vals) > 1:
        tag = f"MEAN of {len(vals)}"
        print(f"{label:<{w}}  {tag:<{d}}  {sum(vals) / len(vals):>7.4f}")
print("=" * (w + d + 26))
print("acc is mean pass@1; timed-out samples are dropped from a problem's")
print("denominator, so a non-zero timeout column makes the accuracy optimistic.")
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
    # The converter refuses to write into an existing directory unless forced.
    # We reach this point either because FORCE_CONVERT=1, or because hf/ has no
    # config.json -- i.e. a previous conversion died part-way. Both want --force;
    # without it the run aborts and the stale directory has to be removed by hand.
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

# When a DP worker dies, SGLang's SIGCHLD handler kills the remaining workers with
# SIGQUIT and the parent then fails in _wait_for_scheduler_ready with
# "Rank 0 scheduler died during initialization (exit code: -3)". That traceback is
# collateral, not the cause -- exit code -3 is the sibling being killed, and the
# first worker's actual error is further up the log, usually outside a tail -n 80.
# So print the head of the errors as well as the tail.
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

start_sglang_server() {
    local model_path="$1"
    local checkpoint_name="$2"
    local server_log="${SERVER_LOG_DIR}/sglang_eval_${checkpoint_name}_${PORT}.log"
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
        # SGLang spawns TP workers via multiprocessing "spawn"; each child
        # chdirs into the parent cwd, so move somewhere always accessible.
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

num_samples_for_dataset() {
    local dataset="$1"
    case "${dataset}" in
        aime24|aime25|amc23)
            printf '%s\n' "${NUM_SAMPLES_HARD}"
            ;;
        *)
            printf '%s\n' "${NUM_SAMPLES_OTHER}"
            ;;
    esac
}

run_eval_dataset() {
    local model_path="$1"
    local output_root="$2"
    local dataset="$3"
    local dataset_num_samples
    dataset_num_samples="$(num_samples_for_dataset "${dataset}")"
    local output_dir_ds="${output_root}/${dataset}"
    mkdir -p "${output_dir_ds}"
    EVAL_CMD=(
        python "${REPO_ROOT}/eval/math/eval.py"
        --model-path "${model_path}"
        --dataset "${DATA_ROOT}/${dataset}"
        --num-samples "${dataset_num_samples}"
        --max-new-tokens "${MAX_NEW_TOKENS}"
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
        # Already an HF checkpoint: serve it directly.
        HF_DIR="${CHECKPOINT_DIR}"
    else
        HF_DIR="${CHECKPOINT_DIR}/hf"
        convert_checkpoint "${CHECKPOINT_DIR}" "${HF_DIR}"
    fi
    server_attempt=1
    until start_sglang_server "${HF_DIR}" "${CHECKPOINT_NAME}"; do
        if [ "${server_attempt}" -ge "${SERVER_START_RETRIES}" ]; then
            echo "SGLang server failed to start ${server_attempt} times for" >&2
            echo "${CHECKPOINT_NAME}; giving up. If every attempt died on the same" >&2
            echo "port, something outside this script holds it -- a random port in" >&2
            echo "the traceback is a losable race, a fixed one is not." >&2
            exit 1
        fi
        server_attempt=$((server_attempt + 1))
        echo "Retrying SGLang server launch for ${CHECKPOINT_NAME}" \
             "(attempt ${server_attempt}/${SERVER_START_RETRIES}) in ${SERVER_RETRY_SLEEP}s..."
        sleep "${SERVER_RETRY_SLEEP}"
    done
    for dataset in "${DATASET_LIST[@]}"; do
        run_eval_dataset "${HF_DIR}" "${OUTPUT_ROOT}" "${dataset}"
        SUMMARY_PATHS+=("${OUTPUT_ROOT}/${dataset}/summary.json")
    done
    stop_sglang_server
done
