#!/bin/bash
# Convert a Megatron torch_dist checkpoint back to HuggingFace format.
#
# Main use: turn a cold-start / SFT / RL checkpoint saved by slime (an
# iter_XXXXXXX/ directory) into an HF checkpoint — e.g. to re-materialize clean
# weights before teacher GRPO, or to serve/evaluate a trained model.
#
# Usage:
#   INPUT_DIR=/path/to/run/iter_0000154 \
#   OUTPUT_DIR=/path/to/run/hf \
#   ORIGIN_HF_DIR=/path/to/base_hf \
#       bash scripts/train/convert_torch_dist_to_hf.sh
#
#   INPUT_DIR      The torch_dist checkpoint directory to convert
#                  (the iter_XXXXXXX/ dir that holds common.pt).
#   OUTPUT_DIR     Where to write the HF checkpoint.
#   ORIGIN_HF_DIR  A base HF checkpoint of the same architecture; its
#                  tokenizer / config.json etc. are copied into OUTPUT_DIR.
#
# The model architecture is read from the checkpoint, so no MODEL_CONFIG is
# needed here (unlike the hf -> torch_dist direction).

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_DIR}/third_party/slime"

# ===========================================================================
# KEY SETTINGS — every variable is env-overridable.
# ===========================================================================
INPUT_DIR=${INPUT_DIR:?"set INPUT_DIR to the torch_dist iter_XXXXXXX dir to convert"}
OUTPUT_DIR=${OUTPUT_DIR:?"set OUTPUT_DIR to the HF output dir"}
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:?"set ORIGIN_HF_DIR to a base HF checkpoint (tokenizer/config donor)"}

# Overwrite OUTPUT_DIR if it already exists.
FORCE=${FORCE:-0}
# Extra flags passed through to convert_torch_dist_to_hf.py
# (e.g. "--add-missing-from-origin-hf" or "--vocab-size 151936").
CONVERT_EXTRA_ARGS=${CONVERT_EXTRA_ARGS:-}

# ---------------------------------------------------------------------------
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_DIR}:${PYTHONPATH:-}"

cmd=(
    python "${REPO_DIR}/tools/convert_torch_dist_to_hf.py"
    --input-dir "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --origin-hf-dir "${ORIGIN_HF_DIR}"
)
if [ "${FORCE}" = "1" ]; then
    cmd+=(--force)
fi
if [ -n "${CONVERT_EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    cmd+=(${CONVERT_EXTRA_ARGS})
fi

"${cmd[@]}"
