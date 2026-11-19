#!/bin/bash
# Convert a HuggingFace checkpoint to the Megatron torch_dist format that slime
# trains from (needed as --ref-load / REF_LOAD for cold start, teacher RL, and
# distillation).
#
# Usage (Qwen3-4B-Instruct-2507, the default MODEL_CONFIG):
#   HF_CHECKPOINT=/path/to/Qwen3-4B-Instruct-2507 \
#   SAVE=/path/to/Qwen3-4B-Instruct-2507/torch_dist \
#       bash scripts/train/convert_hf_to_torch_dist.sh

set -ex

# tools/ is a verbatim copy of slime v0.3.1, so the two guards this conversion
# needs live here rather than as patches on the copied file.
#
# 1. Megatron's --attention-backend defaults to auto, and TransformerEngine
#    raises if any NVTE_*_ATTN override contradicts it. Every training script
#    here clears them for the same reason.
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_DIR}/third_party/slime"

# ===========================================================================
# KEY SETTINGS — every variable is env-overridable.
# ===========================================================================
# slime model-args file matching the base model (see scripts/models/).
# The rotary base is baked into the *-Instruct-2507 configs; for other configs
# export MODEL_ARGS_ROTARY_BASE before running if the default is wrong.
MODEL_CONFIG=${MODEL_CONFIG:-"${REPO_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"}
HF_CHECKPOINT=${HF_CHECKPOINT:?"set HF_CHECKPOINT to the HF checkpoint dir to convert"}
SAVE=${SAVE:?"set SAVE to the output torch_dist dir"}

# ---------------------------------------------------------------------------
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_DIR}:${PYTHONPATH:-}"

source "${MODEL_CONFIG}"

# 2. If MODEL_ARGS fails to expand, Megatron's parse_args leaves num_layers and
#    friends at None and the failure surfaces much later as an opaque TypeError.
if [ "${#MODEL_ARGS[@]}" -eq 0 ]; then
    echo "MODEL_ARGS is empty after sourcing ${MODEL_CONFIG}." >&2
    echo "Check that the model-args file defines MODEL_ARGS=(...) and that you are running under bash." >&2
    exit 1
fi

mkdir -p "${SAVE}"

python "${REPO_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${SAVE}"
