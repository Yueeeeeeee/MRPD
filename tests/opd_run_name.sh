#!/usr/bin/env bash
# The run name opd_multi.sh (or, with LAUNCHER=grpo, grpo_multi.sh) gives a set of knobs, from the
# launchers' own knob, replay and naming blocks evaluated under bash. Knobs: MULTI_DOMAIN_USE_TASK_REWARD
# (default 1), TEACHER_DENSE_KL / TEACHER_DENSE_KL_COEF / TEACHER_TOPK, *_TEACHER_CKPT_STEP (default 299),
# REPLAY=1 (a 300-step three-teacher manifest), REPLAY_GROUPS=<G> / REPLAY_SHUFFLE=<seed> (what the merge
# wrote into it). A refused combination exits with the launcher's message. Examples:
#   REPLAY=1 TEACHER_DENSE_KL=reverse bash tests/opd_run_name.sh        -> grpo-opd-3d-replay-dense-reverse-t299-qwen3-4b
#   LAUNCHER=grpo REPLAY=1 REPLAY_GROUPS=24 bash tests/opd_run_name.sh  -> grpo-3d-replay-g24-qwen3-4b
#   LAUNCHER=grpo REPLAY=1 REPLAY_LOSS=rft bash tests/opd_run_name.sh   -> rft-3d-replay-qwen3-4b (RFT_MAX_PER_PROMPT=1: -max1)
#   LAUNCHER=grpo RAFT=1 bash tests/opd_run_name.sh                     -> raft-3d-qwen3-4b (RFT_MAX_PER_PROMPT=1: -max1)
# Under LAUNCHER=grpo it also checks grpo_multi.sh's loss arguments: --loss-type sft_loss under REPLAY_LOSS or
# RAFT=1, --calculate-per-token-loss under RAFT=1 only (a mismatch exits 1 with the arguments).
set -eu
LAUNCHER=${LAUNCHER:-opd}
P="$(cd "$(dirname "$0")/.." && pwd)/scripts/train/${LAUNCHER}_multi.sh"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
REPLAY_LOGS=
if [ "${REPLAY:-0}" = "1" ]; then
   python3 - "$tmp" "${REPLAY_GROUPS:-}" "${REPLAY_SHUFFLE:-}" <<'EOF'
import json, sys
d, g, s = sys.argv[1:]
json.dump({"steps": 300, "prompts_per_step": 3 * (int(g) if g else 64), "n_samples_per_prompt": 8,
           "order": ["retool", "search-r1", "sql"], "groups_per_teacher": int(g) if g else None,
           "shuffle_seed": int(s) if s else None}, open(f"{d}/manifest.json", "w"))
EOF
   REPLAY_LOGS=$tmp
fi
OPD_TEACHER_SOURCE=megatron _DOMAIN=3d STUDENT=qwen3-4b MODEL=qwen3-4b
MULTI_DOMAIN_USE_TASK_REWARD=${MULTI_DOMAIN_USE_TASK_REWARD:-1}
MATH_TEACHER_CKPT_STEP=${MATH_TEACHER_CKPT_STEP:-299} SEARCH_R1_TEACHER_CKPT_STEP=${SEARCH_R1_TEACHER_CKPT_STEP:-299}
SQL_PROMPT_DATA=x SQL_TEACHER_CKPT_STEP=${SQL_TEACHER_CKPT_STEP:-299}
unset WANDB_GROUP TEACHER_TAG
# the lines from the first match of $1 to the first later match of $2
block() {
   local a b
   a=$(grep -n -- "$1" "$P" | head -1 | cut -d: -f1)
   b=$(awk -v s="$a" -v pat="$2" 'NR > s && $0 ~ pat { print NR; exit }' "$P")
   [ -n "$a" ] && [ -n "$b" ] || { echo "block '$1' .. '$2' not found in $P" >&2; exit 2; }
   sed -n "${a},${b}p" "$P"
}
if [ "${LAUNCHER}" = "opd" ]; then
   eval "$(block '^TEACHER_DENSE_KL=\${TEACHER_DENSE_KL:-}' '^# --- Replay' | sed '$d')"
   eval "$(block '^REPLAY_LOGS=\${REPLAY_LOGS:-}' '^fi$')" >/dev/null
   eval "$(block '^_teacher_tag() {' '^WANDB_GROUP=')"
else
   eval "$(block '^REPLAY_LOGS=\${REPLAY_LOGS:-}' '^fi$')" >/dev/null
   eval "$(block '^REPLAY_LOSS=\${REPLAY_LOSS:-}' '^fi$')"
   eval "$(block '^RAFT=\${RAFT:-0}' '^fi$')"
   eval "$(grep -- '^WANDB_GROUP=\${WANDB_GROUP:-\${_LOSS_PREFIX}-' "$P")"
   USE_TIS=${USE_TIS:-0}
   eval "$(block '^GRPO_ARGS=($' '^OPTIMIZER_ARGS=' | sed '$d')"
   want_sft=0 want_token=0
   if [ -n "${REPLAY_LOSS}" ] || [ "${RAFT}" = "1" ]; then want_sft=1; fi
   if [ "${RAFT}" = "1" ]; then want_token=1; fi
   has() { case " ${GRPO_ARGS[*]} " in *" $1 "*) echo 1 ;; *) echo 0 ;; esac; }
   [ "$(has '--loss-type sft_loss')$(has --calculate-per-token-loss)" = "${want_sft}${want_token}" ] || {
      echo "loss arguments '${GRPO_ARGS[*]}' under REPLAY_LOSS='${REPLAY_LOSS}' RAFT=${RAFT}" >&2; exit 1; }
fi
echo "${WANDB_GROUP}"
