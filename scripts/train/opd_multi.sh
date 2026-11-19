#!/bin/bash

set -ex

unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export USER="${USER:-user}"
export LOGNAME="${LOGNAME:-$USER}"
export HOME="${HOME:-/tmp}"

export no_proxy="localhost,127.0.0.1,0.0.0.0,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,${NO_PROXY:-}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
RELEASE_DIR="${REPO_DIR}/train"
SRC_DIR="${RELEASE_DIR}/src"
SLIME_DIR="${REPO_DIR}/third_party/slime"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"

HF_CHECKPOINT=${HF_CHECKPOINT:?"set HF_CHECKPOINT to the cold-start HF checkpoint"}
REF_LOAD=${REF_LOAD:?"set REF_LOAD to the cold-start torch_dist checkpoint"}
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

SAVE_DIR=${SAVE_DIR:?"set SAVE_DIR to the student output dir"}
ROLLOUT_DEBUG_DIR=${ROLLOUT_DEBUG_DIR:-${SAVE_DIR}/rollout_debug}

RETOOL_PROMPT_DATA=${RETOOL_PROMPT_DATA:?"set RETOOL_PROMPT_DATA to the math RL prompt file (baohao/math-train)"}
SEARCH_R1_PROMPT_DATA=${SEARCH_R1_PROMPT_DATA:?"set SEARCH_R1_PROMPT_DATA to the search RL prompt file (baohao/search-train)"}
# A third domain, text-to-SQL, when SQL_PROMPT_DATA is set (prepare_sql.py's train.jsonl,
# SQL_DB_ROOT its databases/): the batch then cycles retool,search-r1,sql -- 24 prompts
# each at the 72-prompt default below (576 sequences, against 64 x 8 = 512 for the
# teachers and the two-domain runs) -- the SQL teacher joins the teacher map, and the
# run name says 3d where the two-domain runs say both. The SQL rollout's own knobs (tool calls, turns, timeouts) keep
# teacher_grpo_sql.sh's values; SQL_MAX_CONTEXT_LEN narrows its context to 9216.
SQL_PROMPT_DATA=${SQL_PROMPT_DATA:-}
if [ -n "${SQL_PROMPT_DATA}" ]; then
   [ -f "${SQL_PROMPT_DATA}" ] || { echo "[opd] SQL_PROMPT_DATA=${SQL_PROMPT_DATA} does not exist." >&2; exit 1; }
   SQL_DB_ROOT=${SQL_DB_ROOT:?"set SQL_DB_ROOT to prepare_sql.py's databases/ directory"}
   [ -d "${SQL_DB_ROOT}" ] || { echo "[opd] SQL_DB_ROOT=${SQL_DB_ROOT} is not a directory." >&2; exit 1; }
   MIXED_TASK_ORDER=${MIXED_TASK_ORDER:-retool,search-r1,sql}
   _DOMAIN=3d
else
   SQL_DB_ROOT=${SQL_DB_ROOT:-}
   _DOMAIN=both
fi
case ",${MIXED_TASK_ORDER:-}," in
   *,sql,*) [ -n "${SQL_PROMPT_DATA}" ] || { echo "[opd] MIXED_TASK_ORDER names sql but SQL_PROMPT_DATA is unset." >&2; exit 1; } ;;
esac
SQL_INPUT_KEY=${SQL_INPUT_KEY:-prompt}
SQL_LABEL_KEY=${SQL_LABEL_KEY:-label}
SQL_METADATA_KEY=${SQL_METADATA_KEY:-metadata}
SQL_MAX_CONTEXT_LEN=${SQL_MAX_CONTEXT_LEN:-9216}
SQL_MAX_TOOL_CALLS=${SQL_MAX_TOOL_CALLS:-6}
SQL_MAX_TURNS=${SQL_MAX_TURNS:-8}
SQL_MAX_TOKENS_PER_TURN=${SQL_MAX_TOKENS_PER_TURN:-4096}
SQL_EXEC_TIMEOUT=${SQL_EXEC_TIMEOUT:-5}
SQL_SCORE_TIMEOUT=${SQL_SCORE_TIMEOUT:-30}
SQL_MAX_RESULT_ROWS=${SQL_MAX_RESULT_ROWS:-20}
SQL_MAX_RESULT_CHARS=${SQL_MAX_RESULT_CHARS:-4000}
SQL_TOOL_CONCURRENCY=${SQL_TOOL_CONCURRENCY:-128}

RETOOL_INPUT_KEY=${RETOOL_INPUT_KEY:-prompt}
RETOOL_LABEL_KEY=${RETOOL_LABEL_KEY:-label}
RETOOL_METADATA_KEY=${RETOOL_METADATA_KEY:-metadata}
SEARCH_R1_INPUT_KEY=${SEARCH_R1_INPUT_KEY:-prompt}
SEARCH_R1_LABEL_KEY=${SEARCH_R1_LABEL_KEY:-reward_model}
SEARCH_R1_METADATA_KEY=${SEARCH_R1_METADATA_KEY:-metadata}

# Repeating a task changes the within-batch ratio:
#   retool,search-r1             -> roughly 1:1
#   retool,search-r1,search-r1   -> roughly 1:2
MIXED_TASK_ORDER=${MIXED_TASK_ORDER:-retool,search-r1}

# Set to 1 to wipe SAVE_DIR before training. Off by default so that --load can
# resume an interrupted run.
CLEAN_SAVE_DIR=${CLEAN_SAVE_DIR:-0}
if [ "${CLEAN_SAVE_DIR}" = "1" ]; then
   rm -rf "${SAVE_DIR}"
fi
# The math teacher's variables say MATH_, not RETOOL_ (the *task* is still
# "retool"). A stale RETOOL_TEACHER_* in the shell would simply stop being read,
# and most of them have defaults, so the run would start with the wrong one.
_stale=""
for _v in $(env | cut -d= -f1); do
   case "${_v}" in
      RETOOL_TEACHER_*|START_RETOOL_TEACHER|OPD_TEACHER_RETOOL_MAX_INFLIGHT)
         _stale="${_stale} ${_v}" ;;
   esac
done
if [ -n "${_stale}" ]; then
   echo "These variables were renamed and are no longer read:${_stale}" >&2
   echo "  RETOOL_TEACHER_*                -> MATH_TEACHER_*" >&2
   echo "  START_RETOOL_TEACHER            -> START_MATH_TEACHER" >&2
   echo "  OPD_TEACHER_RETOOL_MAX_INFLIGHT -> OPD_TEACHER_MATH_MAX_INFLIGHT" >&2
   exit 1
fi

# --- where the teacher log-probs come from ----------------------------------
# megatron (default): both teachers are weight sets inside the trainer, via
#           --opd-teacher-loads. No servers, no HF conversion, no GPU reserved
#           -- all 8 go to the student. It is also the setting under which both
#           sides of the OPD KL are measured on the same engine: the OPD KL is
#           student_logp - teacher_logp, and here both sides come out of the
#           same Megatron forward. Served teachers put the student side on the
#           rollout engine's incremental decode and the teacher side on a
#           separate server's one-shot prefill, so any systematic difference
#           between those two numerical paths lands in the advantage with a
#           uniform sign -- and a served teacher can also simply fail to answer,
#           which the first run of this arm did.
# sglang:   both teachers served on their own GPUs and queried over HTTP. Keep
#           for a teacher whose architecture differs from the student's, which
#           the Megatron path cannot express.
OPD_TEACHER_SOURCE=${OPD_TEACHER_SOURCE:-megatron}
case "${OPD_TEACHER_SOURCE}" in
   megatron|sglang) ;;
   *) echo "OPD_TEACHER_SOURCE must be 'megatron' or 'sglang', got '${OPD_TEACHER_SOURCE}'" >&2
      exit 1 ;;
esac


# TEACHER_DENSE_KL=reverse|forward|js (run name -dense-reverse / -dense-forward / -dense-js; until
# 2026-11-01 -dklrev / -dklfwd / -dkljs): the teacher term on the teacher's whole next-token
# distribution instead of at the sampled token -- the KL between the student's and the teacher's
# (top-k + one tail atom) distributions at every model token, the teacher's top-k (TEACHER_TOPK
# atoms, default 256) taken in the Megatron teachers' forward, the student's log-probs at those ids
# out of the policy loss's own log-prob function (one full-vocabulary buffer), the term
# differentiated directly in the loss: no importance ratio, no clip
# (train/src/teacher_dense_kl.py through patches/slime-opd-dense-kl.patch). reverse is
# KL(student || teacher), the quantity slime's sampled-token term estimates; forward is
# KL(teacher || student); js is the Jensen-Shannon divergence. Weight TEACHER_DENSE_KL_COEF
# (default 1; in the name when not 1, as the term's beta: -dense-reverse-b0.3; TEACHER_TOPK in the
# name when not 256: -dense-reverse-k128). slime's own sampled-token term goes to 0 (its
# opd_reverse_kl is still logged, as is train/teacher_kl-<domain>); the task term is untouched.
# Logged: train/teacher_dense_kl (the term before its weight), train/teacher_topk_hit (the share of
# logged tokens inside the teacher's top-k), train/teacher_dense_err_teacher (the top-k against
# slime's own log-softmax at the logged token, ~1e-5 when the tensor-parallel paths are right) and
# train/teacher_dense_err_student (the fused gather against the same function's log-prob there,
# 0 up to rounding).
TEACHER_DENSE_KL=${TEACHER_DENSE_KL:-}
_DKL_TAG=
if [ -n "${TEACHER_DENSE_KL}" ]; then
   case "${TEACHER_DENSE_KL}" in
      reverse) _DKL_TAG=-dense-reverse ;;
      forward) _DKL_TAG=-dense-forward ;;
      js) _DKL_TAG=-dense-js ;;
      *) echo "[opd] TEACHER_DENSE_KL must be reverse, forward or js, got '${TEACHER_DENSE_KL}'" >&2; exit 1 ;;
   esac
   [ "${OPD_TEACHER_SOURCE}" = "megatron" ] || { echo "[opd] TEACHER_DENSE_KL needs OPD_TEACHER_SOURCE=megatron: the top-k comes from the trainer's teacher forward" >&2; exit 1; }
   [ -z "${OPD_KL_COEF:-}" ] || { echo "[opd] OPD_KL_COEF with TEACHER_DENSE_KL: the dense term replaces slime's sampled-token term (its coefficient goes to 0); its weight is TEACHER_DENSE_KL_COEF" >&2; exit 1; }
   TEACHER_DENSE_KL_COEF=${TEACHER_DENSE_KL_COEF:-1}
   [[ "${TEACHER_DENSE_KL_COEF}" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "[opd] TEACHER_DENSE_KL_COEF must be a positive number, got '${TEACHER_DENSE_KL_COEF}'" >&2; exit 1; }
   TEACHER_TOPK=${TEACHER_TOPK:-256}
   [[ "${TEACHER_TOPK}" =~ ^[1-9][0-9]*$ ]] || { echo "[opd] TEACHER_TOPK must be a positive integer, got '${TEACHER_TOPK}'" >&2; exit 1; }
   [ "${TEACHER_DENSE_KL_COEF}" = "1" ] || _DKL_TAG="${_DKL_TAG}-b${TEACHER_DENSE_KL_COEF}"
   [ "${TEACHER_TOPK}" = "256" ] || _DKL_TAG="${_DKL_TAG}-k${TEACHER_TOPK}"
else
   [ -z "${TEACHER_DENSE_KL_COEF:-}${TEACHER_TOPK:-}" ] || { echo "[opd] TEACHER_DENSE_KL_COEF / TEACHER_TOPK without TEACHER_DENSE_KL" >&2; exit 1; }
fi
# OPD_KL_COEF=<beta> (run name -b<beta> when not 1): the weight of slime's sampled-token teacher
# term, the advantage log pi_T - log pi at the sampled token; 1 is OPD / GRPO+OPD as published
# (every run so far). Refused with the dense term, which sets that weight itself.
_B_TAG=
if [ -n "${OPD_KL_COEF:-}" ]; then
   [[ "${OPD_KL_COEF}" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "[opd] OPD_KL_COEF must be a non-negative number, got '${OPD_KL_COEF}'" >&2; exit 1; }
   case "${OPD_KL_COEF}" in 1|1.0) ;; *) _B_TAG=-b${OPD_KL_COEF} ;; esac
fi

# --- Replay (REPLAY_LOGS=<dir>; run name ...-replay-t<steps>) --------------------
# As grpo_multi.sh's: no rollout, step k's batch is tools/merge_rollout_logs.py's
# rollout_k.pt under REPLAY_LOGS (the teachers' own step-k samples with their rewards
# and sampling log-probs), trained on with this arm's loss -- the Megatron teachers'
# term computed in the trainer on the replayed tokens, the task term from the logged
# rewards, the ratio against the dump's rollout_log_probs -- through slime's
# --load-debug-rollout-data: no sglang engine, the trainer on every GPU. Batch
# shape, step cap and domain tag come from the directory's manifest.json. Served
# teachers are refused (their log-probs are taken in the generate function, which
# never runs here); the retriever is not asked; the prompt files still have to exist.
REPLAY_LOGS=${REPLAY_LOGS:-}
_REPLAY_TAG=
if [ -n "${REPLAY_LOGS}" ]; then
   [ -f "${REPLAY_LOGS}/manifest.json" ] || { echo "[opd] REPLAY_LOGS=${REPLAY_LOGS} has no manifest.json (tools/merge_rollout_logs.py writes it)." >&2; exit 1; }
   [ "${OPD_TEACHER_SOURCE}" = "megatron" ] || { echo "[opd] REPLAY_LOGS needs OPD_TEACHER_SOURCE=megatron: served teachers score in the generate function, which a replay never calls." >&2; exit 1; }
   [ "${USE_ROLLOUT_LOGPROBS:-1}" = "1" ] || { echo "[opd] REPLAY_LOGS needs USE_ROLLOUT_LOGPROBS=1: the ratio is to the dump's sampling log-probs." >&2; exit 1; }
   [ "${ASYNC:-0}" != "1" ] || { echo "[opd] REPLAY_LOGS with ASYNC=1: nothing to overlap, the batches are read from disk." >&2; exit 1; }
   read -r _REPLAY_STEPS _REPLAY_PROMPTS _REPLAY_N _REPLAY_TASKS _REPLAY_SUBTAG <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
tag = ("-g%d" % m["groups_per_teacher"] if m.get("groups_per_teacher") else "") + ("-shuf%d" % m["shuffle_seed"] if m.get("shuffle_seed") is not None else "")
print(m["steps"], m["prompts_per_step"], m["n_samples_per_prompt"], ",".join(m["order"]), tag or "none")' "${REPLAY_LOGS}/manifest.json")"
   [ -n "${_REPLAY_SUBTAG:-}" ] || { echo "[opd] ${REPLAY_LOGS}/manifest.json is not one merge_rollout_logs.py wrote." >&2; exit 1; }
   [ "${_REPLAY_SUBTAG}" != "none" ] || _REPLAY_SUBTAG=
   case "${_REPLAY_TASKS}" in
      retool,search-r1,sql) _DOMAIN=3d ;;
      retool,search-r1) _DOMAIN=both ;;
      retool) _DOMAIN=math ;;
      search-r1) _DOMAIN=search ;;
      sql) _DOMAIN=sql ;;
      *) _DOMAIN=${_REPLAY_TASKS//,/-} ;;
   esac
   # -replay, then -g<groups> (a subset of each teacher's groups per step) and -shuf<seed> (the
   # steps in a shuffled order) as the merge recorded them in the manifest.
   _REPLAY_TAG=-replay${_REPLAY_SUBTAG}
   ASYNC=0
   ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}
   if [ -n "${ROLLOUT_BATCH_SIZE:-}" ] && [ "${ROLLOUT_BATCH_SIZE}" != "${_REPLAY_PROMPTS}" ]; then
      echo "[opd] ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}, but ${REPLAY_LOGS} holds ${_REPLAY_PROMPTS} prompts per step: the batch is what was logged." >&2; exit 1
   fi
   if [ -n "${N_SAMPLES_PER_PROMPT:-}" ] && [ "${N_SAMPLES_PER_PROMPT}" != "${_REPLAY_N}" ]; then
      echo "[opd] N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}, but ${REPLAY_LOGS} was merged at ${_REPLAY_N}." >&2; exit 1
   fi
   ROLLOUT_BATCH_SIZE=${_REPLAY_PROMPTS}
   N_SAMPLES_PER_PROMPT=${_REPLAY_N}
   if [ "${NUM_ROLLOUT:-500}" -gt "${_REPLAY_STEPS}" ]; then
      echo "[opd] replay: ${REPLAY_LOGS} holds ${_REPLAY_STEPS} steps; NUM_ROLLOUT ${NUM_ROLLOUT:-500} -> ${_REPLAY_STEPS}"
      NUM_ROLLOUT=${_REPLAY_STEPS}
   fi
   echo "[opd] replay of ${REPLAY_LOGS}: ${_REPLAY_TASKS}, ${_REPLAY_PROMPTS} prompts x ${_REPLAY_N} per step, ${NUM_ROLLOUT:-500} steps, the trainer on all GPUs"
fi

# --- the slime patch this arm depends on ---------------------------------------
# restore-actor-after-teacher-load, for the *reference* model rather than the
# teacher. This arm serves its teachers over HTTP, so nothing loads a teacher
# into the live model -- but --use-kl-loss does load a ref through the same
# load_other_checkpoint, and slime only restores the actor afterwards under
# `if args.offload_train`, which is False here. Unpatched, a *resumed* run
# pushes the SFT reference to the rollout engines and generates its first two
# rollouts from it instead of the trained actor, then trains on them. Silent.
# A fresh run is unaffected (ref == the actor's init), which is why this can sit
# unnoticed until the first resume.
#
# decouple-student-logprobs is deliberately NOT applied here. It moves the
# student side of the OPD KL onto Megatron so it matches a Megatron teacher.
# This arm's teachers are served by SGLang, so leaving it off keeps both sides
# of the KL on the inference stack, which is the pairing that makes the two
# log-probs comparable.
#
# Applied here rather than left to the operator because the failure is silent.
# A plain `git pull` does not disturb it -- only `git submodule update` (or a
# fresh clone) resets the submodule, and then it re-applies. Idempotent: the
# marker grep is the guard, not `git apply`.
# Set AUTO_PATCH_SLIME=0 to require it to be applied by hand instead.
AUTO_PATCH_SLIME=${AUTO_PATCH_SLIME:-1}

# _apply_slime_patch <patch> <marker> <target> [legacy-marker ...]
#
# The legacy markers are markers used by *earlier versions of these same
# patches*. If one is present and the current marker is not, the file carries a
# superseded version of our own patch: `git apply` would fail on the changed
# context, so reset that one file first. Only ever reset on a marker we
# ourselves wrote -- an unrecognised local modification is reported, never
# discarded.
_apply_slime_patch() {
   local patch_name="$1" marker="$2" target_rel="$3"
   shift 3
   local legacy=("$@")
   local patch_file="${REPO_DIR}/patches/${patch_name}"
   local target="${SLIME_DIR}/${target_rel}"

   if [ ! -f "${target}" ]; then
      echo "[opd] ${target} not found -- is the slime submodule checked out?" >&2
      echo "[opd]   git submodule sync --recursive && git submodule update --init --recursive" >&2
      exit 1
   fi
   if grep -q "${marker}" "${target}"; then
      echo "[opd] ${patch_name}: already applied"
      return 0
   fi
   if [ "${AUTO_PATCH_SLIME}" != "1" ]; then
      echo "[opd] ${target} is missing ${patch_name} and AUTO_PATCH_SLIME=0." >&2
      echo "[opd] Apply it:  cd ${SLIME_DIR} && git apply ${patch_file}" >&2
      exit 1
   fi
   local lm
   for lm in ${legacy[@]+"${legacy[@]}"}; do
      if grep -q "${lm}" "${target}"; then
         echo "[opd] ${patch_name}: found a superseded version of this patch in"
         echo "[opd]   ${target_rel} (marker: ${lm}); resetting that file first."
         # -c safe.directory is scoped to this one invocation. git refuses to
         # operate on a repo owned by another uid ("dubious ownership"), which is
         # the norm for a bind-mounted checkout inside a container running as
         # root. `git apply` does not hit this (it works on the worktree and can
         # run outside a repo at all); `git checkout` does.
         if ! (cd "${SLIME_DIR}" && git -c safe.directory="${SLIME_DIR}" checkout -- "${target_rel}"); then
            echo "[opd] Could not reset ${target_rel}." >&2
            echo "[opd] Reset it by hand and re-run:" >&2
            echo "[opd]   git config --global --add safe.directory ${SLIME_DIR}" >&2
            echo "[opd]   git -C ${SLIME_DIR} checkout -- slime/backends/megatron_utils/" >&2
            exit 1
         fi
         break
      fi
   done
   echo "[opd] ${patch_name}: applying"
   if ! (cd "${SLIME_DIR}" && git -c safe.directory="${SLIME_DIR}" apply "${patch_file}"); then
      echo "[opd] Failed to apply ${patch_file}." >&2
      echo "[opd] The submodule may have moved off the pinned commit, or the file" >&2
      echo "[opd] carries edits this script does not recognise. Inspect:" >&2
      echo "[opd]   cd ${SLIME_DIR} && git status && git log --oneline -1" >&2
      echo "[opd] If you have no edits of your own there, discarding them fixes it:" >&2
      echo "[opd]   git config --global --add safe.directory ${SLIME_DIR}" >&2
      echo "[opd]   git -C ${SLIME_DIR} checkout -- slime/backends/megatron_utils/" >&2
      exit 1
   fi
   # Trust the marker, not git apply's exit code.
   if ! grep -q "${marker}" "${target}"; then
      echo "[opd] git apply reported success but the marker is absent." >&2
      exit 1
   fi
   echo "[opd] ${patch_name}: applied"
}

# The marker must be unique to the patch. `if self._active_model_tag != "actor":`
# is NOT usable -- that exact line already appears in the training loop, so it
# matches unpatched source and the patch gets silently skipped.
# --- a superseded version of our own patches ---------------------------------
# _apply_slime_patch's marker answers "is this patched", not "is this patched at
# the version this checkout ships". When a patch's *content* changes, the old
# marker is still there, the helper reports "already applied", and the run goes
# out with the previous version -- which is how this arm reached argparse with a
# stale arguments.py and failed on an argument format that had already been
# fixed, twice, with ray reusing the identical working-dir package hash both
# times because the tree genuinely had not changed.
#
# So look for markers that exist ONLY in superseded versions, and when one is
# found reset the whole slime/ tree so all three patches re-apply in order. The
# patches build on each other -- multi-teacher is cut against a tree that
# already has the other two -- so resetting one file and re-applying one patch
# is not enough.
#
# Only ever triggered by a marker we wrote ourselves; an unrecognised local edit
# under slime/ is left alone and will surface as a git apply failure instead.
_SUPERSEDED_MARKERS=(
   # v1 of slime-opd-multi-teacher parsed the teacher map as JSON only
   'slime/utils/arguments.py|loads = json.loads(args.opd_teacher_loads)'
   # v1 of slime-opd-dense-kl (2026-10-31, marker 'slime-opd-dense-kl:') took the student side of the
   # dense term through autograd next to slime's log-prob function, a second full-vocabulary buffer,
   # and ran out of memory; v2 ('slime-opd-dense-kl-fused') is fused into that function
   # (2026-11-01)
   'slime/backends/megatron_utils/loss.py|slime-opd-dense-kl:'
   # before 2026-11-07, slime-opd-multi-teacher forwarded every
   # teacher over the whole batch; the by-source forward replaced it
   'slime/backends/megatron_utils/actor.py|teacher therefore also scores samples that are not its own'
   # before 2026-11-07, slime-opd-dense-kl-fused ran the student's standalone log-prob forward that nothing
   # read under --opd-kl-coef 0 (22 s of the replay arms' step); the version that skips it rewrote this line
   'slime/backends/megatron_utils/actor.py|with --opd-dense-kl also stores the top-k ids'
)
_stale_marker=""
for _entry in "${_SUPERSEDED_MARKERS[@]}"; do
   _f="${SLIME_DIR}/${_entry%%|*}"
   if [ -f "${_f}" ] && grep -qF -- "${_entry#*|}" "${_f}"; then
      _stale_marker="${_entry}"
      break
   fi
done
if [ -n "${_stale_marker}" ]; then
   echo "[opd] slime carries a superseded version of these patches"
   echo "[opd]   (${_stale_marker%%|*}: ${_stale_marker#*|})"
   echo "[opd] resetting the slime tree so all of them re-apply at this checkout's version"
   # The whole tree, not slime/ alone: a patch may edit files outside slime/, and a tree left
   # with only some of its hunks would refuse it the next time.
   if ! (cd "${SLIME_DIR}" && git -c safe.directory="${SLIME_DIR}" checkout -- .); then
      echo "[opd] Could not reset ${SLIME_DIR}. Reset it by hand and re-run:" >&2
      echo "[opd]   git config --global --add safe.directory ${SLIME_DIR}" >&2
      echo "[opd]   git -C ${SLIME_DIR} checkout -- ." >&2
      exit 1
   fi
fi

_apply_slime_patch slime-opd-restore-actor-after-teacher-load.patch \
   'slime-opd-patch' \
   slime/backends/megatron_utils/actor.py \
   'Recover to actor in the end. Unconditional'

if [ "${OPD_TEACHER_SOURCE}" = "megatron" ]; then
   # Both sides of the OPD KL on the trainer's own forward. Required here
   # because, with a Megatron
   # teacher, taking the student side from the rollout engine would put
   # cross-engine disagreement straight into the advantage.
   _apply_slime_patch slime-opd-decouple-student-logprobs.patch \
      'rollout_data.get("log_probs", log_probs)' \
      slime/backends/megatron_utils/loss.py \
      'student_opd_log_probs'
   # A teacher per sample source. Upstream slime loads exactly one.
   _apply_slime_patch slime-opd-multi-teacher.patch \
      'slime-opd-multi-teacher' \
      slime/backends/megatron_utils/actor.py
   if [ -n "${TEACHER_DENSE_KL}" ]; then
      # The dense teacher term: the teachers' top-k stored in their forward, the student's
      # log-probs at those ids out of the policy loss's own log-prob function
      # (train/src/teacher_dense_kl.py). Cut against a tree with the three patches above. The first version's marker ('slime-opd-dense-kl:') is in
      # _SUPERSEDED_MARKERS above: a tree still carrying it is reset before any patch is applied.
      _apply_slime_patch slime-opd-dense-kl.patch \
         'slime-opd-dense-kl-fused' \
         slime/backends/megatron_utils/loss.py
   fi
fi
if [ -n "${REPLAY_LOGS}" ]; then
   # The next step's dump read in a thread while this one trains (rollout.py only, independent of
   # the patches above; the replay's perf/train_wait_time was ~6 s of a 179 s step before 2026-11-07).
   _apply_slime_patch slime-opd-replay-prefetch.patch \
      'slime-opd-replay-prefetch' \
      slime/ray/rollout.py
fi

# Assert, rather than trust the block above. The first run of this arm submitted
# --opd-teacher-loads against an unpatched slime, which rejects it with
# "--opd-teacher-load is required when --opd-type=megatron" -- a message about a
# flag the command line does not even contain. The cause was ordering:
# OPD_TEACHER_SOURCE was still unset when that `if` was evaluated, so both
# conditional patches were skipped silently. Ordering is fixed, and this checks
# the outcome instead of the reasoning.
if [ "${OPD_TEACHER_SOURCE}" = "megatron" ]; then
   for _need in \
      'slime-opd-multi-teacher:slime/backends/megatron_utils/actor.py' \
      '_parse_teacher_map:slime/utils/arguments.py' \
      'rollout_data.get("log_probs", log_probs):slime/backends/megatron_utils/loss.py'; do
      _marker="${_need%%:*}"; _file="${SLIME_DIR}/${_need##*:}"
      if ! grep -q "${_marker}" "${_file}"; then
         echo "[opd] ${_file} is missing the patch marker '${_marker}'." >&2
         echo "[opd] --opd-type megatron with two teachers cannot work unpatched:" >&2
         echo "[opd] slime would reject --opd-teacher-loads as an unknown flag and" >&2
         echo "[opd] then demand --opd-teacher-load instead." >&2
         exit 1
      fi
   done
   if [ -n "${TEACHER_DENSE_KL}" ]; then
      for _f in slime/backends/megatron_utils/loss.py slime/backends/megatron_utils/actor.py \
                slime/backends/megatron_utils/model.py slime/backends/megatron_utils/data.py slime/utils/arguments.py; do
         grep -q 'slime-opd-dense-kl-fused' "${SLIME_DIR}/${_f}" || { echo "[opd] ${_f} is missing the patch marker 'slime-opd-dense-kl-fused'." >&2; exit 1; }
      done
   fi
   echo "[opd] megatron teacher patches verified present"
fi

mkdir -p "${SAVE_DIR}" "${ROLLOUT_DEBUG_DIR}"

launch_teacher() {
   local name="$1"
   local model_path="$2"
   local gpus="$3"
   local port="$4"
   local tp="$5"
   local dp="$6"
   local mem_fraction="$7"
   local max_running_requests="$8"
   local schedule_conservativeness="$9"
   local max_total_tokens="${10}"
   local extra_args="${11:-}"
   # Only cap the KV-cache token budget when explicitly requested. Leaving
   # --max-total-tokens unset lets SGLang auto-size from --mem-fraction-static.
   # A small cap starves the teacher's KV cache: each teacher logprob request
   # prefills the FULL prompt+response, so only a couple of sequences fit at
   # once and the in-flight requests queue and time out -> retries.
   local max_total_tokens_arg=""
   if [ -n "${max_total_tokens}" ]; then
      max_total_tokens_arg="--max-total-tokens '${max_total_tokens}'"
   fi
   local log_path="${SAVE_DIR}/sglang_${name}_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

   CUDA_VISIBLE_DEVICES=${gpus} \
      bash -c "cd /tmp && exec python3 -m sglang.launch_server \
      --model-path '${model_path}' \
      --host 0.0.0.0 \
      --port '${port}' \
      --tp '${tp}' \
      --dp '${dp}' \
      --chunked-prefill-size '${TEACHER_CHUNKED_PREFILL_SIZE:-2048}' \
      --mem-fraction-static '${mem_fraction}' \
      --max-running-requests '${max_running_requests}' \
      ${max_total_tokens_arg} \
      --schedule-conservativeness '${schedule_conservativeness}' \
      --disable-radix-cache \
      ${extra_args}" \
      > "${log_path}" 2>&1 &

   local pid=$!
   echo "${pid}:${log_path}"
}

TEACHER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"

if [ "${OPD_TEACHER_SOURCE}" = "megatron" ]; then
   START_MATH_TEACHER=${START_MATH_TEACHER:-0}
   START_SEARCH_R1_TEACHER=${START_SEARCH_R1_TEACHER:-0}
else
   START_MATH_TEACHER=${START_MATH_TEACHER:-1}
   START_SEARCH_R1_TEACHER=${START_SEARCH_R1_TEACHER:-1}
fi

if [ "${OPD_TEACHER_SOURCE}" = "megatron" ]; then
   # teacher_grpo_math.sh / teacher_grpo_search.sh's SAVE_DIR: a torch_dist run
   # directory, NOT an HF export and NOT an iter_* subdirectory. Nothing to
   # convert -- this is the format those runs already wrote.
   #
   # The step is pinned rather than defaulted to the newest iteration:
   # the teacher budget is an independent variable of this
   # experiment, and a run that followed latest_checkpointed_iteration.txt would
   # distil against a different teacher after a resume.
   MATH_TEACHER_LOAD=${MATH_TEACHER_LOAD:?"set MATH_TEACHER_LOAD to teacher_grpo_math.sh's SAVE_DIR (torch_dist), or use OPD_TEACHER_SOURCE=sglang"}
   MATH_TEACHER_CKPT_STEP=${MATH_TEACHER_CKPT_STEP:?"set MATH_TEACHER_CKPT_STEP to the math teacher iteration to distil from, e.g. 499"}
   SEARCH_R1_TEACHER_LOAD=${SEARCH_R1_TEACHER_LOAD:?"set SEARCH_R1_TEACHER_LOAD to teacher_grpo_search.sh's SAVE_DIR (torch_dist), or use OPD_TEACHER_SOURCE=sglang"}
   SEARCH_R1_TEACHER_CKPT_STEP=${SEARCH_R1_TEACHER_CKPT_STEP:?"set SEARCH_R1_TEACHER_CKPT_STEP to the search teacher iteration to distil from, e.g. 499"}
   if [ -n "${SQL_PROMPT_DATA}" ]; then
      SQL_TEACHER_LOAD=${SQL_TEACHER_LOAD:?"set SQL_TEACHER_LOAD to teacher_grpo_sql.sh's SAVE_DIR (torch_dist)"}
      SQL_TEACHER_CKPT_STEP=${SQL_TEACHER_CKPT_STEP:?"set SQL_TEACHER_CKPT_STEP to the SQL teacher iteration to distil from, e.g. 99"}
   fi
   for _t in MATH SEARCH_R1 ${SQL_PROMPT_DATA:+SQL}; do
      eval "_dir=\${${_t}_TEACHER_LOAD}; _step=\${${_t}_TEACHER_CKPT_STEP}"
      if [ ! -d "${_dir}" ]; then
         echo "[opd] ${_t}_TEACHER_LOAD=${_dir} does not exist." >&2
         exit 1
      fi
      _iter="${_dir}/iter_$(printf '%07d' "${_step}")"
      if [ ! -d "${_iter}" ]; then
         echo "[opd] ${_dir} has no iter_$(printf '%07d' "${_step}") directory." >&2
         echo "[opd] Available:" >&2
         ls -d "${_dir}"/iter_* 2>/dev/null >&2 || echo "[opd]   (none)" >&2
         exit 1
      fi
   done
   MATH_TEACHER_MODEL_PATH=""
   SEARCH_R1_TEACHER_MODEL_PATH=""
else
   if [ -n "${SQL_PROMPT_DATA}" ]; then
      echo "[opd] the SQL teacher is only served from inside the trainer (OPD_TEACHER_SOURCE=megatron, the default)." >&2
      exit 1
   fi
   MATH_TEACHER_MODEL_PATH=${MATH_TEACHER_MODEL_PATH:?"set MATH_TEACHER_MODEL_PATH to the math teacher HF checkpoint"}
   SEARCH_R1_TEACHER_MODEL_PATH=${SEARCH_R1_TEACHER_MODEL_PATH:?"set SEARCH_R1_TEACHER_MODEL_PATH to the search teacher HF checkpoint"}
fi

MATH_TEACHER_GPUS=${MATH_TEACHER_GPUS:-4,5}
MATH_TEACHER_PORT=${MATH_TEACHER_PORT:-13141}
MATH_TEACHER_TP=${MATH_TEACHER_TP:-1}
MATH_TEACHER_DP=${MATH_TEACHER_DP:-2}

SEARCH_R1_TEACHER_GPUS=${SEARCH_R1_TEACHER_GPUS:-6,7}
SEARCH_R1_TEACHER_PORT=${SEARCH_R1_TEACHER_PORT:-13142}
SEARCH_R1_TEACHER_TP=${SEARCH_R1_TEACHER_TP:-1}
SEARCH_R1_TEACHER_DP=${SEARCH_R1_TEACHER_DP:-2}

MATH_TEACHER_PID=""
SEARCH_R1_TEACHER_PID=""

cleanup_teachers() {
   if [ -n "${MATH_TEACHER_PID:-}" ]; then
      kill "${MATH_TEACHER_PID}" 2>/dev/null || true
   fi
   if [ -n "${SEARCH_R1_TEACHER_PID:-}" ]; then
      kill "${SEARCH_R1_TEACHER_PID}" 2>/dev/null || true
   fi
}
trap cleanup_teachers EXIT

if [ "${START_MATH_TEACHER}" = "1" ]; then
   MATH_LAUNCH_INFO=$(
      launch_teacher \
         retool \
         "${MATH_TEACHER_MODEL_PATH}" \
         "${MATH_TEACHER_GPUS}" \
         "${MATH_TEACHER_PORT}" \
         "${MATH_TEACHER_TP}" \
         "${MATH_TEACHER_DP}" \
         "${MATH_TEACHER_MEM_FRACTION_STATIC:-0.85}" \
         "${MATH_TEACHER_MAX_RUNNING_REQUESTS:-128}" \
         "${MATH_TEACHER_SCHEDULE_CONSERVATIVENESS:-0.3}" \
         "${MATH_TEACHER_MAX_TOTAL_TOKENS:-131072}" \
         "${MATH_TEACHER_EXTRA_ARGS:-}"
   )
   MATH_TEACHER_PID="${MATH_LAUNCH_INFO%%:*}"
   MATH_TEACHER_LOG="${MATH_LAUNCH_INFO#*:}"
   MATH_TEACHER_URL="http://${TEACHER_IP}:${MATH_TEACHER_PORT}/generate"
else
   MATH_TEACHER_URL=${MATH_TEACHER_URL:-http://127.0.0.1:13141/generate}
fi

if [ "${START_SEARCH_R1_TEACHER}" = "1" ]; then
   SEARCH_R1_LAUNCH_INFO=$(
      launch_teacher \
         search_r1 \
         "${SEARCH_R1_TEACHER_MODEL_PATH}" \
         "${SEARCH_R1_TEACHER_GPUS}" \
         "${SEARCH_R1_TEACHER_PORT}" \
         "${SEARCH_R1_TEACHER_TP}" \
         "${SEARCH_R1_TEACHER_DP}" \
         "${SEARCH_R1_TEACHER_MEM_FRACTION_STATIC:-0.80}" \
         "${SEARCH_R1_TEACHER_MAX_RUNNING_REQUESTS:-128}" \
         "${SEARCH_R1_TEACHER_SCHEDULE_CONSERVATIVENESS:-0.5}" \
         "${SEARCH_R1_TEACHER_MAX_TOTAL_TOKENS:-131072}" \
         "${SEARCH_R1_TEACHER_EXTRA_ARGS:-}"
   )
   SEARCH_R1_TEACHER_PID="${SEARCH_R1_LAUNCH_INFO%%:*}"
   SEARCH_R1_TEACHER_LOG="${SEARCH_R1_LAUNCH_INFO#*:}"
   SEARCH_R1_TEACHER_URL="http://${TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/generate"
else
   SEARCH_R1_TEACHER_URL=${SEARCH_R1_TEACHER_URL:-http://127.0.0.1:13142/generate}
fi

if [ "${START_MATH_TEACHER}" = "1" ]; then
   echo "Starting Retool teacher model server..."
   until curl -sf --noproxy '*' "http://${TEACHER_IP}:${MATH_TEACHER_PORT}/health_generate" > /dev/null; do
      echo "Waiting for Retool teacher server..."
      tail -n 10 "${MATH_TEACHER_LOG}"
      sleep 5
   done
   curl --noproxy '*' "http://${TEACHER_IP}:${MATH_TEACHER_PORT}/get_model_info"
   echo "Retool teacher server is up at ${MATH_TEACHER_URL}."
fi

if [ "${START_SEARCH_R1_TEACHER}" = "1" ]; then
   echo "Starting Search-R1 teacher model server..."
   until curl -sf --noproxy '*' "http://${TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/health_generate" > /dev/null; do
      echo "Waiting for Search-R1 teacher server..."
      tail -n 10 "${SEARCH_R1_TEACHER_LOG}"
      sleep 5
   done
   curl --noproxy '*' "http://${TEACHER_IP}:${SEARCH_R1_TEACHER_PORT}/get_model_info"
   echo "Search-R1 teacher server is up at ${SEARCH_R1_TEACHER_URL}."
fi

sleep 10

# --- STUDENT preset: model config, optimizer offload, rotary base -----------
# It replaces a hardcoded
# qwen3-4B.sh, which is not the config the experts load: that file
# defaults MODEL_ARGS_ROTARY_BASE to 1000000 and the Instruct-2507 wrapper is
# what sets 5000000. The explicit --rotary-base in CKPT_ARGS happened to
# override it because CKPT_ARGS is submitted after MODEL_ARGS, so the value was
# right by argument order rather than by configuration.
STUDENT=${STUDENT:-qwen3-4b}
case "${STUDENT}" in
    qwen3-4b) _SCONFIG="qwen3-4B-Instruct-2507.sh"; _SOFFLOAD=0; _SROTARY=5000000; _KVTOK=147456; _WGIB=8 ;;
    *)
        echo "Unknown STUDENT='${STUDENT}'. Use qwen3-4b." >&2
        exit 1 ;;
esac
MODEL_CONFIG=${MODEL_CONFIG:-"${REPO_DIR}/scripts/models/${_SCONFIG}"}
# --- Anything already on the GPUs -------------------------------------------
# Same detection the other launchers do. SGLang sizes its static KV pool
# as a fraction of *total* card memory, not free memory, so whatever is resident
# at launch -- the search retriever normally lives on these same cards -- has to
# be subtracted explicitly, or the pool is sized as if the cards were empty and
# the run OOMs an hour in rather than at startup. Measured, not guessed: the
# most-used GPU right now, before anything of ours launches, plus half a GiB of
# slack. A clean node reads 0.
if [ -z "${RETRIEVER_RESERVE_GIB:-}" ]; then
   _GPU_USED_MAX_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | tail -1)
   RETRIEVER_RESERVE_GIB=$(awk -v used="${_GPU_USED_MAX_MIB:-0}" \
                               'BEGIN { printf "%d", (used < 256) ? 0 : int((used + 512) / 1024 + 0.999) }')
   echo "[gpu] most-used GPU before launch: ${_GPU_USED_MAX_MIB:-unknown} MiB -> reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (override with RETRIEVER_RESERVE_GIB)"
else
   echo "[gpu] reserving ${RETRIEVER_RESERVE_GIB} GiB/GPU (RETRIEVER_RESERVE_GIB given)"
fi

# --- SGLang static fraction, sized from the card and what is already on it ---
# 0.85 of a 40 GB card leaves only ~6 GB for CUDA graphs, prefill activations,
# NCCL buffers and fragmentation, and it fails *late*. On an 80 GB card the same
# fraction leaves ~12 GB and is fine. Detect the card, then subtract the
# resident share of the total, which is the denominator SGLang measures against.
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

source "${MODEL_CONFIG}"

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
#
# CAUTION, OPD-specific: --use-rollout-logprobs also changes which logprobs feed
# the OPD reverse KL. loss.py:686-689 picks rollout_log_probs over the recomputed
# ones, and loss.py:772 hands that same list to apply_opd_kl_to_advantages as
# student_log_probs, so the penalty becomes logp_rollout - logp_teacher. That is
# the better estimator -- the tokens really were sampled from the rollout policy
# -- but the function returns silently when the list is None (loss.py:641-642),
# which would turn OPD into a no-op with no error at all. Confirm opd_reverse_kl
# shows up in wandb on the first step; if it is missing, OPD is not running.
# In short: 1, because under ASYNC the
# rollout is one update stale and the importance ratio has real off-policyness to
# correct; the OPD KL is unaffected either way because
# slime-opd-decouple-student-logprobs takes the student side of the KL from the
# trainer's own forward, not from this tensor.
USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-1}

ASYNC=${ASYNC:-1}
# Total GPUs in this pool (whatever CUDA_VISIBLE_DEVICES exposes below).
# With Megatron teachers nothing else holds a GPU, so the student gets all 8 and
# the split is 2 train / 6 rollout -- the same shape
# the GRPO launchers run on this hardware. With served teachers, 4,5,6,7 are
# theirs and the student pool is the other four.
if [ "${OPD_TEACHER_SOURCE}" = "megatron" ]; then
   RAY_NUM_GPUS=${RAY_NUM_GPUS:-8}
else
   RAY_NUM_GPUS=${RAY_NUM_GPUS:-4}
fi
if [ "${ASYNC}" = "1" ]; then
   ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-2}
   ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-$((RAY_NUM_GPUS - ACTOR_NUM_GPUS_PER_NODE))}
   # 2 GPUs per engine. At a 16k context TP=1 both halves each engine's KV cache
   # and halves decode speed, and decode speed is what the router timeout measures.
   ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
   # The rollout GPUs are no longer shared with the trainer, so SGLang can hold a
   # much bigger KV cache -- that is what actually raises rollout concurrency.
   SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-${_MEM_ASYNC}}
   # 2 training GPUs at TP=2 means DP=1, and the distributed optimizer cannot
   # shard at DP=1: the fp32 Adam state alone is ~24 GB/GPU for a 4B model. Both
   # other OPD arms offload it here, so this one does too -- it is a step-time
   # difference as much as a memory one, and these three are compared on both.
   OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-1}
   if [ $((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS)) -ne "${RAY_NUM_GPUS}" ]; then
      echo "ACTOR_NUM_GPUS_PER_NODE(${ACTOR_NUM_GPUS_PER_NODE}) + ROLLOUT_NUM_GPUS(${ROLLOUT_NUM_GPUS})" >&2
      echo "must equal RAY_NUM_GPUS(${RAY_NUM_GPUS})." >&2
      exit 1
   fi
else
   ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-${RAY_NUM_GPUS}}
   ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-${RAY_NUM_GPUS}}
fi
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-${_SOFFLOAD}}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}

# --- Rollout shape ----------------------------------------------------------
# Hoisted out of ROLLOUT_ARGS because the concurrency derivation below reads
# ROLLOUT_MAX_CONTEXT_LEN.
# 64 prompts (32 per domain), 72 (24 per domain) with the third domain.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-$([ -n "${SQL_PROMPT_DATA}" ] && echo 72 || echo 64)}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-16384}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-16384}

# --- Rollout concurrency ----------------------------------------------------
# Derived, not hardcoded, as in the other launchers. slime fires
# rollout_batch_size x n_samples_per_prompt requests at once behind a semaphore
# of sglang_server_concurrency x num_engines; the default 512 makes that
# semaphore effectively unlimited, and an engine whose KV pool holds fewer
# sequences than that sits permanently retracting and re-prefilling until one
# transient allocation does not fit. Cap in-flight requests per engine at
#
#   pool bytes = frac x card bytes x GPUs-per-engine - bf16 weights
#   capacity   = 0.8 x pool bytes / (context length x KV bytes per token)
#
# Qwen3-4B on 80 GB at 16384 gives 32; the search half runs a shorter
# context, so this is the conservative bound of the two.
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


CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --load "${SAVE_DIR}"
   # 50. At 20 a 1000-rollout run writes 50 student
   # checkpoints at ~56 GB each (8 GB weights + 48 GB optimizer state): 2.8 TB.
   --save-interval "${SAVE_INTERVAL:-50}"
   --rotary-base "${ROTARY_BASE:-${_SROTARY}}"
)

ROLLOUT_ARGS=(
   --data-source-path mixed_data_source.MixedTaskDataSource
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   # 1000, matching the other OPD arms: high enough that the run is stopped by
   # hand, not by the loop. Stopping early is equivalent to having configured a
   # smaller number -- but CHANGING it on a resume is not free. model.py:204-207
   # derives lr_decay_steps from num_rollout x rollout_batch_size x
   # n_samples_per_prompt, and Megatron's OptimizerParamScheduler.load_state_dict
   # asserts the value matches the checkpoint's. --override-opt_param-scheduler
   # (passed below) is what makes that assert survivable; without it, editing any
   # of those three numbers makes a resume abort. Under --lr-decay-style constant
   # the derived value never reaches get_lr(), so overriding it changes no LR.
   --num-rollout "${NUM_ROLLOUT:-500}"
   # 64 x 8, the shape every other launcher uses. 256 x 1 gave the same prompt count but
   # half the samples per step, so at a matched step count the OPD arm saw half
   # the data per domain.
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   # 16384, as the reasoning expert's launcher uses. This is the
   # per-turn ceiling for the math rollout, which unlike the search one has no
   # separate cap (generate_with_retool.py:615-617 takes only min(this,
   # remaining_context)), so 8192 here halved what one ReTool turn could write
   # relative to the single-domain run. The search half is
   # unaffected either way -- SEARCH_R1_MAX_TOKENS_PER_TURN caps it at 4096.
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   # Stated rather than left to the fallback. Unset, both rollouts derive it as
   # context_parallel_size * max_tokens_per_gpu, which lands on the same 16384
   # today but silently tracks a training-side packing knob.
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   # no --rollout-stop: the ReTool rollout puts </tool_call> in the stop list itself
   # (generate_with_retool.py). The launcher flag never reached SGLang intact.
)
if [ -n "${REPLAY_LOGS}" ]; then
   # The replayed batches are the dumps; writing them out again would copy the directory.
   ROLLOUT_ARGS+=(--load-debug-rollout-data "${REPLAY_LOGS}/rollout_{rollout_id}.pt")
else
   ROLLOUT_ARGS+=(--save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt")
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16400}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-kl-coef "${OPD_KL_COEF:-$([ -n "${TEACHER_DENSE_KL}" ] && echo 0 || echo 1.0)}"
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# --use-kl-loss is what makes slime load and forward a reference model
# (placement_group.py:178: with_ref = kl_coef != 0 or use_kl_loss). At
# KL_LOSS_COEF=0.0 that forward contributes exactly nothing to the loss while
# costing a full extra forward pass per step and an extra 4B weight set in
# memory -- and neither RL arm pays it, so it is also a compute asymmetry.
# Only ask for the ref model when the coefficient actually uses it.
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
if [ "${KL_LOSS_COEF}" != "0.0" ] && [ "${KL_LOSS_COEF}" != "0" ]; then
   GRPO_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF}")
fi

if [ "${OPD_TEACHER_SOURCE}" = "megatron" ]; then
   # One teacher per sample source, routed by metadata["source_name"], which
   # generate_multi_domain sets to the task.
   #
   # source=path pairs rather than JSON. Ray joins the entrypoint into a single
   # string, and a JSON value reached argparse with its outer single quotes
   # still attached -- json.loads then failed at character 0 with a message
   # about neither quoting nor this flag. These pairs contain no quotes, braces
   # or spaces, so there is nothing for that join to get wrong. slime still
   # accepts JSON when driven directly.
   GRPO_ARGS+=(
      --opd-type megatron
      --opd-teacher-loads "retool=${MATH_TEACHER_LOAD},search-r1=${SEARCH_R1_TEACHER_LOAD}${SQL_PROMPT_DATA:+,sql=${SQL_TEACHER_LOAD}}"
      --opd-teacher-ckpt-steps "retool=${MATH_TEACHER_CKPT_STEP},search-r1=${SEARCH_R1_TEACHER_CKPT_STEP}${SQL_PROMPT_DATA:+,sql=${SQL_TEACHER_CKPT_STEP}}"
   )
else
   GRPO_ARGS+=(--opd-type sglang)
fi
if [ -n "${TEACHER_DENSE_KL}" ]; then
   # The dense term in the loss (slime-opd-dense-kl.patch); slime's sampled-token term is at 0 above.
   GRPO_ARGS+=(--opd-dense-kl "${TEACHER_DENSE_KL}" --opd-dense-kl-coef "${TEACHER_DENSE_KL_COEF}" --opd-teacher-topk "${TEACHER_TOPK}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
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
   --weight-decay "${WEIGHT_DECAY:-0.1}"
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
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.5}"
   # Bound the prefill peak: this
   # arm runs the 16k ReTool context. Left unset, SGLang sizes the chunk
   # generously and a late-trajectory prefill needs a logits buffer of
   # chunk x vocab_shard x 2 bytes -- ~1.6 GB for an 11k chunk at TP=2 on
   # Qwen3-4B. 4096 caps it near 0.6 GB for a little prefill latency.
   --sglang-chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}"
   --sglang-cuda-graph-backend-prefill "${CUDA_GRAPH_BACKEND_PREFILL:-disabled}"
   # Per-engine cap on in-flight requests; see the Rollout concurrency block.
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   # Bound how long a single /generate may hang. slime builds its HTTP client with
   # httpx.Timeout(None) and only retries on an exception, so without this a lost
   # response stalls the whole rollout (the router default is 14400s = 4 hours).
   # This MUST stay below GENERATE_SAMPLE_TIMEOUT, otherwise the sample gives up
   # first and _post never sees the exception that triggers its retry.
   --sglang-router-request-timeout-secs "${ROUTER_REQUEST_TIMEOUT_SECS:-600}"
)

MISC_ARGS=(
   --log-multi-turn
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_multi_domain.generate
   --custom-rm-path generate_multi_domain.reward_func
   --custom-reward-post-process-path generate_multi_domain.post_process_rewards
)
# train/teacher_kl-<domain>: the per-token teacher KL by domain (slime's own train/opd_reverse_kl
# pools the domains).
CUSTOM_ARGS+=(--rollout-data-postprocess-path teacher_kl_metric.log_teacher_kl)

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-agentic-opd}
# grpo-opd-both when the task reward is added on top of the teacher term (GRPO+OPD).
# -t<step> names the teacher budget when it is not the 499 every run so far distilled
# from (-t<math>-<search>[-<sql>] when the teachers' steps differ). Set TEACHER_TAG to
# override the whole suffix, or to "" to leave it out. opd-both / grpo-opd-both for math + search,
# opd-3d / grpo-opd-3d with the SQL domain.
_teacher_tag() {
   local m="${MATH_TEACHER_CKPT_STEP:-}" s="${SEARCH_R1_TEACHER_CKPT_STEP:-}" q="${SQL_PROMPT_DATA:+${SQL_TEACHER_CKPT_STEP:-}}"
   if [ -n "${TEACHER_TAG+x}" ]; then printf '%s' "${TEACHER_TAG:+-${TEACHER_TAG}}"; return; fi
   [ -n "${m}" ] || return 0
   local steps="${m} ${s}${q:+ ${q}}" step all499=1 same=1
   for step in ${steps}; do
      [ "${step}" = "499" ] || all499=0
      [ "${step}" = "${m}" ] || same=0
   done
   if [ "${all499}" = "1" ]; then return 0; fi
   if [ "${same}" = "1" ]; then printf -- '-t%s' "${m}"; else printf -- '-t%s' "${steps// /-}"; fi
}
WANDB_GROUP=${WANDB_GROUP:-$([ "${MULTI_DOMAIN_USE_TASK_REWARD:-0}" = "1" ] && echo grpo-opd || echo opd)-${_DOMAIN}${_REPLAY_TAG}${_DKL_TAG}${_B_TAG}$(_teacher_tag)-${STUDENT}}
DISABLE_WANDB_RANDOM_SUFFIX=${DISABLE_WANDB_RANDOM_SUFFIX:-1}
export WANDB_DIR=${WANDB_DIR:-${SAVE_DIR}}

WANDB_ARGS=()
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

if [ "${OPD_TEACHER_SOURCE}" = "megatron" ]; then
   export CUDA_VISIBLE_DEVICES=${STUDENT_GPUS:-0,1,2,3,4,5,6,7}
else
   export CUDA_VISIBLE_DEVICES=${STUDENT_GPUS:-0,1,2,3}
fi
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export TOOL_SANDBOX_BACKEND=${TOOL_SANDBOX_BACKEND:-subprocess}
# 128:
# this is the cap on concurrent Python sandbox subprocesses. It is a throughput
# knob, not a numerical one, but a deeper queue is one more reason two otherwise
# identical math rollouts could time out differently.
export TOOL_SANDBOX_CONCURRENCY=${TOOL_SANDBOX_CONCURRENCY:-128}
export TOOL_SANDBOX_MAX_TURNS=${TOOL_SANDBOX_MAX_TURNS:-16}
export TOOL_SANDBOX_MAX_TOOL_CALLS=${TOOL_SANDBOX_MAX_TOOL_CALLS:-16}
export TOOL_SANDBOX_JUPYTER_TIMEOUT=${TOOL_SANDBOX_JUPYTER_TIMEOUT:-600}
# 1200, as every other arm. Unset, generate_with_retool.py defaults it to 600 --
# equal to --sglang-router-request-timeout-secs, and the sample then gives up at
# the same moment the router does, before _post ever sees the exception that
# triggers its retry. Must stay ABOVE the router timeout.
export GENERATE_SAMPLE_TIMEOUT=${GENERATE_SAMPLE_TIMEOUT:-1200}
# One server, as start_retriever.sh and every other script now default to. This
# arm never starts its own, so SEARCH_URL is the whole interface to it.
export SEARCH_URL=${SEARCH_URL:-"http://127.0.0.1:8000/retrieve"}
# A retriever that is not running does not fail anything: generate_with_search
# catches the connection error and hands the model "[ERROR] ..." as if it were a
# result, so the run trains to completion on transcripts that retrieved nothing.
if [ -n "${REPLAY_LOGS}" ]; then
   echo "[opd] replay: no rollout, the retriever is not asked"
else
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
   echo "[retriever] nothing in SEARCH_URL answered. This arm never starts its own;" >&2
   echo "[retriever] start one with  bash scripts/eval/start_retriever.sh" >&2
   exit 1
fi
fi
export SEARCH_R1_MAX_TURNS=${SEARCH_R1_MAX_TURNS:-8}
export SEARCH_R1_MAX_TOKENS_PER_TURN=${SEARCH_R1_MAX_TOKENS_PER_TURN:-4096}
# Per-task context budgets, equal to each task's single-domain setting.
# slime has one --rollout-max-context-len, so without these both
# tasks run at the global 16384 -- which would put search on a window 1.8x wider
# than the one its teacher was trained in. The two would then not be measuring the same task.
# Applied with min() against the global, so they can only narrow.
export RETOOL_MAX_CONTEXT_LEN=${RETOOL_MAX_CONTEXT_LEN:-16384}
export SEARCH_R1_MAX_CONTEXT_LEN=${SEARCH_R1_MAX_CONTEXT_LEN:-9216}
export SQL_DB_ROOT SQL_MAX_CONTEXT_LEN SQL_MAX_TOOL_CALLS SQL_MAX_TURNS SQL_MAX_TOKENS_PER_TURN \
       SQL_EXEC_TIMEOUT SQL_SCORE_TIMEOUT SQL_MAX_RESULT_ROWS SQL_MAX_RESULT_CHARS SQL_TOOL_CONCURRENCY

# Set to 0 for pure OPD with task rewards logged only.
# Set to 1 to add the task reward on top of OPD; 0 = pure OPD (reward logged only).
export MULTI_DOMAIN_USE_TASK_REWARD=${MULTI_DOMAIN_USE_TASK_REWARD:-0}

# generate_multi_domain reads this to decide whether to fetch teacher log-probs
# over HTTP during the rollout ("sglang") or leave teacher_log_probs to the
# trainer's forward ("megatron"). Exported and passed through RUNTIME_ENV_JSON
# rather than left to whatever the raylet inherited, because a wrong value here
# is the difference between the teacher being consulted and not.
export OPD_TEACHER_SOURCE

# Per-teacher in-flight caps. Both teachers run at once, so a single shared cap
# lets heavy retool prefills (long math+code traces) flood the retool teacher and
# overrun its KV cache, queueing requests past the sock-read timeout. Bound each.
export OPD_TEACHER_MATH_MAX_INFLIGHT=${OPD_TEACHER_MATH_MAX_INFLIGHT:-64}
export OPD_TEACHER_SEARCH_R1_MAX_INFLIGHT=${OPD_TEACHER_SEARCH_R1_MAX_INFLIGHT:-64}


ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
ulimit -u 65536 2>/dev/null || true
echo "[opd] ulimit -n=$(ulimit -n)  ulimit -u=$(ulimit -u)"

# RAY_NUM_GPUS and CUDA_VISIBLE_DEVICES are set from the same OPD_TEACHER_SOURCE
# branch above, so the shipped pair agrees. Override one without the other --
# e.g. giving the student all 8 while served teachers still hold 4-7 -- and
# telling Ray about GPUs it cannot see fails later, deep in placement, with a
# message that names neither variable. The ASYNC split above also sizes its
# pools from RAY_NUM_GPUS, so check the pair here.
_VISIBLE=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
if [ "${_VISIBLE}" -ne "${RAY_NUM_GPUS}" ]; then
   echo "RAY_NUM_GPUS=${RAY_NUM_GPUS} but CUDA_VISIBLE_DEVICES exposes ${_VISIBLE} GPU(s)" >&2
   echo "(STUDENT_GPUS=${CUDA_VISIBLE_DEVICES}). Set STUDENT_GPUS to ${RAY_NUM_GPUS} devices," >&2
   echo "or set RAY_NUM_GPUS=${_VISIBLE}." >&2
   exit 1
fi

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# PYTORCH_CUDA_ALLOC_CONF set on the command line reaches the training workers through the
# runtime env (otherwise they inherit the raylet's environment, not this shell's).
# expandable_segments:True is the value against the allocator fragmentation the dense arm's long
# micro-batches ran into (33 GB reserved-but-unallocated at the OOM, 2026-11-01); slime's
# own SFT scripts ship it. Not set by default: unverified next to a co-located SGLang engine, and
# the sampled-token arms ran without it.
_ALLOC_CONF_ENV=
[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] || _ALLOC_CONF_ENV="\"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\","
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SRC_DIR}:${RELEASE_DIR}:${REPO_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"HF_HUB_OFFLINE\": \"1\",
    \"TRANSFORMERS_OFFLINE\": \"1\",
    \"TOOL_SANDBOX_BACKEND\": \"${TOOL_SANDBOX_BACKEND}\",
    \"TOOL_SANDBOX_CONCURRENCY\": \"${TOOL_SANDBOX_CONCURRENCY}\",
    \"TOOL_SANDBOX_MAX_TURNS\": \"${TOOL_SANDBOX_MAX_TURNS}\",
    \"TOOL_SANDBOX_MAX_TOOL_CALLS\": \"${TOOL_SANDBOX_MAX_TOOL_CALLS}\",
    \"TOOL_SANDBOX_JUPYTER_TIMEOUT\": \"${TOOL_SANDBOX_JUPYTER_TIMEOUT}\",
    \"SEARCH_URL\": \"${SEARCH_URL}\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_R1_MAX_TURNS}\",
    \"SEARCH_R1_MAX_TOKENS_PER_TURN\": \"${SEARCH_R1_MAX_TOKENS_PER_TURN}\",
    \"RETOOL_MAX_CONTEXT_LEN\": \"${RETOOL_MAX_CONTEXT_LEN}\",
    \"SEARCH_R1_MAX_CONTEXT_LEN\": \"${SEARCH_R1_MAX_CONTEXT_LEN}\",
    \"MATH_TEACHER_URL\": \"${MATH_TEACHER_URL}\",
    \"SEARCH_R1_TEACHER_URL\": \"${SEARCH_R1_TEACHER_URL}\",
    \"RETOOL_PROMPT_DATA\": \"${RETOOL_PROMPT_DATA}\",
    \"RETOOL_INPUT_KEY\": \"${RETOOL_INPUT_KEY}\",
    \"RETOOL_LABEL_KEY\": \"${RETOOL_LABEL_KEY}\",
    \"RETOOL_METADATA_KEY\": \"${RETOOL_METADATA_KEY}\",
    \"SEARCH_R1_PROMPT_DATA\": \"${SEARCH_R1_PROMPT_DATA}\",
    \"SEARCH_R1_INPUT_KEY\": \"${SEARCH_R1_INPUT_KEY}\",
    \"SEARCH_R1_LABEL_KEY\": \"${SEARCH_R1_LABEL_KEY}\",
    \"SEARCH_R1_METADATA_KEY\": \"${SEARCH_R1_METADATA_KEY}\",
    \"SQL_PROMPT_DATA\": \"${SQL_PROMPT_DATA}\",
    \"SQL_INPUT_KEY\": \"${SQL_INPUT_KEY}\",
    \"SQL_LABEL_KEY\": \"${SQL_LABEL_KEY}\",
    \"SQL_METADATA_KEY\": \"${SQL_METADATA_KEY}\",
    \"SQL_DB_ROOT\": \"${SQL_DB_ROOT}\",
    \"SQL_MAX_CONTEXT_LEN\": \"${SQL_MAX_CONTEXT_LEN}\",
    \"SQL_MAX_TOOL_CALLS\": \"${SQL_MAX_TOOL_CALLS}\",
    \"SQL_MAX_TURNS\": \"${SQL_MAX_TURNS}\",
    \"SQL_MAX_TOKENS_PER_TURN\": \"${SQL_MAX_TOKENS_PER_TURN}\",
    \"SQL_EXEC_TIMEOUT\": \"${SQL_EXEC_TIMEOUT}\",
    \"SQL_SCORE_TIMEOUT\": \"${SQL_SCORE_TIMEOUT}\",
    \"SQL_MAX_RESULT_ROWS\": \"${SQL_MAX_RESULT_ROWS}\",
    \"SQL_MAX_RESULT_CHARS\": \"${SQL_MAX_RESULT_CHARS}\",
    \"SQL_TOOL_CONCURRENCY\": \"${SQL_TOOL_CONCURRENCY}\",
    \"MIXED_TASK_ORDER\": \"${MIXED_TASK_ORDER}\",
    \"MULTI_DOMAIN_USE_TASK_REWARD\": \"${MULTI_DOMAIN_USE_TASK_REWARD}\",
    \"OPD_TEACHER_MATH_MAX_INFLIGHT\": \"${OPD_TEACHER_MATH_MAX_INFLIGHT}\",
    \"OPD_TEACHER_SEARCH_R1_MAX_INFLIGHT\": \"${OPD_TEACHER_SEARCH_R1_MAX_INFLIGHT}\",
    \"OPD_TEACHER_SOURCE\": \"${OPD_TEACHER_SOURCE}\",
    \"GENERATE_SAMPLE_TIMEOUT\": \"${GENERATE_SAMPLE_TIMEOUT}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"WANDB_DIR\": \"${WANDB_DIR}\",
    \"USER\": \"${USER}\",
    \"LOGNAME\": \"${LOGNAME}\",
    \"HOME\": \"${HOME}\",
    ${_ALLOC_CONF_ENV}
    \"no_proxy\": \"${no_proxy:-}\",
    \"NO_PROXY\": \"${NO_PROXY:-}\"
  }
}"

if [ -n "${REPLAY_LOGS}" ]; then
   # Sync, nothing to overlap; the ratio to the dump's sampling log-probs; no colocation
   # (--load-debug-rollout-data makes the run train-only, the placement group the
   # training pool alone).
   ENTRYPOINT=train.py
   PARALLEL_ARGS=(--use-rollout-logprobs)
elif [ "${ASYNC}" = "1" ]; then
   ENTRYPOINT=train_async.py
   PARALLEL_ARGS=()
   if [ "${USE_ROLLOUT_LOGPROBS}" = "1" ]; then
      PARALLEL_ARGS+=(--use-rollout-logprobs)
   fi
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
