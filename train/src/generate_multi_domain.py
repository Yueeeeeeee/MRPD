"""Cross-domain rollout: one generate/reward/post-process triple for every arm.

Dispatches each sample to the ReTool, Search-R1 or text-to-SQL rollout by
metadata.task, and scores it with that task's reward function. GRPO and OPD both
use this module for a mixed run; the only OPD-specific part is the teacher
log-probs, which are gated on args.use_opd rather than on an environment
variable so that a run without teachers cannot accidentally inherit them.

Environment variables split by subsystem, so the name says which one owns it:
MULTI_DOMAIN_* configures the cross-domain rollout, OPD_TEACHER_* the teacher
client. They were renamed from a MULTITEACHER_OPD_* prefix that predates
GRPO using this module; _reject_legacy_env() below turns a stale one into an
error rather than a silent change of behaviour.
"""
from __future__ import annotations

import asyncio

# Keep the stdlib loop policy used by the existing OPD helpers. It is more
# stable with many concurrent subprocess/tool and aiohttp calls.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import json
import os
import sys
from typing import Any

import aiohttp
import torch

import generate_with_retool
import generate_with_search
import generate_with_sql
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


TASK_ALIASES = {
    "retool": "retool",
    "search": "search-r1",
    "search-r1": "search-r1",
    "search_r1": "search-r1",
    "sql": "sql",
}

# Every task this module can route, and the prefix of its rollout/* metrics.
TASKS = ("retool", "search-r1", "sql")
METRIC_PREFIX = {"retool": "retool", "search-r1": "search_r1", "sql": "sql"}

TEACHER_MODEL_ENV = {
    "retool": "MATH_TEACHER_MODEL_NAME",
    "search-r1": "SEARCH_R1_TEACHER_MODEL_NAME",
    "sql": "SQL_TEACHER_MODEL_NAME",
}

TEACHER_URL_ENV = {
    "retool": "MATH_TEACHER_URL",
    "search-r1": "SEARCH_R1_TEACHER_URL",
    "sql": "SQL_TEACHER_URL",
}

DEFAULT_TEACHER_MODEL = {
    "retool": "retool_teacher",
    "search-r1": "search_r1_teacher",
    "sql": "sql_teacher",
}

_session_lock = asyncio.Lock()
_session_by_loop: dict[int, aiohttp.ClientSession] = {}
# Per-(loop, task) in-flight semaphore. The two teachers have very different
# capacities (e.g. retool on 3-4 GPUs, search-r1 on 1 GPU). A single shared
# semaphore lets the slow single-GPU teacher hold slots that the retool teacher
# needs, and lets up to MAX_INFLIGHT heavy retool prefills hit a 3-GPU teacher
# at once -> KV overrun -> server queueing -> sock_read timeout -> retries.
# Bounding each teacher independently keeps each server's queue shallow.
_semaphore_by_loop: dict[tuple[int, str], asyncio.Semaphore] = {}
def _reject_legacy_env() -> None:
    """Fail loudly on the pre-rename variable names.

    Renaming an environment variable that gates behaviour is a silent failure by
    default: a stale MULTITEACHER_OPD_USE_TASK_REWARD=0 left in a shell or a
    launcher would simply stop being read, and the OPD arm would quietly train
    with the task reward that the zeroing exists to remove. Refuse to start.
    """
    stale = sorted(k for k in os.environ if k.startswith("MULTITEACHER_OPD_"))
    if stale:
        raise RuntimeError(
            "These environment variables were renamed and are no longer read: "
            + ", ".join(stale)
            + ".\n  MULTITEACHER_OPD_USE_TASK_REWARD -> MULTI_DOMAIN_USE_TASK_REWARD"
            + "\n  every other MULTITEACHER_OPD_* -> OPD_TEACHER_*"
            + " (e.g. MULTITEACHER_OPD_RETOOL_MAX_INFLIGHT ->"
            + " OPD_TEACHER_MATH_MAX_INFLIGHT).\nUnset the old names."
        )

    # The math teacher's variables say MATH_, not RETOOL_. "retool" is still the
    # task identifier -- it is what the datasets carry in metadata, what
    # MIXED_TASK_ORDER spells and what keys --opd-teacher-loads -- but the
    # variables naming that task's *teacher* read MATH_TEACHER_*.
    stale = sorted(
        k for k in os.environ if k.startswith(("RETOOL_TEACHER_", "START_RETOOL_TEACHER"))
    ) + sorted(k for k in os.environ if k == "OPD_TEACHER_RETOOL_MAX_INFLIGHT")
    if stale:
        raise RuntimeError(
            "These environment variables were renamed and are no longer read: "
            + ", ".join(stale)
            + ".\n  RETOOL_TEACHER_*             -> MATH_TEACHER_*"
            + "\n  START_RETOOL_TEACHER          -> START_MATH_TEACHER"
            + "\n  OPD_TEACHER_RETOOL_MAX_INFLIGHT -> OPD_TEACHER_MATH_MAX_INFLIGHT"
            + "\nThe task itself is still called 'retool'; only the math teacher's"
            + " variables were renamed.\nUnset the old names."
        )


_reject_legacy_env()

_TEACHER_MAX_INFLIGHT = int(os.environ.get("OPD_TEACHER_MAX_INFLIGHT", "64"))
_TEACHER_MAX_INFLIGHT_BY_TASK = {
    "retool": int(
        os.environ.get("OPD_TEACHER_MATH_MAX_INFLIGHT", str(_TEACHER_MAX_INFLIGHT))
    ),
    "search-r1": int(
        os.environ.get("OPD_TEACHER_SEARCH_R1_MAX_INFLIGHT", str(_TEACHER_MAX_INFLIGHT))
    ),
    "sql": int(os.environ.get("OPD_TEACHER_SQL_MAX_INFLIGHT", str(_TEACHER_MAX_INFLIGHT))),
}
_TEACHER_REQUEST_TIMEOUT = int(os.environ.get("OPD_TEACHER_REQUEST_TIMEOUT", "600"))
_TEACHER_CONNECT_TIMEOUT = int(os.environ.get("OPD_TEACHER_CONNECT_TIMEOUT", "60"))
_TEACHER_SOCK_READ_TIMEOUT = int(os.environ.get("OPD_TEACHER_SOCK_READ_TIMEOUT", "300"))
_TEACHER_TOTAL_BUDGET = int(os.environ.get("OPD_TEACHER_TOTAL_BUDGET", "900"))
# 4 rather than 2. The request is a pure scoring call (max_new_tokens=0) with
# no server-side state, so retrying it is free of side effects, and the
# connection errors it has to survive are exactly the ones a retry fixes.
_TEACHER_MAX_RETRIES = int(os.environ.get("OPD_TEACHER_MAX_RETRIES", "4"))
# Per-task running values, carried over a step in which a task happens to be absent.
_LAST_TASK_REWARD_MEAN = {task: 0.0 for task in TASKS}
_LAST_TASK_TRAIN_REWARD_MEAN = {task: 0.0 for task in TASKS}
_LAST_TASK_SCORE_MEAN = {task: 0.0 for task in TASKS}
_LAST_TASK_TOOL_CALL_COUNT_MEAN = {task: 0.0 for task in TASKS}
_LAST_TASK_RESPONSE_LEN_MEAN = {task: 0.0 for task in TASKS}
_LAST_TASK_RAW_RESPONSE_LEN_MEAN = {task: 0.0 for task in TASKS}
_LAST_TASK_TRUNCATED_RATIO = {task: 0.0 for task in TASKS}
# The tasks this run has produced so far: only their metrics are logged, so a
# two-domain run does not report a third domain's zeros.
_SEEN_TASKS: set[str] = set()


def _metadata(sample: Sample) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _get_task(sample: Sample) -> str:
    metadata = _metadata(sample)
    task = metadata.get("task") or metadata.get("source") or metadata.get("dataset")
    task = TASK_ALIASES.get(str(task).strip().lower() if task is not None else "")
    if task is None:
        raise ValueError(
            "Each sample must set metadata.task to 'retool', 'search-r1' or 'sql'. "
            "The launchers set it through mixed_data_source.MixedTaskDataSource; a pre-mixed prompt file needs the same metadata column."
        )
    return task


def _use_task_reward() -> bool:
    return os.environ.get("MULTI_DOMAIN_USE_TASK_REWARD", "1").lower() not in {"0", "false", "no"}


def _replay_loss() -> str:
    """REPLAY_LOSS on grpo_multi.sh's replay arm: "" (the GRPO loss), "sft" (slime's SFT loss on every
    logged sample) or "rft" (the SFT loss on the correct samples only: rejection-sampling fine-tuning,
    Yuan et al. 2023, arXiv:2308.01825). The loss itself is the launcher's --loss-type; this module
    only takes the rejected samples out of the loss (Sample.remove_sample: slime zeroes their loss
    mask, so they count in neither the numerator nor the denominator of the per-sample mean)."""
    value = os.environ.get("REPLAY_LOSS", "").strip().lower()
    if value not in {"", "sft", "rft"}:
        raise ValueError(f"REPLAY_LOSS must be unset, sft or rft, got {value!r}")
    return value


def _raft() -> bool:
    """RAFT=1 on grpo_multi.sh's online arm: RAFT (Dong et al. 2023, arXiv:2304.06767) with the binary
    task reward -- the student samples its own rollouts in the environments, as online GRPO does, and
    slime's SFT loss trains on the correct ones only, every correct sample of a prompt as in Xiong et
    al. 2025 (arXiv:2504.11343), at most RFT_MAX_PER_PROMPT of them when set (1: RAFT's best-of-K).
    The online counterpart of REPLAY_LOSS=rft: the same selection, on the student's samples instead of
    the experts' logs."""
    value = os.environ.get("RAFT", "").strip()
    if value not in {"", "0", "1"}:
        raise ValueError(f"RAFT must be unset, 0 or 1, got {value!r}")
    return value == "1"


def _rft_keep(raw_rewards: list[float], n_samples_per_prompt: int, max_per_prompt: int | None = None) -> list[bool]:
    """Which samples rejection-sampling fine-tuning keeps: the ones with a positive raw reward, at most
    max_per_prompt of them per prompt (the first in sampling order; a prompt's samples are consecutive,
    as _normalize_rewards assumes). None keeps every correct sample (Yuan et al. 2023)."""
    assert max_per_prompt is None or max_per_prompt >= 1, max_per_prompt
    keep = []
    for start in range(0, len(raw_rewards), n_samples_per_prompt):
        kept = 0
        for reward in raw_rewards[start : start + n_samples_per_prompt]:
            ok = reward > 0 and (max_per_prompt is None or kept < max_per_prompt)
            if ok:
                kept += 1
            keep.append(ok)
    return keep


def _teacher_enabled(args) -> bool:
    """Whether this run needs teacher log-probs at all.

    Gated on args.use_opd (slime sets it from --use-opd) rather than on an
    environment variable. GRPO shares this module but has no teacher:
    without the gate they would POST to a teacher URL that nothing is serving,
    burn the retry budget once per sample, and then have post_process_rewards
    write a tensor of zeros into rollout_data["teacher_log_probs"] -- unused by
    those estimators, so entirely silent.
    """
    if not getattr(args, "use_opd", False):
        return False
    return not _teacher_from_megatron()


def _teacher_from_megatron() -> bool:
    """Whether teacher log-probs come from slime's Megatron forward instead of HTTP.

    With --opd-type megatron the teacher is a weight set inside the trainer and its
    log-probs are produced by an extra forward pass (actor.py:441-451), so there is
    no teacher server to call. Everything else this module does -- the ReTool /
    Search-R1 multi-turn rollout and, crucially, the task-reward zeroing that makes
    this arm *pure* OPD -- is still needed, which is why the flag lives here rather
    than in a different generate function.
    """
    return os.environ.get("OPD_TEACHER_SOURCE", "sglang").strip().lower() == "megatron"


def _get_teacher_url(args, task: str) -> str:
    url = os.environ.get(TEACHER_URL_ENV[task])
    if url:
        return url

    url_map = os.environ.get("OPD_TEACHER_URLS")
    if url_map:
        urls = json.loads(url_map)
        if task in urls:
            return urls[task]

    model_name = os.environ.get(TEACHER_MODEL_ENV[task], DEFAULT_TEACHER_MODEL[task])
    routers = getattr(args, "sglang_model_routers", None) or {}
    if model_name in routers:
        return get_model_url(args, model_name, "/generate")

    raise ValueError(
        f"No teacher route configured for task '{task}'. Set {TEACHER_URL_ENV[task]} to an external "
        f"SGLang /generate URL, set OPD_TEACHER_URLS, or define model '{model_name}' "
        "in --sglang-config with update_weights: false."
    )


def _get_semaphore(task: str) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = (id(loop), task)
    sem = _semaphore_by_loop.get(key)
    if sem is None:
        limit = _TEACHER_MAX_INFLIGHT_BY_TASK.get(task, _TEACHER_MAX_INFLIGHT)
        sem = asyncio.Semaphore(limit)
        _semaphore_by_loop[key] = sem
    return sem


async def _get_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sess = _session_by_loop.get(key)
    if sess is not None and not sess.closed:
        return sess

    async with _session_lock:
        sess = _session_by_loop.get(key)
        if sess is not None and not sess.closed:
            return sess

        timeout = aiohttp.ClientTimeout(
            total=_TEACHER_REQUEST_TIMEOUT,
            connect=_TEACHER_CONNECT_TIMEOUT,
            sock_connect=_TEACHER_CONNECT_TIMEOUT,
            sock_read=_TEACHER_SOCK_READ_TIMEOUT,
        )
        connector = aiohttp.TCPConnector(
            limit=128,
            limit_per_host=128,
            ttl_dns_cache=300,
            # force_close=True, i.e. NO connection reuse. This was False, and the
            # first run with served teachers failed teacher calls with
            # ServerDisconnectedError('Server disconnected') -- sporadically, on
            # both teachers, on requests of 1.5k-6.7k tokens, i.e. nowhere near
            # any capacity limit. That is the classic pooled-keepalive race:
            # sglang's uvicorn closes an idle connection after ~5s, aiohttp does
            # not find out, hands the dead socket to the next request and the
            # send fails. Under a 64-slot semaphore most connections sit idle
            # well past 5s between uses, so the pool is mostly stale sockets.
            # A fresh connection per request costs a local TCP handshake; a lost
            # teacher answer costs a sample. See _neutral_teacher_log_probs for
            # what a lost answer used to do to that sample's advantage.
            force_close=True,
            enable_cleanup_closed=True,
        )
        sess = aiohttp.ClientSession(trust_env=False, timeout=timeout, connector=connector)
        _session_by_loop[key] = sess
        return sess


async def generate(args, sample: Sample, sampling_params) -> Sample:
    task = _get_task(sample)
    if task == "retool":
        return await generate_with_retool.generate(args, sample, sampling_params)
    if task == "search-r1":
        return await generate_with_search.generate(args, sample, sampling_params)
    if task == "sql":
        return await generate_with_sql.generate(args, sample, sampling_params)
    raise AssertionError(f"Unhandled task: {task}")


async def _teacher_logprob(args, sample: Sample, task: str) -> dict[str, Any] | None:
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]

    last_err: Exception | None = None
    sem = _get_semaphore(task)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TEACHER_TOTAL_BUDGET
    url = _get_teacher_url(args, task)

    for attempt in range(_TEACHER_MAX_RETRIES):
        if deadline - loop.time() <= 0:
            break
        try:
            session = await _get_session()
            async with sem:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            await asyncio.sleep(min(2**attempt, 8))
        except RuntimeError as e:
            msg = str(e)
            if "File descriptor" not in msg and "transport" not in msg:
                raise
            last_err = e
            await asyncio.sleep(0.05 * (attempt + 1))

    print(
        f"[multi-domain] WARN: teacher /generate failed for task={task} "
        f"after {_TEACHER_MAX_RETRIES} retries (url={url}, tokens={len(sample.tokens)}): {last_err!r}",
        file=sys.stderr,
        flush=True,
    )
    return None


async def _task_reward(args, sample: Sample, task: str) -> float:
    if task == "retool":
        solution_str = generate_with_retool._prompt_to_text(sample.prompt) + sample.response
        ground_truth = sample.label if sample.label is not None else ""
        reward = generate_with_retool.math_dapo_compute_score(
            solution_str,
            ground_truth,
            strict_box_verify=True,
        )
        task_score = float(reward["score"])
        num_turns = int(getattr(sample, "tool_call_count", 0) or 0)
        shaped_reward = task_score
        # Gate on the same switch generate_with_retool.reward_func uses, which is
        # OFF by default. Applying the bonus here unconditionally made the reward
        # this arm reports mean something different from what the GRPO
        # arm reports: they log a plain +1/-1 outcome, this logged a version with
        # failed trajectories lifted, so rollout/raw_reward was not comparable
        # across the three wandb runs. OPD zeroes the reward before the estimator
        # sees it, so this only ever affected logging -- which is exactly why it
        # could sit here unnoticed.
        if generate_with_retool._tool_call_shaping() and shaped_reward < 0:
            tool_call_reward = (num_turns - 2) / 2 * 0.1
            shaped_reward = min(-0.6, shaped_reward + tool_call_reward)
        sample._opd_task_score_unshaped = task_score  # type: ignore[attr-defined]
        return float(shaped_reward)
    elif task == "search-r1":
        reward = await generate_with_search.reward_func(args, sample)
    elif task == "sql":
        reward = await generate_with_sql.reward_func(args, sample)  # {"score": +-1, "pred", "error"}
    else:
        raise AssertionError(f"Unhandled task: {task}")

    if isinstance(reward, dict):
        if "score" in reward:
            scalar_reward = float(reward["score"])
            sample._opd_task_score_unshaped = scalar_reward  # type: ignore[attr-defined]
            return scalar_reward
        if "reward" in reward:
            scalar_reward = float(reward["reward"])
            sample._opd_task_score_unshaped = scalar_reward  # type: ignore[attr-defined]
            return scalar_reward
        raise ValueError(f"Cannot extract scalar reward from dict keys: {sorted(reward.keys())}")
    scalar_reward = float(reward)
    sample._opd_task_score_unshaped = scalar_reward  # type: ignore[attr-defined]
    return scalar_reward


async def _reward_one(args, sample: Sample) -> float:
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    task = _get_task(sample)
    # Keep teacher scoring and task scoring in one coroutine. This mirrors the
    # Retool-only OPD path more closely than spawning a per-sample gather.
    teacher_response = await _teacher_logprob(args, sample, task) if _teacher_enabled(args) else None
    task_reward = await _task_reward(args, sample, task)

    sample._opd_task = task  # type: ignore[attr-defined]
    sample._opd_task_score = task_reward  # type: ignore[attr-defined]
    sample._opd_teacher_response = teacher_response  # type: ignore[attr-defined]
    return task_reward


async def reward_func(args, sample, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(*[_reward_one(args, item) for item in sample])
    return await _reward_one(args, sample)


def _neutral_teacher_log_probs(sample: Sample) -> torch.Tensor:
    """Teacher log-probs that make the OPD term vanish for one sample.

    Used when the teacher gave no usable answer. Zeros are the wrong fallback:
    loss.py:653 forms reverse_kl = student_logp - teacher_logp, so a zero
    teacher turns it into student_logp, which is negative at every token, and
    the advantage -opd_kl_coef * reverse_kl comes out positive everywhere. That
    sample would then be trained to raise the probability of every token it
    happened to emit -- REINFORCE with reward +|logp|, and no error anywhere.

    Under --opd-type sglang without the decouple patch, the student side of the
    KL is the rollout engine's log-probs (loss.py:687-689 -> :772), so copying
    those makes reverse_kl exactly 0 and, with the task reward already zeroed,
    the whole advantage 0: the sample drops out of the gradient instead of
    corrupting it. Falls back to zeros only when no rollout log-probs exist
    either, and that case is counted and warned about by the caller.
    """
    response_length = sample.response_length
    rollout_lp = getattr(sample, "rollout_log_probs", None)
    if rollout_lp is not None and len(rollout_lp) == response_length:
        return torch.tensor(rollout_lp, dtype=torch.float32)
    return torch.zeros(response_length, dtype=torch.float32)


def _extract_teacher_log_probs(sample: Sample) -> tuple[torch.Tensor, bool]:
    """Return (teacher_log_probs, ok). ok=False means the neutral fallback was used."""
    response_length = sample.response_length
    resp = getattr(sample, "_opd_teacher_response", None)
    if resp is None:
        return _neutral_teacher_log_probs(sample), False

    full = torch.tensor(
        [item[0] for item in resp["meta_info"]["input_token_logprobs"][1:]],
        dtype=torch.float32,
    )
    sliced = full[:0] if response_length == 0 else full[-response_length:]
    if sliced.numel() != response_length:
        # Fewer teacher log-probs than response tokens means the teacher saw a
        # different sequence than the one we trained on (e.g. its server
        # truncated the input). Padding would silently misalign every token, so
        # treat it as a failed teacher call instead.
        return _neutral_teacher_log_probs(sample), False
    return sliced, True


def _normalize_rewards(args, rewards: list[float]) -> list[float]:
    if not _use_task_reward():
        # Zeroing the reward is what makes the OPD arm *pure* OPD: the advantage
        # then reduces to the reverse-KL term alone. Without OPD there is no such
        # term, so the same setting would hand the estimator an all-zero reward
        # vector and the run would train on nothing at all, silently.
        if not getattr(args, "use_opd", False):
            raise ValueError(
                "MULTI_DOMAIN_USE_TASK_REWARD=0 zeroes the task reward, which only makes "
                "sense with --use-opd (the reverse-KL term replaces it). This run has OPD off, "
                "so the advantage would be identically zero. Set it to 1 for GRPO."
            )
        return [0.0] * len(rewards)

    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        reward_tensor = torch.tensor(rewards, dtype=torch.float)
        expected = args.n_samples_per_prompt * args.rollout_batch_size
        if reward_tensor.shape[-1] == expected:
            reward_tensor = reward_tensor.reshape(-1, args.n_samples_per_prompt)
        else:
            reward_tensor = reward_tensor.view(-1, reward_tensor.shape[-1])
        reward_tensor = reward_tensor - reward_tensor.mean(dim=-1, keepdim=True)

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            reward_tensor = reward_tensor / (reward_tensor.std(dim=-1, keepdim=True) + 1e-6)

        return reward_tensor.flatten().tolist()

    return rewards


def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards = []
    sample_tasks = []
    task_counts = {task: 0 for task in TASKS}
    task_raw_rewards = {task: [] for task in TASKS}
    task_scores = {task: [] for task in TASKS}
    # ReTool and SQL set sample.tool_call_count; the search rollout counts turns itself.
    task_tool_call_counts = {task: [] for task in TASKS}
    # slime logs length twice and the two mean different things: response_len/*
    # is Sample.effective_response_length, the tokens that carry gradient
    # (rollout.py:1313), while raw_response_length/* is the full response
    # including tool output and retrieved passages (data.py:451). Mirror both per
    # task rather than picking one. The clip_ratio beside them compares every
    # sample against a single args.rollout_max_response_len (data.py:456), which
    # is right for at most one task once the context budgets differ -- hence the
    # per-task truncation count here.
    task_response_lengths = {task: [] for task in TASKS}
    task_raw_response_lengths = {task: [] for task in TASKS}
    task_truncated = {task: 0 for task in TASKS}
    failed_teachers = {task: 0 for task in TASKS}
    # Samples whose OPD term was neutralised (see _neutral_teacher_log_probs).
    # A superset of failed_teachers: it also counts length mismatches.
    neutralized_teachers = {task: 0 for task in TASKS}
    has_round_number = any(sample.metadata and "round_number" in sample.metadata for sample in samples)

    teacher_enabled = _teacher_enabled(args)

    for sample in samples:
        task = getattr(sample, "_opd_task", None) or _get_task(sample)
        sample_tasks.append(task)
        task_counts[task] += 1
        if teacher_enabled and getattr(sample, "_opd_teacher_response", None) is None:
            failed_teachers[task] += 1

        # Only the served-teacher path sets this. Under the Megatron teacher the
        # training forward writes the real values under the same key, so setting
        # a tensor of zeros here would race it; under GRPO there is no
        # teacher at all. Unset is safe in both cases: actor.py:277 skips the key
        # when absent, and if the Megatron forward somehow does not run,
        # apply_opd_kl_to_advantages raises instead of silently distilling
        # against zeros.
        if teacher_enabled:
            sample.teacher_log_probs, teacher_ok = _extract_teacher_log_probs(sample)
            if not teacher_ok:
                neutralized_teachers[task] += 1
        sample.metadata = _metadata(sample)
        sample.metadata["task"] = task
        # Also as "source_name", which is the key slime itself reads
        # (slime.utils.data.get_source) and carries into the trainer as
        # rollout_data["source_names"] (ray/rollout.py). Under --opd-type
        # megatron with --opd-teacher-loads that is what routes each sample to
        # its own teacher, and it is what splits the per-source metrics.
        sample.metadata["source_name"] = task
        if has_round_number:
            sample.metadata.setdefault("round_number", 0)

        raw_reward = float(getattr(sample, "_opd_task_score", 0.0) or 0.0)
        raw_rewards.append(raw_reward)
        task_raw_rewards[task].append(raw_reward)
        task_score = float(getattr(sample, "_opd_task_score_unshaped", raw_reward) or 0.0)
        task_scores[task].append(task_score)
        if task in ("retool", "sql"):
            task_tool_call_counts[task].append(float(getattr(sample, "tool_call_count", 0) or 0))

        task_response_lengths[task].append(float(getattr(sample, "effective_response_length", 0) or 0))
        task_raw_response_lengths[task].append(float(getattr(sample, "response_length", 0) or 0))
        if getattr(sample, "status", None) == Sample.Status.TRUNCATED:
            task_truncated[task] += 1

    if any(neutralized_teachers.values()):
        print(
            f"[multi-domain] WARN: teacher logprobs missing or misaligned for {neutralized_teachers} "
            f"samples (no response: {failed_teachers}); their OPD advantage was set to 0. "
            "If this is not rare, the teacher server is overloaded or truncating -- "
            "see *_TEACHER_MAX_INFLIGHT / *_TEACHER_MAX_TOTAL_TOKENS.",
            file=sys.stderr,
            flush=True,
        )

    # REPLAY_LOSS=rft (grpo_multi.sh's replay arm under slime's SFT loss) and RAFT=1 (its online arm under
    # the same loss): the rejected samples leave the loss here. Sample.remove_sample makes slime zero
    # their loss mask when it builds the training data (ray/rollout.py, after this hook, on replayed and
    # on live rollouts alike), so they count in neither the numerator nor the denominator; the batch and
    # the loss normalisation are otherwise the arm's own.
    rft_keep = None
    raft = _raft()
    if raft and _replay_loss():
        raise ValueError("RAFT=1 trains on the student's own rollouts and REPLAY_LOSS on the replayed logs: not both")
    if _replay_loss() == "rft" or raft:
        if not _use_task_reward():
            raise ValueError("REPLAY_LOSS=rft / RAFT=1 select the samples by the task reward; MULTI_DOMAIN_USE_TASK_REWARD=0 zeroes it")
        cap = os.environ.get("RFT_MAX_PER_PROMPT", "").strip()
        rft_keep = _rft_keep(raw_rewards, args.n_samples_per_prompt, int(cap) if cap else None)
        for sample, kept in zip(samples, rft_keep, strict=True):
            if not kept:
                sample.remove_sample = True

    train_rewards = _normalize_rewards(args, raw_rewards)
    task_train_rewards = {task: [] for task in TASKS}
    for task, reward in zip(sample_tasks, train_rewards, strict=False):
        task_train_rewards[task].append(float(reward))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def mean_or_previous(task: str, values: list[float], cache: dict[str, float]) -> float:
        if values:
            cache[task] = mean(values)
        return cache[task]

    _SEEN_TASKS.update(task for task in TASKS if task_counts[task])
    seen = [task for task in TASKS if task in _SEEN_TASKS]

    log_metrics = {
        "rollout/student_task_score": mean([score for task in seen for score in task_scores[task]]),
        "rollout/shaped_reward_mean": mean(raw_rewards),
    }
    if rft_keep is not None:
        # the share of the step's samples in the SFT loss, overall and per domain
        log_metrics["rollout/rft_kept_ratio"] = mean([float(k) for k in rft_keep])
        for task in seen:
            kept_t = [float(k) for k, t in zip(rft_keep, sample_tasks, strict=True) if t == task]
            log_metrics[f"rollout/{METRIC_PREFIX[task]}_rft_kept_ratio"] = mean(kept_t)
    for task in seen:
        prefix = METRIC_PREFIX[task]
        truncated_ratio = [task_truncated[task] / task_counts[task]] if task_counts[task] else []
        log_metrics.update(
            {
                f"rollout/{prefix}_count": task_counts[task],
                f"rollout/{prefix}_sample_count": task_counts[task],
                f"rollout/{prefix}_reward_mean": mean_or_previous(task, task_raw_rewards[task], _LAST_TASK_REWARD_MEAN),
                f"rollout/{prefix}_task_score_mean": mean_or_previous(task, task_scores[task], _LAST_TASK_SCORE_MEAN),
                f"rollout/{prefix}_train_reward_mean": mean_or_previous(
                    task, task_train_rewards[task], _LAST_TASK_TRAIN_REWARD_MEAN
                ),
                f"rollout/{prefix}_response_len_mean": mean_or_previous(
                    task, task_response_lengths[task], _LAST_TASK_RESPONSE_LEN_MEAN
                ),
                f"rollout/{prefix}_raw_response_len_mean": mean_or_previous(
                    task, task_raw_response_lengths[task], _LAST_TASK_RAW_RESPONSE_LEN_MEAN
                ),
                f"rollout/{prefix}_truncated_ratio": mean_or_previous(task, truncated_ratio, _LAST_TASK_TRUNCATED_RATIO),
                # Should sit at 0. Anything else is samples the teacher never scored and
                # that therefore contributed nothing this step.
                f"rollout/{prefix}_teacher_neutralized": neutralized_teachers[task],
            }
        )
        if task in ("retool", "sql"):
            log_metrics[f"rollout/{prefix}_tool_call_count_mean"] = mean_or_previous(
                task, task_tool_call_counts[task], _LAST_TASK_TOOL_CALL_COUNT_MEAN
            )
    if "retool" in seen:
        log_metrics["rollout/retool_shaped_reward_mean"] = log_metrics["rollout/retool_reward_mean"]
    print(
        "[multi-domain] "
        + " ".join(
            f"{METRIC_PREFIX[task]}_samples={task_counts[task]} "
            f"{METRIC_PREFIX[task]}_reward_mean={log_metrics[f'rollout/{METRIC_PREFIX[task]}_reward_mean']:.4f} "
            f"{METRIC_PREFIX[task]}_train_reward_mean={log_metrics[f'rollout/{METRIC_PREFIX[task]}_train_reward_mean']:.4f} "
            f"{METRIC_PREFIX[task]}_trunc={log_metrics[f'rollout/{METRIC_PREFIX[task]}_truncated_ratio']:.3f}"
            for task in seen
        ),
        flush=True,
    )

    try:
        import wandb

        if wandb.run is not None and raw_rewards:
            wandb.log(log_metrics)
    except ImportError:
        pass

    return raw_rewards, train_rewards
