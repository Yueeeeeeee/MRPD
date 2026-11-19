"""Multi-turn text-to-SQL rollouts for slime: the ReTool loop of generate_with_retool.py
with the Python sandbox replaced by sql_tools' read-only SQLite executor.

    --custom-generate-function-path generate_with_sql.generate
    --custom-rm-path               generate_with_sql.reward_func

The prompt file carries the schema and question as the user message, the gold query as
the label and db_id in the metadata (scripts/data/prepare_sql.py). Each turn ends at
</tool_call> (the query is run and its result appended as a <tool_response> user turn,
loss_mask 0) or with the model's own end of turn (done: the turn's last ```sql block or
<answer> is the final query, none scores -1). The reward is +1 when the final query's
result equals the gold query's and -1 otherwise (sql_tools.score_response). Token streams are
append-only and the engine's token ids are the source of truth, exactly as in the math
environment, so --use-rollout-logprobs sees the sequence that is trained on.

Environment knobs: sql_tools' SQL_* (SQL_DB_ROOT is required), SQL_MAX_CONTEXT_LEN (a
per-task cap applied with min() against --rollout-max-context-len, for mixed runs) and
GENERATE_SAMPLE_TIMEOUT (wall-clock cap per sample, seconds).
"""

from __future__ import annotations

import asyncio

# Installed before slime's AsyncLoopThread creates its event loop, as in
# generate_with_retool.py: uvloop + many concurrent sockets aborts under load.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
import os
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

import sql_tools
from sql_tools import TOOL_CONFIGS

_SAMPLE_TIMEOUT = float(os.environ.get("GENERATE_SAMPLE_TIMEOUT", "600"))
_FABRICATED_LOGPROB_TOKENS = 0


def _count_fabricated_logprobs(reason: str, num_tokens: int) -> None:
    global _FABRICATED_LOGPROB_TOKENS
    _FABRICATED_LOGPROB_TOKENS += num_tokens
    print(
        f"[sql] fabricated log probs ({reason}) for {num_tokens} tokens; masked out of the loss. "
        f"running total={_FABRICATED_LOGPROB_TOKENS}",
        flush=True,
    )


def _max_context_len_override() -> int:
    """Per-task context budget for mixed runs; 0 means none (see generate_with_retool)."""
    return int(os.environ.get("SQL_MAX_CONTEXT_LEN", "0") or 0)


def db_id_of(sample: Sample) -> str:
    metadata = sample.metadata or {}
    db_id = metadata.get("db_id")
    if not db_id:
        raise ValueError("the SQL environment needs sample.metadata['db_id'] (see scripts/data/prepare_sql.py)")
    return str(db_id)


def gold_sql_of(sample: Sample) -> str:
    label = sample.label
    if isinstance(label, dict):
        label = label.get("sql") or label.get("query") or label.get("ground_truth") or ""
    return str(label or "")


def _apply_final_token_clip(tokenizer, prompt_token_ids, response_token_ids, loss_masks, rollout_log_probs, max_context_length):
    max_response_tokens = max(0, max_context_length - len(prompt_token_ids))
    if len(response_token_ids) <= max_response_tokens:
        text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        return prompt_token_ids + response_token_ids, text, response_token_ids, loss_masks, rollout_log_probs, False
    ids = response_token_ids[:max_response_tokens]
    masks = loss_masks[:max_response_tokens]
    log_probs = rollout_log_probs[:max_response_tokens] if rollout_log_probs is not None else None
    return prompt_token_ids + ids, tokenizer.decode(ids, skip_special_tokens=False), ids, masks, log_probs, True


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    db_id = db_id_of(sample)

    prompt = sql_tools.format_conversation_with_tools(sample.prompt)
    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    recorded_messages: list[dict[str, Any]] = [{"role": "system", "content": sql_tools.full_system_prompt()}]
    recorded_messages += sql_tools.normalize_prompt_messages(sample.prompt)

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
    task_context_len = _max_context_len_override()
    if task_context_len > 0:
        max_context_length = min(max_context_length, task_context_len)
    max_context_length = max_context_length - 16  # buffer, as in the math environment

    # Reset, not just initialise: an ABORTED group is regenerated through this same
    # function with the previous attempt's state still on the Sample.
    sample.tokens = list(prompt_tokens_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.messages = list(recorded_messages)

    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    tool_call_count = 0
    last_finish_reason = None

    async def _run_turns():
        nonlocal response_token_ids, loss_masks, tool_call_count, last_finish_reason
        for turn in range(TOOL_CONFIGS["max_turns"]):
            current_token_ids = prompt_tokens_ids + response_token_ids
            total_length = len(current_token_ids)
            if total_length >= max_context_length:
                sample.status = Sample.Status.TRUNCATED
                break
            current_sampling_params = dict(sampling_params)
            current_sampling_params["stop"] = sql_tools.merged_stop_strings(current_sampling_params.get("stop"))
            current_sampling_params["no_stop_trim"] = True
            current_sampling_params["max_new_tokens"] = min(
                current_sampling_params["max_new_tokens"],
                TOOL_CONFIGS["max_tokens_per_turn"],
                max_context_length - total_length,
            )
            if current_sampling_params["max_new_tokens"] <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            output = await post(url, {"input_ids": current_token_ids, "sampling_params": current_sampling_params, "return_logprob": True})
            last_finish_reason = output["meta_info"]["finish_reason"]["type"]
            if last_finish_reason == "abort":
                sample.status = Sample.Status.ABORTED
                break
            if "output_token_logprobs" not in output["meta_info"]:
                # No per-token log probs: nothing to train the importance ratio against.
                # Abort so the group is regenerated, as the math environment does.
                print(f"[sql] db={db_id} turn={turn} no output_token_logprobs in meta_info -> ABORTED", flush=True)
                sample.status = Sample.Status.ABORTED
                return
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids, skip_special_tokens=False)
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
            response_token_ids += cur_response_token_ids
            loss_masks += [1] * len(cur_response_token_ids)

            assistant_message = sql_tools.build_assistant_message(cur_response)
            if assistant_message is not None:
                recorded_messages.append(assistant_message)
                sample.messages = list(recorded_messages)

            if last_finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                break
            if TOOL_CONFIGS["max_tool_calls"] <= 0 or TOOL_CONFIGS["max_turns"] == 1:
                break

            next_obs, done, tool_message = await sql_tools.execute_predictions(cur_response, db_id)
            if done:
                break
            assert next_obs != "", "Next observation should not be empty."
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            if len(prompt_tokens_ids) + len(response_token_ids) + len(obs_tokens_ids) > max_context_length:
                # An observation that does not fit ends the trajectory after the model's turn.
                sample.status = Sample.Status.TRUNCATED
                break
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)
            if "<tool_response>" in next_obs:
                tool_call_count += 1
            if tool_message is not None:
                recorded_messages.append(tool_message)
                sample.messages = list(recorded_messages)
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
            assert len(response_token_ids) == len(sample.rollout_log_probs), (
                f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} vs {len(sample.rollout_log_probs)}"
            )
            if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
                break

    try:
        await asyncio.wait_for(_run_turns(), timeout=_SAMPLE_TIMEOUT)
    except asyncio.TimeoutError:
        print(f"[sql] db={db_id} sample TIMEOUT after {_SAMPLE_TIMEOUT:.0f}s (tool_calls={tool_call_count})", flush=True)
        sample.status = Sample.Status.TRUNCATED

    (
        sample.tokens,
        sample.response,
        final_response_token_ids,
        sample.loss_mask,
        sample.rollout_log_probs,
        final_was_clipped,
    ) = _apply_final_token_clip(
        state.tokenizer, prompt_tokens_ids, response_token_ids, loss_masks, sample.rollout_log_probs, max_context_length
    )
    sample.response_length = len(final_response_token_ids)

    # A fabricated log prob (p = 1) would pin the importance ratio to the clip
    # boundary under --use-rollout-logprobs, so every fabricated entry is masked out.
    if sample.rollout_log_probs is None:
        _count_fabricated_logprobs("no engine logprobs", sample.response_length)
        sample.rollout_log_probs = [0.0] * sample.response_length
        sample.loss_mask = [0] * sample.response_length
    elif len(sample.rollout_log_probs) != sample.response_length:
        n = len(sample.rollout_log_probs)
        if n < sample.response_length:
            pad = sample.response_length - n
            _count_fabricated_logprobs("short logprobs", pad)
            sample.rollout_log_probs = sample.rollout_log_probs + [0.0] * pad
            sample.loss_mask = list(sample.loss_mask[:n]) + [0] * pad
        else:
            sample.rollout_log_probs = sample.rollout_log_probs[: sample.response_length]

    # Never a zero-length response: it would produce an empty micro-batch.
    if sample.response_length == 0:
        pad_token_id = state.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = state.tokenizer.eos_token_id or 0
        sample.tokens = list(sample.tokens) + [pad_token_id]
        sample.response = state.tokenizer.decode([pad_token_id], skip_special_tokens=False)
        sample.response_length = 1
        sample.loss_mask = [0]
        sample.rollout_log_probs = [0.0]
        if sample.status not in {Sample.Status.ABORTED, Sample.Status.TRUNCATED}:
            sample.status = Sample.Status.TRUNCATED
    if final_was_clipped:
        sample.status = Sample.Status.TRUNCATED

    sample.payload_text = sql_tools.render_recorded_messages(recorded_messages)
    sample.payload_has_system = True
    sample.payload_has_tools = True
    sample.tool_call_count = tool_call_count
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["round_number"] = tool_call_count
    sample.metadata["opd_rev_kl_weight"] = 1.0

    if sample.status not in {Sample.Status.TRUNCATED, Sample.Status.ABORTED}:
        if last_finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
        elif last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
        elif last_finish_reason == "stop":
            sample.status = Sample.Status.COMPLETED
    return sample


async def reward_func(args, sample, **kwargs):
    """+1 / -1 on execution match of the final query, as a dict with 'score' (the
    launcher passes --reward-key score) and the extracted query under 'pred'."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    result = await asyncio.to_thread(sql_tools.score_response, sample.response, db_id_of(sample), gold_sql_of(sample))
    return {"score": result["score"], "pred": result["pred"], "error": result["error"]}
