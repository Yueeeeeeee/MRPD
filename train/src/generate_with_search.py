# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This version queries the local retriever, with optional log probability collection

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# System prompt and tool definition, matched with eval_generate_with_search.py.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools to answer questions. "
    "You have access to a search tool. When you need to look up information, "
    "call the search tool as described below."
)

TOOL_SYSTEM_PROMPT = """# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
<tool name=\"search\" description=\"Search for information.\" args=\"{\\\"query\\\": string}\" />
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": int(os.environ.get("SEARCH_R1_MAX_TURNS", "8")),
    # Ceiling on ONE turn's generation. Without it a turn's cap is
    # rollout_max_response_len, i.e. the whole response budget, and a single
    # turn that never emits </tool_call> or </answer> runs the entire context
    # in one /generate call -- long enough to blow past the router timeout, get
    # its connection dropped, and be retried from scratch (up to 60 times).
    # A search turn is a query or a final answer; it does not need thousands of
    # tokens. 4096 is what the per-turn cap effectively was before
    # ROLLOUT_MAX_RESPONSE_LEN went to 8192, and the total response budget is
    # still bounded by rollout_max_context_len either way.
    "max_tokens_per_turn": int(os.environ.get("SEARCH_R1_MAX_TOKENS_PER_TURN", "4096")),
    # Per-task context budget, for cross-domain runs where one global
    # --rollout-max-context-len has to serve two tasks whose single-domain
    # settings differ (math 16384, search 9216). Applied with min() against the
    # global, so it can only ever narrow: a run that forgets to set it falls
    # back to the wider global budget rather than silently truncating.
    # 0 or unset means no override.
    "max_context_len": int(os.environ.get("SEARCH_R1_MAX_CONTEXT_LEN", "0") or 0),
    "topk": 3,
    "search_concurrency": 256,
    # ============== Retriever ==============
    "search_backend": "local",
    "local": {
        "search_url": os.environ.get(
            "SEARCH_URL",
            "http://127.0.0.1:8000/retrieve",
        ),  # Comma-separate multiple local retriever URLs.
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


def _tool_system_content() -> str:
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_SYSTEM_PROMPT}"


def _system_block() -> str:
    return f"<|im_start|>system\n{_tool_system_content()}<|im_end|>"


# The system prompt is a property of the runtime, not of the data: the released
# checkpoints were SFT'd on exactly _tool_system_content() (byte-identical to the
# system message in baohao/*-cold-start), and the RL and eval prompt sets carry
# only user turns. So a system message arriving in the data is dropped and
# _system_block() is rendered instead -- eval does the same, which is the point:
# the two sides must render the same prompt or the comparison is not one.
#
# Dropping it silently is the part worth guarding. A custom dataset carrying a
# *different* system prompt would be off-distribution for these checkpoints and
# is worth one loud line, once per process.
_WARNED_SYSTEM_PROMPTS: set[str] = set()


def _note_dropped_system_prompt(content: str) -> None:
    text = (content or "").strip()
    if not text or text == _tool_system_content().strip():
        return
    if text in _WARNED_SYSTEM_PROMPTS:
        return
    _WARNED_SYSTEM_PROMPTS.add(text)
    print(
        "[search-r1] ignoring a system prompt supplied by the dataset; the runtime "
        "renders its own so that training and eval agree. Dropped:\n"
        f"    {text[:200]}{'...' if len(text) > 200 else ''}",
        flush=True,
    )


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(_stringify_message_content(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "content" in content:
            return _stringify_message_content(content.get("content"))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _parse_chat_transcript(prompt: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_end\|>)<\|im_end\|>",
        re.DOTALL,
    )
    messages: list[dict[str, str]] = []
    for role, content in pattern.findall(prompt):
        text = content.strip()
        if role == "assistant" and not text:
            continue
        if not text:
            continue
        if role == "system":
            _note_dropped_system_prompt(text)
            continue
        messages.append({"role": role, "content": text})
    return messages


def _normalize_prompt_messages(prompt: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if prompt is None:
        return []
    if isinstance(prompt, str) and "<|im_start|>" in prompt:
        parsed_messages = _parse_chat_transcript(prompt)
        if parsed_messages:
            return parsed_messages
    if isinstance(prompt, list):
        normalized_messages = []
        for message in prompt:
            if not isinstance(message, dict):
                content = str(message)
                if content.strip():
                    normalized_messages.append({"role": "user", "content": content})
                continue
            role = message.get("role", "user")
            if role == "system":
                _note_dropped_system_prompt(_stringify_message_content(message.get("content", "")))
                continue
            content = _stringify_message_content(message.get("content", ""))
            if content.strip():
                normalized_messages.append({"role": role, "content": content})
        return normalized_messages
    content = str(prompt)
    return [{"role": "user", "content": content}] if content.strip() else []


def _render_message_with_separator(message: dict[str, Any]) -> str:
    role = message.get("role", "user")
    content = message.get("content", "") or ""
    if role == "user":
        return f"\n<|im_start|>user\n{content}<|im_end|>"
    if role == "assistant":
        return f"\n<|im_start|>assistant\n{content}<|im_end|>"
    if role == "tool":
        return (
            "\n<|im_start|>user\n"
            "<tool_response>\n"
            f"{content}\n"
            "</tool_response><|im_end|>"
        )
    return ""


def _render_messages(messages: list[dict[str, Any]]) -> str:
    return "".join(_render_message_with_separator(message) for message in messages)


def format_conversation_with_tools(prompt: str | list[dict[str, Any]]) -> str:
    """Add the eval-style system/tool prompt to a raw or already-rendered prompt."""
    if isinstance(prompt, str) and (TOOL_SYSTEM_PROMPT in prompt or '<tool name="search"' in prompt):
        return prompt

    if isinstance(prompt, str) and prompt.startswith("<|im_start|>system"):
        end_tag = "<|im_end|>"
        end = prompt.find(end_tag)
        if end != -1:
            system_content_end = end
            return (
                prompt[:system_content_end].rstrip()
                + "\n\n"
                + TOOL_SYSTEM_PROMPT
                + prompt[system_content_end:]
            )

    if isinstance(prompt, str) and "<|im_start|>user" in prompt:
        return _system_block() + "\n" + prompt

    messages = _normalize_prompt_messages(prompt)
    return _system_block() + _render_messages(messages) + "\n<|im_start|>assistant\n"


def _extract_user_messages(prompt: str | list[dict[str, Any]]) -> list[str]:
    if not isinstance(prompt, str):
        return [
            message["content"]
            for message in _normalize_prompt_messages(prompt)
            if message.get("role") == "user"
        ]
    matches = re.findall(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", prompt, re.DOTALL)
    if matches:
        return [match.strip() for match in matches if match.strip()]
    return [prompt.strip()] if prompt.strip() else []


def _extract_tool_response_content(observation: str) -> str | None:
    match = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", observation, re.DOTALL)
    if match:
        return match.group(1).strip()
    start_tag = "<tool_response>"
    start = observation.find(start_tag)
    if start != -1:
        return observation[start + len(start_tag):].strip()
    return None


def _apply_final_token_clip(
    tokenizer,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float] | None,
    max_context_length: int,
) -> tuple[list[int], str, list[int], list[int], list[float] | None, bool]:
    max_response_tokens = max(0, max_context_length - len(prompt_token_ids))
    if len(response_token_ids) <= max_response_tokens:
        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        return prompt_token_ids + response_token_ids, response_text, response_token_ids, loss_mask, rollout_log_probs, False

    clipped_response_token_ids = response_token_ids[:max_response_tokens]
    clipped_loss_mask = loss_mask[:max_response_tokens]
    clipped_log_probs = rollout_log_probs[:max_response_tokens] if rollout_log_probs is not None else None
    clipped_response = tokenizer.decode(clipped_response_token_ids, skip_special_tokens=False)
    return (
        prompt_token_ids + clipped_response_token_ids,
        clipped_response,
        clipped_response_token_ids,
        clipped_loss_mask,
        clipped_log_probs,
        True,
    )


def _initial_trace_messages(raw_prompt: str | list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _tool_system_content()}]
    for content in _extract_user_messages(raw_prompt):
        messages.append({"role": "user", "content": content})
    return messages


def _passages2string(retrieval_result):
    """
    Convert retrieval results to a formatted string.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    """
    Query the local retriever (SEARCH_R1_CONFIGS["local"]["search_url"]) and format its passages.
    """
    from local_search_server import local_search

    local_config = SEARCH_R1_CONFIGS["local"]
    result = await local_search(
        local_config["search_url"],
        query,
        SEARCH_R1_CONFIGS["topk"],
        proxy=local_config["proxy"],
    )
    return _passages2string(result)


def extract_boxed_answer(prediction: str) -> str | None:
    """Extract the final answer from the last LaTeX-style \boxed{...}, if present."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", prediction)
    if matches:
        return matches[-1].strip()
    return None


def postprocess_responses(resp: str) -> str:
    """Keep only the first complete action/final-answer block."""
    tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, resp, re.DOTALL)
    if tool_call_match:
        return resp[: tool_call_match.end()]
    answer_match = re.search(r"<answer>.*?</answer>", resp, re.DOTALL)
    if answer_match:
        return resp[: answer_match.end()]
    boxed = extract_boxed_answer(resp)
    if boxed is not None:
        boxed_match = re.search(r"\\boxed\{[^}]*\}", resp)
        if boxed_match:
            return resp[: boxed_match.end()]
    return resp


def _trim_response_token_prefix(
    tokenizer,
    raw_token_ids: list[int],
    raw_log_probs: list[float],
    sanitized_response: str,
) -> tuple[list[int], list[float]]:
    raw_response = tokenizer.decode(raw_token_ids)
    if sanitized_response == raw_response:
        return raw_token_ids, raw_log_probs
    if not sanitized_response:
        return [], []
    for prefix_len in range(1, len(raw_token_ids) + 1):
        if tokenizer.decode(raw_token_ids[:prefix_len]) == sanitized_response:
            return raw_token_ids[:prefix_len], raw_log_probs[:prefix_len]
    sanitized_token_ids = tokenizer(sanitized_response, add_special_tokens=False)["input_ids"]
    trimmed_log_probs = raw_log_probs[: len(sanitized_token_ids)]
    if len(trimmed_log_probs) < len(sanitized_token_ids):
        trimmed_log_probs += [0.0] * (len(sanitized_token_ids) - len(trimmed_log_probs))
    return sanitized_token_ids, trimmed_log_probs


def _extract_first_tool_call(prediction: str) -> dict | None:
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if not tool_call_match:
        return None
    try:
        json_str = tool_call_match.group(1).replace("\n", "\\n")
        tool_call_data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(tool_call_data, dict):
        return None
    return tool_call_data


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """Return (action, content) where action is 'search'|'answer'|'boxed_answer'|None."""
    tool_call = _extract_first_tool_call(prediction)
    if tool_call and tool_call.get("name") == "search":
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        if isinstance(arguments, dict):
            return "search", str(arguments.get("query", ""))
        return "search", ""

    match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if match:
        return "answer", match.group(1).strip()

    boxed = extract_boxed_answer(prediction)
    if boxed is not None:
        return "boxed_answer", boxed

    return None, ""


async def execute_predictions(prediction: str) -> tuple[str, bool, float]:
    """Run the action implied by `prediction`.

    Returns (next_observation, done, search_seconds). The third element is the
    wall time this turn spent waiting on retrieval -- the concurrency semaphore
    plus the HTTP round trip -- and is zero for the turns that do not search.
    The caller accumulates it into ``sample.non_generation_time``, which is what
    slime needs to report perf/non_generation_time (rollout.py:1332): without it
    a rollout that is slow because the retriever is slow looks exactly like one
    that is slow because generation is slow.
    """
    action, content = postprocess_predictions(prediction)
    search_seconds = 0.0
    # A turn that stopped at </tool_call> is still open, so the observation closes
    # it. A turn the model ended with its own <|im_end|> is closed already, and
    # closing it again would put two <|im_end|> in a row.
    close = "" if prediction.endswith("<|im_end|>") else "<|im_end|>"

    if action == "search":
        search_t0 = time.monotonic()
        try:
            async with SEMAPHORE:
                search_results = await search(content)
            tool_response_content = search_results.strip()
        except Exception as e:
            tool_response_content = f"[ERROR] {type(e).__name__}: {e}"
        search_seconds = time.monotonic() - search_t0
        next_obs = (
            f"{close}\n"
            "<|im_start|>user\n<tool_response>\n"
            f"{tool_response_content}\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
        done = False
    elif action == "answer" or action == "boxed_answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            f"{close}\n"
            "<|im_start|>user\n"
            "Your previous action is invalid. "
            "If you want to use a tool, you should return a <tool_call>...</tool_call> block. "
            "If you want to give the final answer, you should put the answer in \\boxed{}. "
            "Please try again.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        done = False

    return next_obs, done, search_seconds


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    sample_t0 = time.monotonic()
    search_seconds_total = 0.0

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = format_conversation_with_tools(sample.prompt)
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    if getattr(args, "rollout_max_context_len", None) is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
    if SEARCH_R1_CONFIGS["max_context_len"] > 0:
        max_context_length = min(max_context_length, SEARCH_R1_CONFIGS["max_context_len"])
    max_context_length = max_context_length - 16
    if len(prompt_tokens_ids) >= max_context_length:
        sample.tokens = prompt_tokens_ids[:max_context_length]
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []
        sample.prompt = state.tokenizer.decode(sample.tokens, skip_special_tokens=False)
        sample.status = Sample.Status.TRUNCATED
        sample.rollout_log_probs = []
        return sample

    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if SEARCH_R1_CONFIGS["return_logprob"] else None
    messages = _initial_trace_messages(sample.prompt)
    search_count = 0
    final_was_clipped = False
    last_finish_reason = None

    def _merged_stop_strings(existing_stop) -> list[str]:
        stop_strings: list[str] = []
        if isinstance(existing_stop, str):
            stop_strings.append(existing_stop)
        elif isinstance(existing_stop, list):
            stop_strings.extend(str(item) for item in existing_stop)
        for stop in ("</tool_call>", "</answer>"):
            if stop not in stop_strings:
                stop_strings.append(stop)
        return stop_strings

    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break
        remaining_context = max_context_length - total_length
        turn_sampling_params = dict(sampling_params)
        turn_sampling_params["stop"] = _merged_stop_strings(turn_sampling_params.get("stop"))
        turn_sampling_params["no_stop_trim"] = True
        turn_sampling_params["max_new_tokens"] = min(
            turn_sampling_params.get("max_new_tokens", remaining_context),
            remaining_context,
            SEARCH_R1_CONFIGS["max_tokens_per_turn"],
        )
        if turn_sampling_params["max_new_tokens"] <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        # Append-only, as in generate_with_retool.py: the request carries exactly the
        # tokens stored for training, so the log-probs describe the trained sequence.
        # Sending text re-tokenizes the transcript each turn, and where a turn
        # boundary merges into one token the two streams differ.
        payload = {
            "input_ids": prompt_tokens_ids + response_token_ids,
            "sampling_params": turn_sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)
        last_finish_reason = output["meta_info"]["finish_reason"]["type"]

        if last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            break

        if SEARCH_R1_CONFIGS["return_logprob"]:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )
            raw_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            raw_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = postprocess_responses(state.tokenizer.decode(raw_token_ids, skip_special_tokens=False))
            cur_response_token_ids, cur_response_log_probs = _trim_response_token_prefix(
                state.tokenizer,
                raw_token_ids,
                raw_log_probs,
                cur_response,
            )
        else:
            cur_response = postprocess_responses(output["text"])
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        messages.append({"role": "assistant", "content": cur_response})

        # Add log probs if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        action, _ = postprocess_predictions(cur_response)
        if last_finish_reason == "length" and action not in {"search", "answer", "boxed_answer"}:
            sample.status = Sample.Status.TRUNCATED
            break

        if SEARCH_R1_CONFIGS["max_turns"] == 1:
            break

        next_obs, done, turn_search_seconds = await execute_predictions(cur_response)
        search_seconds_total += turn_search_seconds
        if done:
            break
        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        if len(prompt_tokens_ids) + len(response_token_ids) + len(obs_tokens_ids) > max_context_length:
            # The observation does not fit. End the trajectory after the model's
            # turn: a cut observation is one the model could never answer, and
            # its tokens would only ever be masked out.
            sample.status = Sample.Status.TRUNCATED
            break
        tool_response_content = _extract_tool_response_content(next_obs)
        if tool_response_content is not None:
            messages.append({"role": "tool", "content": tool_response_content})
            search_count += 1
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

    # Store statistics for wandb logging
    (
        final_tokens,
        final_response,
        final_response_token_ids,
        final_loss_mask,
        final_rollout_log_probs,
        final_was_clipped,
    ) = _apply_final_token_clip(
        state.tokenizer,
        prompt_tokens_ids,
        response_token_ids,
        loss_mask,
        rollout_log_probs,
        max_context_length,
    )

    sample.tokens = final_tokens
    sample.response_length = len(final_response_token_ids)
    sample.response = final_response
    sample.loss_mask = final_loss_mask
    sample.prompt = prompt_text
    sample.messages = messages
    sample.payload_text = prompt_text + final_response
    sample.payload_has_system = "<|im_start|>system" in sample.payload_text
    sample.payload_has_tools = "# Tools" in sample.payload_text
    sample.tool_call_count = search_count
    sample.search_count = search_count
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["messages"] = messages
    sample.metadata["payload_text"] = sample.payload_text
    sample.metadata["payload_has_system"] = sample.payload_has_system
    sample.metadata["payload_has_tools"] = sample.payload_has_tools
    sample.metadata["tool_call_count"] = search_count
    sample.metadata["search_count"] = search_count
    sample.metadata["max_context_length"] = max_context_length
    sample.metadata["final_was_clipped"] = final_was_clipped

    # Time this sample spent waiting on the retriever rather than generating.
    # slime turns it into perf/non_generation_time/* and
    # perf/longest_sample_tokens_per_sec_without_non_generation, which is the
    # only way to tell a retriever-bound rollout from a generation-bound one.
    # sample_seconds is the sample's own end-to-end wall time; the two together
    # give the fraction, which rollout_time cannot because samples overlap.
    sample.non_generation_time = search_seconds_total
    sample.metadata["search_seconds"] = search_seconds_total
    sample.metadata["sample_seconds"] = time.monotonic() - sample_t0

    # Store log probs if enabled
    if SEARCH_R1_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = final_rollout_log_probs if final_rollout_log_probs is not None else []
        if len(sample.rollout_log_probs) != sample.response_length:
            if len(sample.rollout_log_probs) < sample.response_length:
                sample.rollout_log_probs += [0.0] * (sample.response_length - len(sample.rollout_log_probs))
            else:
                sample.rollout_log_probs = sample.rollout_log_probs[: sample.response_length]

    if sample.response_length == 0:
        pad_token_id = (
            state.tokenizer.pad_token_id
            if state.tokenizer.pad_token_id is not None
            else (state.tokenizer.eos_token_id or 0)
        )
        sample.tokens = list(sample.tokens) + [pad_token_id]
        sample.response = state.tokenizer.decode([pad_token_id], skip_special_tokens=False)
        sample.response_length = 1
        sample.loss_mask = [0]
        sample.rollout_log_probs = [0.0]
        if sample.status not in {Sample.Status.ABORTED, Sample.Status.TRUNCATED}:
            sample.status = Sample.Status.TRUNCATED

    if final_was_clipped:
        sample.status = Sample.Status.TRUNCATED

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["opd_rev_kl_weight"] = 1.0

    if sample.status == Sample.Status.PENDING:
        if last_finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
        elif last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
        else:
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # solution_str is the whole transcript because the format and retrieval
    # checks need it; response_str keeps the *answer* out of the prompt, which
    # carries \boxed{$Answer} and \boxed{Beijing} as examples of its own.
    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        response_str=sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    # +1 / -1, the same convention as the math environment's math_dapo score, so
    # rollout/raw_reward means the same thing in both. GRPO is
    # indifferent (group normalisation absorbs any affine map) and OPD zeroes the
    # reward anyway; only the comparability of the curves cares.
    #
    # The grader's own value is not shifted: eval/search/eval.py calls
    # compute_score_em directly and keeps reporting 0/1 accuracy. "Correct" is
    # score >= 1.0, the same test eval.py applies -- the format-shaping terms in
    # compute_score_em are all zero here, so score is 1.0 or 0.0 in practice.
    return 1.0 if float(score) >= 1.0 else -1.0
