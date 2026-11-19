# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
from __future__ import annotations

import asyncio
# IMPORTANT: install the stdlib asyncio loop policy BEFORE slime's
# AsyncLoopThread creates its event loop. uvloop + concurrent subprocess.Popen
# in tool_sandbox + many aiohttp sockets reliably triggers a SIGABRT in the
# event-loop thread under load. The stdlib selector loop is slower but stable.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
import json
import os
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import TOOL_CONFIGS, ToolRegistry

# Wall-clock cap for the full multi-turn rollout of a single sample. When
# exceeded, the partial response is kept (so far decoded tokens) and the
# sample is marked TRUNCATED so the rest of the batch can proceed. Override
# via the GENERATE_SAMPLE_TIMEOUT env var (seconds).
_SAMPLE_TIMEOUT = float(os.environ.get("GENERATE_SAMPLE_TIMEOUT", "600"))


_FABRICATED_LOGPROB_TOKENS = 0


def _count_fabricated_logprobs(reason: str, num_tokens: int) -> None:
    """Log how many response tokens got a synthetic log prob instead of a real one.

    Only reached on abort / length-mismatch paths. If this number is not ~0 the
    importance ratio is being built on masked-out tokens far more often than it
    should be, and the rollout path needs a look.
    """
    global _FABRICATED_LOGPROB_TOKENS
    _FABRICATED_LOGPROB_TOKENS += num_tokens
    print(
        f"[rollout] fabricated log probs ({reason}) for {num_tokens} tokens; "
        f"masked out of the loss. running total={_FABRICATED_LOGPROB_TOKENS}",
        flush=True,
    )


def _tool_call_shaping() -> bool:
    """Whether reward_func adds the tool-call bonus on top of the +1/-1 score."""
    return os.environ.get("RETOOL_TOOL_CALL_SHAPING", "0").lower() not in {"0", "false", "no"}


def _max_context_len_override() -> int:
    """Per-task context budget, for cross-domain runs where one global
    # --rollout-max-context-len has to serve two tasks whose single-domain
    # settings differ (math 16384, search 9216). Applied with min() against the
    # global, so it can only ever narrow: a run that forgets to set it falls
    # back to the wider global budget rather than silently truncating.

    Returns 0 when unset, meaning no override.
    """
    return int(os.environ.get("RETOOL_MAX_CONTEXT_LEN", "0") or 0)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use Python "
    "tools to solve mathematical problems. When you need "
    "to perform calculations, use the code_interpreter "
    "tool to execute code and get results."
)

TOOL_SYSTEM_PROMPT = """# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
__TOOLS__
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def format_conversation_with_tools(
    prompt: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] = None,
    system_prompt: str = None,
    messages: list[dict[str, Any]] = None,
) -> str:
    """Format conversation using a static system prompt and explicit message rendering."""

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = DEFAULT_SYSTEM_PROMPT

    if tools:
        tool_lines = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
        system_content = f"{system_content}\n\n{TOOL_SYSTEM_PROMPT.replace('__TOOLS__', tool_lines)}"

    messages_to_render.append({"role": "system", "content": system_content})

    prompt_messages = _normalize_prompt_messages(prompt)
    if prompt_messages:
        messages_to_render.extend(prompt_messages)

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(
            message
            for message in messages
            if _should_keep_message(message)
        )

    rendered_parts = []
    for message in messages_to_render:
        role = message["role"]
        if role == "system":
            rendered_parts.append(f"<|im_start|>system\n{message['content']}<|im_end|>")
        elif role == "user":
            rendered_parts.append(f"<|im_start|>user\n{message['content']}<|im_end|>")
        elif role == "assistant":
            assistant_text = ["<|im_start|>assistant", message.get("content", "")]
            for tool_call in message.get("tool_calls", []):
                assistant_text.append("<tool_call>")
                assistant_text.append(json.dumps(tool_call["function"], ensure_ascii=False))
                assistant_text.append("</tool_call>")
            assistant_text.append("<|im_end|>")
            rendered_parts.append("\n".join(part for part in assistant_text if part != ""))
        elif role == "tool":
            rendered_parts.append(
                "<|im_start|>user\n"
                "<tool_response>\n"
                f"{message['content']}\n"
                "</tool_response><|im_end|>"
            )

    rendered_parts.append("<|im_start|>assistant\n")
    return "\n".join(rendered_parts)


def _render_recorded_messages(messages: list[dict[str, Any]]) -> str:
    """Render recorded chat messages into transcript text without adding a new assistant turn."""
    rendered_parts = []
    for message in messages:
        role = message["role"]
        if role == "system":
            rendered_parts.append(f"<|im_start|>system\n{message['content']}<|im_end|>")
        elif role == "user":
            rendered_parts.append(f"<|im_start|>user\n{message['content']}<|im_end|>")
        elif role == "assistant":
            assistant_text = ["<|im_start|>assistant", message.get("content", "")]
            for tool_call in message.get("tool_calls", []):
                assistant_text.append("<tool_call>")
                assistant_text.append(json.dumps(tool_call["function"], ensure_ascii=False))
                assistant_text.append("</tool_call>")
            assistant_text.append("<|im_end|>")
            rendered_parts.append("\n".join(part for part in assistant_text if part != ""))
        elif role == "tool":
            rendered_parts.append(
                "<|im_start|>user\n"
                "<tool_response>\n"
                f"{message['content']}\n"
                "</tool_response><|im_end|>"
            )
    return "\n".join(rendered_parts)


def _stringify_message_content(content: Any) -> str:
    """Convert structured message content into plain text for prompting and rewards."""
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


def _has_message_content(content: Any) -> bool:
    """Return whether a message content should be kept in the rendered chat."""
    if isinstance(content, str):
        return bool(content.strip())
    return bool(_stringify_message_content(content).strip())


def _should_keep_message(message: Any) -> bool:
    """Return whether a structured message should be kept in the chat history."""
    if not isinstance(message, dict):
        return False
    if message.get("tool_calls"):
        return True
    return _has_message_content(message.get("content", ""))


def _extract_first_tool_call(prediction: str) -> dict[str, Any] | None:
    """Extract the first tool call from a model response."""
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


def _build_assistant_message(prediction: str) -> dict[str, Any] | None:
    """Build a structured assistant message from a model response."""
    sanitized_prediction = postprocess_responses(prediction)
    tool_call = _extract_first_tool_call(sanitized_prediction)
    if tool_call is None:
        content = sanitized_prediction.rstrip()
        if not content:
            return None
        return {"role": "assistant", "content": content}

    tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, sanitized_prediction, re.DOTALL)
    if tool_call_match is None:
        return {"role": "assistant", "content": sanitized_prediction.rstrip()}

    content = sanitized_prediction[: tool_call_match.start()].rstrip()
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": tool_call}],
    }


def _merged_stop_strings(existing_stop) -> list[str]:
    """Make the ReTool turn end AT </tool_call>, whatever the launcher passed.

    _render_tool_message prepends <|im_end|> to the tool observation, which is
    correct only if generation stopped at the tag with the assistant turn still
    open. Left to the model's own EOS, the stream gets <|im_end|> twice -- the
    model's and the renderer's -- on every tool turn: measured at 94.5% of
    an earlier run's ReTool samples, which ran without --rollout-stop. And nothing
    then cuts the model off after </tool_call>, so it can keep writing (a
    fabricated tool result) before the real one is appended: 3.7% there.

    generate_with_search.py has always added its own stop strings; this is the
    same thing for ReTool. It also stops depending on --rollout-stop reaching
    slime intact through ray's entrypoint quoting, which is where the JSON
    teacher map was lost. Set alongside no_stop_trim, so the tag stays in the
    output tokens for the tool-call parser and the trajectory.
    """
    stop_strings: list[str] = []
    if isinstance(existing_stop, str):
        stop_strings.append(existing_stop)
    elif isinstance(existing_stop, list):
        stop_strings.extend(str(item) for item in existing_stop)
    if "</tool_call>" not in stop_strings:
        stop_strings.append("</tool_call>")
    return stop_strings


def _render_tool_message(tool_message: dict[str, Any], turn: str) -> str:
    """Render a tool message as the user turn that follows `turn`, the model's text.

    A turn that stopped at </tool_call> is still open, so the observation closes
    it. A turn the model ended with its own <|im_end|> is closed already, and
    closing it again would put two <|im_end|> in a row.
    """
    close = "" if turn.endswith("<|im_end|>") else "<|im_end|>"
    return (
        f"{close}\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        f"{tool_message['content']}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_initial_recorded_messages(
    prompt: str | list[dict[str, Any]], system_prompt: str = None
) -> list[dict[str, Any]]:
    """Build the initial structured message list used for debugging and replay."""
    recorded_messages = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}]
    recorded_messages.extend(_normalize_prompt_messages(prompt))
    return recorded_messages


def _apply_final_token_clip(
    tokenizer,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_masks: list[int],
    rollout_log_probs: list[float] | None,
    max_context_length: int,
) -> tuple[list[int], str, list[int], list[int], list[float] | None, bool]:
    """Hard-clip the final sample to the max token budget used by training."""
    max_response_tokens = max(0, max_context_length - len(prompt_token_ids))
    was_clipped = len(response_token_ids) > max_response_tokens

    if not was_clipped:
        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        return prompt_token_ids + response_token_ids, response_text, response_token_ids, loss_masks, rollout_log_probs, False

    clipped_response_token_ids = response_token_ids[:max_response_tokens]
    clipped_loss_masks = loss_masks[:max_response_tokens]
    clipped_log_probs = rollout_log_probs[:max_response_tokens] if rollout_log_probs is not None else None
    clipped_response = tokenizer.decode(clipped_response_token_ids, skip_special_tokens=False)
    clipped_tokens = prompt_token_ids + clipped_response_token_ids
    return (
        clipped_tokens,
        clipped_response,
        clipped_response_token_ids,
        clipped_loss_masks,
        clipped_log_probs,
        True,
    )


def _normalize_prompt_messages(prompt: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Normalize prompt input into chat messages."""
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
                if not content.strip():
                    continue
                normalized_messages.append({"role": "user", "content": content})
                continue
            content = _stringify_message_content(message.get("content", ""))
            if not content.strip():
                continue
            normalized_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": content,
                }
            )
        return normalized_messages
    content = str(prompt)
    if not content.strip():
        return []
    return [{"role": "user", "content": content}]


def _parse_chat_transcript(prompt: str) -> list[dict[str, str]]:
    """Parse a rendered chat transcript back into structured messages."""
    pattern = re.compile(r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_end\|>)<\|im_end\|>", re.DOTALL)
    messages: list[dict[str, str]] = []

    for role, content in pattern.findall(prompt):
        text = content.strip()
        if role == "assistant" and not text:
            # Ignore generation prompts like <|im_start|>assistant\n with no content.
            continue
        if not text:
            continue
        messages.append({"role": role, "content": text})

    return messages


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    """Convert prompt data into a string for reward computation."""
    if isinstance(prompt, str):
        return prompt
    normalized_messages = _normalize_prompt_messages(prompt)
    lines = []
    for message in normalized_messages:
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


def _find_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Find all top-level ``\\boxed{...}`` spans with balanced braces.

    Handles arbitrary brace nesting (e.g. ``\\frac{a^{b}}{c}``) which a
    fixed-depth regex cannot match.
    """
    spans: list[tuple[int, int, str]] = []
    needle = "\\boxed{"
    i = 0
    while True:
        start = text.find(needle, i)
        if start == -1:
            break
        j = start + len(needle)
        depth = 1
        content_start = j
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "\\" and j + 1 < len(text):
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, j + 1, text[content_start:j]))
                    j += 1
                    break
            j += 1
        if depth != 0:
            break
        i = j
    return spans


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Stop on any assistant content that contains a boxed final answer.
    # Prefer the last boxed span so trailing reasoning before the final box is tolerated.
    boxed_spans = _find_boxed_spans(prediction)
    if boxed_spans:
        content = boxed_spans[-1][2].strip()
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_data = _extract_first_tool_call(prediction)
    if tool_call_data:
        tool_name = tool_call_data.get("name")
        arguments = tool_call_data.get("arguments", {})

        # Some models emit arguments as a JSON-encoded string (OpenAI-style)
        # rather than a dict. Decode defensively before .get().
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        if tool_name == "code_interpreter":
            code = arguments.get("code", "")
            if isinstance(code, str) and code.strip():
                return "code", code

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Keep only the first complete <tool_call>...</tool_call> block.
        tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"
        match = re.search(tool_call_pattern, resp, re.DOTALL)
        if match:
            return resp[: match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Stop once any boxed final answer appears in the assistant response.
    if "\\boxed{" in resp:
        boxed_spans = _find_boxed_spans(resp)
        if boxed_spans:
            return resp[: boxed_spans[-1][1]]

    return resp


async def execute_predictions(
    prediction: str, tool_registry: ToolRegistry
) -> tuple[str, bool, dict[str, Any] | None]:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            tool_message = {"role": "tool", "content": str(result)}
            next_obs = _render_tool_message(tool_message, prediction)
            done = False
        else:
            tool_message = {"role": "tool", "content": "Error: No Python code found"}
            next_obs = _render_tool_message(tool_message, prediction)
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
        tool_message = None
    else:
        tool_message = {
            "role": "tool",
            "content": (
                "The previous action is invalid. "
                "If executing code, you should return a JSON object inside <tool_call></tool_call>. "
                "If giving the final answer, you should use the format 'Answer: \\boxed{answer}'. PLease try again."
            ),
        }
        next_obs = _render_tool_message(tool_message, prediction)
        done = False

    return next_obs, done, tool_message


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."
    state = GenerateState(args)
    tool_registry = ToolRegistry()
    sandbox_session_id = tool_registry.session_id
    sandbox_backend = tool_registry.backend
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()

    # Build the full system prompt text once for recorded_messages.
    full_system_prompt = DEFAULT_SYSTEM_PROMPT
    if tool_specs:
        tool_lines = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tool_specs)
        full_system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_SYSTEM_PROMPT.replace('__TOOLS__', tool_lines)}"

    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # Build recorded_messages with the full system prompt (including tool descriptions)
    # so that payload_text faithfully reflects what the model actually sees.
    recorded_messages = _build_initial_recorded_messages(sample.prompt, system_prompt=full_system_prompt)
    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
    _task_context_len = _max_context_len_override()
    if _task_context_len > 0:
        max_context_length = min(max_context_length, _task_context_len)

    max_context_length = max_context_length - 16  # Leave some buffer tokens for safety

    # Keep the sample structurally valid even if a later rollout turn aborts.
    #
    # These also have to be *reset*, not just initialised: an ABORTED group goes back
    # to data_buffer and is regenerated through this same function, so the Sample
    # arriving here can still carry the previous attempt's rollout state. The token
    # lists below start empty, so leaving stale rollout_log_probs in place would make
    # the `if sample.rollout_log_probs is None` guard further down append this
    # attempt's log probs onto the last attempt's -- the length assert then fires, or
    # the final length-fixup trims the fresh tail and keeps the stale head.
    sample.tokens = list(prompt_tokens_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend
    sample.messages = list(recorded_messages)

    print(f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} sample_start")

    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    last_finish_reason = None

    async def _run_turns():
        nonlocal response_token_ids, loss_masks, tool_call_count, last_finish_reason
        for turn in range(TOOL_CONFIGS["max_turns"]):
            # Append-only, as in slime's examples/retool: the request carries exactly
            # prompt_tokens_ids plus everything accumulated so far, which is also
            # exactly the sequence stored for training.
            #
            # An earlier version re-rendered the whole conversation through the chat template every
            # turn and re-tokenized it. That makes the sampled sequence and the trained
            # sequence two different token streams: the chat template re-serialises each
            # assistant turn (json.dumps of the parsed tool call need not reproduce what
            # the model emitted byte for byte), and re-tokenizing a concatenation can
            # merge differently at the seams. Under --use-rollout-logprobs that is no
            # longer cosmetic -- the engine's log probs are conditioned on what was
            # sent, so they must describe the sequence being trained on, or the
            # importance ratio is wrong for reasons unrelated to policy staleness.
            #
            # It also drops the per-turn re-tokenization of the whole prefix, which was
            # quadratic in trajectory length.
            current_token_ids = prompt_tokens_ids + response_token_ids
            total_length = len(current_token_ids)
            if total_length >= max_context_length:
                sample.status = Sample.Status.TRUNCATED
                break
            remaining_context = max_context_length - total_length
            current_sampling_params = dict(sampling_params)
            current_sampling_params["stop"] = _merged_stop_strings(current_sampling_params.get("stop"))
            current_sampling_params["no_stop_trim"] = True
            current_sampling_params["max_new_tokens"] = min(
                current_sampling_params["max_new_tokens"],
                remaining_context,
            )
            if current_sampling_params["max_new_tokens"] <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            payload = {
                "input_ids": current_token_ids,
                "sampling_params": current_sampling_params,
                "return_logprob": True,  # Request log probabilities for training
            }

            print(
                f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} turn={turn} tool_calls={tool_call_count}"
            )

            output = await post(url, payload)

            last_finish_reason = output["meta_info"]["finish_reason"]["type"]

            # Handle abort
            if last_finish_reason == "abort":
                sample.status = Sample.Status.ABORTED
                break

            if "output_token_logprobs" in output["meta_info"]:
                # The engine's tokens are the source of truth; text is decoded from
                # them, never the other way round. This mirrors slime's own retool
                # path (examples/retool/generate_with_retool.py).
                #
                # An earlier version ran postprocess_responses here as well and then tried to map
                # the resulting *character* offset back onto a token prefix. When the
                # cut landed inside a token no prefix matched, so it re-tokenized the
                # trimmed text -- and then the log probs described the engine's
                # tokenization while the token ids came from a fresh one. Dropping the
                # text-level step removes that whole failure mode.
                #
                # The decode keeps <|im_end|>: _render_tool_message reads it to tell a
                # turn the model closed from one that stopped at </tool_call>.
                cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                cur_response = state.tokenizer.decode(cur_response_token_ids, skip_special_tokens=False)
                if sample.rollout_log_probs is None:
                    sample.rollout_log_probs = []
                sample.rollout_log_probs += cur_log_probs

            else:
                # The engine returned text but no per-token log probs, so this turn's
                # tokens have nothing to train the importance ratio against. Neither
                # local repair is good: fabricating zeros means log p = 0, i.e. p = 1,
                # which pins the ratio to the clip boundary under
                # --use-rollout-logprobs, and masking the turn out silently shortens
                # the trajectory mid-stream while leaving the tokens in the prompt of
                # every later turn.
                #
                # slime's own retool example aborts instead, which is what the async
                # paths already expect: the rollout manager returns the whole group to
                # data_buffer and regenerates it against fresh weights. Keep the turns
                # collected so far -- they have real log probs -- and drop this one.
                print(
                    f"[sandbox] session={sandbox_session_id} turn={turn} "
                    "no output_token_logprobs in meta_info -> ABORTED (group will be retried)",
                    flush=True,
                )
                sample.status = Sample.Status.ABORTED
                return

            response_token_ids += cur_response_token_ids
            loss_masks += [1] * len(cur_response_token_ids)

            assistant_message = _build_assistant_message(cur_response)
            if assistant_message is not None:
                recorded_messages.append(assistant_message)
                sample.messages = list(recorded_messages)

            # Check length limit
            if last_finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            # Skip tool execution entirely when tools are disabled. This makes
            # max_tool_calls=0 a true "no tools" mode (single-shot completion).
            if TOOL_CONFIGS["max_tool_calls"] <= 0:
                break
            if TOOL_CONFIGS["max_turns"] == 1:
                break

            next_obs, done, tool_message = await execute_predictions(cur_response, tool_registry)
            if done:
                break

            assert next_obs != "", "Next observation should not be empty."
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            # response_token_ids already includes this turn's generation.
            if len(prompt_tokens_ids) + len(response_token_ids) + len(obs_tokens_ids) > max_context_length:
                # The observation does not fit. End the trajectory after the model's
                # turn: a cut observation is one the model could never answer, and
                # its tokens would only ever be masked out.
                sample.status = Sample.Status.TRUNCATED
                break
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)
            # Count tool calls (when we get interpreter output, it means a tool
            # was called)
            if "<tool_response>" in next_obs:
                tool_call_count += 1

            if tool_message is not None:
                recorded_messages.append(tool_message)
                sample.messages = list(recorded_messages)

            # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
            # Check if maximum tool call count reached
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

                assert len(response_token_ids) == len(
                    sample.rollout_log_probs
                ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

            if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
                break

    # Wall-clock cap on the entire multi-turn rollout for this sample.
    # On timeout, keep whatever partial response was decoded and mark the
    # sample TRUNCATED so the rest of the batch can proceed instead of
    # hanging forever on a pathological sample / stuck tool subprocess.
    try:
        await asyncio.wait_for(_run_turns(), timeout=_SAMPLE_TIMEOUT)
    except asyncio.TimeoutError:
        print(
            f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} "
            f"sample TIMEOUT after {_SAMPLE_TIMEOUT:.0f}s "
            f"(turns_done<= {TOOL_CONFIGS['max_turns']}, tool_calls={tool_call_count})",
            flush=True,
        )
        # Whether the sample was stuck in /generate or in the tool sandbox has very
        # different fixes, and the timeout alone does not say which. Census the still
        # pending coroutines, as slime's examples/coding_agent_rl does. Never raise.
        try:
            from collections import Counter

            names = []
            for task in asyncio.all_tasks():
                if task.done():
                    continue
                coro = task.get_coro()
                names.append(getattr(coro, "__qualname__", None) or repr(coro))
            print(
                f"[sandbox] session={sandbox_session_id} pending coroutines: "
                f"{Counter(names).most_common(5)}",
                flush=True,
            )
        except Exception:
            pass
        sample.status = Sample.Status.TRUNCATED

    # Set sample attributes
    (
        final_tokens,
        final_response,
        final_response_token_ids,
        final_loss_masks,
        final_rollout_log_probs,
        final_was_clipped,
    ) = _apply_final_token_clip(
        state.tokenizer,
        prompt_tokens_ids,
        response_token_ids,
        loss_masks,
        sample.rollout_log_probs,
        max_context_length,
    )

    sample.tokens = final_tokens
    sample.response_length = len(final_response_token_ids)
    sample.response = final_response
    sample.loss_mask = final_loss_masks
    sample.rollout_log_probs = final_rollout_log_probs
    # If the sample was aborted before any output_token_logprobs were
    # returned (e.g. SGLang abort on the very first turn), rollout_log_probs
    # may still be None even though response_length > 0 (from prior turns).
    # Slime's _convert_samples_to_train_data only checks samples[0] for
    # None-ness, so a single None entry later in the batch becomes
    # `len(None)` in slice_log_prob_with_cp. Substitute zeros to keep the
    # batch trainable; loss_mask already zeros out unobserved tokens where
    # appropriate.
    # Whenever a log prob is fabricated here, the matching loss_mask entry is cleared.
    # Tool observations already pair their dummy 0.0 with loss_mask=0 above, but these
    # two branches cover *generated* tokens, whose mask is 1. That is harmless while
    # the importance ratio is built from recomputed actor log probs, and wrong the
    # moment --use-rollout-logprobs is on: log p = 0 means p = 1, so the ratio for
    # those tokens collapses onto the clip boundary. sum_of_sample_mean drops
    # loss_mask=0 tokens from both numerator and denominator, so masking is the fix.
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

    # Guarantee response_length >= 1. A zero-length response (e.g. SGLang
    # aborted before any token was generated, or max_turns=1 with the
    # context budget already exhausted) propagates an empty entry into
    # slime's dynamic-batch partitioner and ultimately produces an empty
    # micro-batch -> `torch.cat([])` in get_batch. Pad with a single dummy
    # token whose loss_mask=0 so gradients/KL are unaffected but every
    # micro-batch contains at least one token tensor.
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
        # An empty response is by definition not a successful completion.
        if sample.status not in {Sample.Status.ABORTED, Sample.Status.TRUNCATED}:
            sample.status = Sample.Status.TRUNCATED

    if final_was_clipped:
        sample.status = Sample.Status.TRUNCATED

    # Store payload information for wandb logging
    sample.payload_text = _render_recorded_messages(recorded_messages)
    sample.payload_has_system = "<|im_start|>system" in sample.payload_text
    sample.payload_has_tools = "# Tools" in sample.payload_text
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Expose round_number in metadata so slime logs it under multi_turn_metric/
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["round_number"] = tool_call_count

    sample.metadata["opd_rev_kl_weight"] = 1.0

    # Set status
    if sample.status not in {Sample.Status.TRUNCATED, Sample.Status.ABORTED}:
        if last_finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
        elif last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
        elif last_finish_reason == "stop":
            sample.status = Sample.Status.COMPLETED

    await tool_registry.close()
    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = _prompt_to_text(sample.prompt) + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # Encourage the model to call tools. Off by default -- the plain +1/-1 outcome
    # reward is what GRPO and OPD both see, so that a comparison between
    # them is not distorted by shaping. Set RETOOL_TOOL_CALL_SHAPING=1 to restore it.
    #
    # Two notes if you do turn it on. The arithmetic comes from ReTool's verl recipe
    # (recipe/retool/retool.py), where num_turns is user_turns + assistant_turns + 1,
    # i.e. 2k + 2 for k tool calls -- so (num_turns - 2) / 2 recovers k and the bonus
    # is 0.1 per tool call. Here num_turns is already the tool call count, so the same
    # expression yields 0.05k - 0.1: half the intended slope, shifted down. ReTool also
    # capped at 0 rather than -0.6.
    #
    # The shaping only lifts *failed* trajectories, which is precisely what stops an
    # all-wrong GRPO group from having zero std (and therefore zero advantage).
    # OPD ignores the task reward entirely,
    # so leaving this on props up GRPO alone.
    if _tool_call_shaping() and result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
