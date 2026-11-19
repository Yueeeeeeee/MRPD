# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import json
import re
from typing import Any

# Import tool sandbox functionality
from eval_tool_sandbox import SEMAPHORE, TOOL_CONFIGS, ToolRegistry

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

