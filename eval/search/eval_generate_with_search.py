def extract_boxed_answer(prediction: str) -> str | None:
    """Extracts the answer from any occurrence of \\boxed{...} (LaTeX style)."""
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, prediction)
    if matches:
        return matches[-1].strip()
    return None
# Adapted from search-r1/generate_with_search.py.
# This is the eval-only variant: we strip the slime-specific `generate()` /
# `reward_func()` entry points so the file can be imported by `eval.py`
# without needing the slime training stack.
import asyncio
import json
import re
from typing import Any

# Configuration for Search-R1 eval. Override SEARCH_R1_CONFIGS["search_backend"],
# SEARCH_R1_CONFIGS["local"]["search_url"], etc. before calling search().

# System prompt and tool definition (retool style)
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

SEARCH_R1_CONFIGS = {
    "max_turns": 8,
    "topk": 3,
    "search_concurrency": 256,
    "search_backend": "local",
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # Comma-separate multiple local retriever URLs.
        "proxy": None,
    },
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])



def _passages2string(retrieval_result):
    """Convert retrieval results to a formatted reference string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference



async def search(query: str) -> str:
    """Query the local retriever and format its passages."""
    from local_search_server import local_search
    local_config = SEARCH_R1_CONFIGS["local"]
    result = await local_search(
        local_config["search_url"],
        query,
        SEARCH_R1_CONFIGS["topk"],
        proxy=local_config["proxy"],
    )
    return _passages2string(result)



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


def _extract_first_tool_call(prediction: str) -> dict[str, Any] | None:
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


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    """Run the action implied by `prediction`. Return (next_observation, done)."""
    action, content = postprocess_predictions(prediction)
    # A turn that stopped at </tool_call> is still open, so the observation closes
    # it. A turn the model ended with its own <|im_end|> is closed already, and
    # closing it again would put two <|im_end|> in a row.
    close = "" if prediction.endswith("<|im_end|>") else "<|im_end|>"
    if action == "search":
        try:
            async with SEMAPHORE:
                search_results = await search(content)
            tool_response_content = search_results.strip()
        except Exception as e:
            tool_response_content = f"[ERROR] {type(e).__name__}: {e}"
        # Return as <tool_response> in user message, retool style
        next_obs = (
            f"{close}\n"
            "<|im_start|>user\n<tool_response>\n"
            f"{tool_response_content}\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
        return next_obs, False
    if action == "answer" or action == "boxed_answer":
        return "", True
    # Move the invalid action message into a user message, matching retool style
    next_obs = (
        f"{close}\n"
        "<|im_start|>user\n"
        "Your previous action is invalid. "
        "If you want to use a tool, you should return a <tool_call>...</tool_call> block. "
        "If you want to give the final answer, you should put the answer in \\boxed{}. "
        "Please try again.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    return next_obs, False
# ...existing code...

_WARNED_SYSTEM_PROMPTS: set[str] = set()


def _note_dropped_system_prompt(content: str, rendered: str) -> None:
    """Warn once per distinct dataset system prompt that is being ignored."""
    text = (content or "").strip()
    if not text or text == (rendered or "").strip():
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


# Conversation rendering (retool style)
def format_conversation_with_tools(
    prompt: str | list[dict[str, Any]],
    system_prompt: str = None,
    messages: list[dict[str, Any]] = None,
) -> str:
    """Format conversation using a static system prompt and explicit message rendering."""
    messages_to_render = []
    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = DEFAULT_SYSTEM_PROMPT
    system_content = f"{system_content}\n\n{TOOL_SYSTEM_PROMPT}"
    messages_to_render.append({"role": "system", "content": system_content})
    # Add user prompt. A system message arriving in the data is dropped rather
    # than rendered as a second system block: the trainer
    # (generate_with_search._normalize_prompt_messages) drops it too, and the two
    # sides have to render the same prompt or the comparison is not one. The
    # released prompt sets carry only user turns, so this never fires on them --
    # it is here for datasets you build yourself.
    if isinstance(prompt, str):
        messages_to_render.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "system":
                _note_dropped_system_prompt(str(message.get("content", "")), system_content)
                continue
            messages_to_render.append(message)
    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)
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
                "<|im_start|>user\n<tool_response>\n"
                f"{message['content']}\n"
                "</tool_response><|im_end|>"
            )
    rendered_parts.append("<|im_start|>assistant\n")
    return "\n".join(rendered_parts)
