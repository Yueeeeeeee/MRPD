"""Text-to-SQL tool environment: the part shared by training (generate_with_sql.py)
and evaluation (eval/sql/eval.py), with no slime imports so both can use it.

The model sees the schema of one SQLite database and a question, may run queries
against the database with the `sql_executor` tool (ReTool's tool-call format, so the
same chat rendering and stop strings as the math environment), and ends with the final
query in a ```sql code block (the cold-start model's own habit; <answer></answer> tags
around it are accepted too). A turn without a tool call is the final answer, as a turn
without a tool call is in the math environment (no query in it scores -1; the model is
not asked again). The reward executes the final query and the gold query
and compares the result sets: +1 if equal, -1 otherwise (no answer, an error, a
different result). Ordered comparison when the gold query has ORDER BY, multiset
comparison otherwise -- Spider's own execution-accuracy rule.

Databases are found at SQL_DB_ROOT/<db_id>/<db_id>.sqlite (scripts/data/prepare_sql.py
lays them out that way) and are opened read-only.

Environment knobs:
    SQL_DB_ROOT            directory holding <db_id>/<db_id>.sqlite    (required)
    SQL_MAX_TOOL_CALLS     queries the model may run per sample        (6)
    SQL_MAX_TURNS          generation turns per sample                  (8)
    SQL_MAX_TOKENS_PER_TURN  new tokens per generation turn             (4096, as search)
    SQL_EXEC_TIMEOUT       seconds per tool query                       (5)
    SQL_SCORE_TIMEOUT      seconds per query when scoring               (30)
    SQL_MAX_RESULT_ROWS    rows of a result shown to the model          (20)
    SQL_MAX_RESULT_CHARS   characters of a result shown to the model    (4000)
    SQL_TOOL_CONCURRENCY   concurrent tool executions                   (128)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Any

TOOL_CONFIGS = {
    "db_root": os.environ.get("SQL_DB_ROOT", ""),
    "max_tool_calls": int(os.environ.get("SQL_MAX_TOOL_CALLS", "6")),
    "max_turns": int(os.environ.get("SQL_MAX_TURNS", "8")),
    "max_tokens_per_turn": int(os.environ.get("SQL_MAX_TOKENS_PER_TURN", "4096")),
    "exec_timeout": float(os.environ.get("SQL_EXEC_TIMEOUT", "5")),
    "score_timeout": float(os.environ.get("SQL_SCORE_TIMEOUT", "30")),
    "max_result_rows": int(os.environ.get("SQL_MAX_RESULT_ROWS", "20")),
    "max_result_chars": int(os.environ.get("SQL_MAX_RESULT_CHARS", "4000")),
    "concurrency": int(os.environ.get("SQL_TOOL_CONCURRENCY", "128")),
}

TOOL_NAME = "sql_executor"

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Execute one SQLite query against the database and return the first rows of "
                "its result, or the error message. Use it to inspect tables and to check a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "The SQLite query to execute"}},
                "required": ["sql"],
            },
        },
    }
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a data science expert. You are given the schema of a SQLite database and a "
    "question about its data, and your task is to write one SQLite query that answers the "
    f"question. You may run queries against the database with the {TOOL_NAME} tool, at most "
    f"{TOOL_CONFIGS['max_tool_calls']} times, to inspect the tables and to check your query. "
    "Think step by step. When you are confident, give the final query as a single SQLite "
    "statement in a ```sql code block and stop: a turn with no tool call is your final answer."
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

INVALID_ACTION_MESSAGE = (
    "The tool call could not be run. To run a query, return a JSON object with the name "
    f'"{TOOL_NAME}" and an "sql" argument inside <tool_call></tool_call>. To give the final '
    "query, write it in a ```sql code block and stop."
)

# The one stop that hands control to the environment: after a tool call, so that the
# observation can be appended. The answer turn ends with the model's own <|im_end|>,
# as the math environment's boxed answer does, so the model learns to stop after
# </answer> rather than being cut there.
STOP_STRINGS = ["</tool_call>"]

_SEMAPHORE: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["concurrency"])
    return _SEMAPHORE


# --------------------------------------------------------------------------------------
# Databases
# --------------------------------------------------------------------------------------


def resolve_db_path(db_id: str, db_root: str | None = None) -> str:
    root = db_root or TOOL_CONFIGS["db_root"]
    if not root:
        raise RuntimeError("SQL_DB_ROOT is not set; it must point at the directory holding <db_id>/<db_id>.sqlite")
    path = os.path.join(root, db_id, f"{db_id}.sqlite")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no database for db_id={db_id!r} at {path}")
    return path


def _connect_readonly(path: str, timeout: float) -> sqlite3.Connection:
    # mode=ro: a query that tries to write fails with "attempt to write a readonly
    # database" instead of changing the file that every other sample reads. The
    # file URI needs percent-encoding (as_uri does it), and resolve() follows the
    # symlinks prepare_sql.py lays down.
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=timeout, check_same_thread=False)


def run_query(
    db_id: str, sql: str, timeout: float, max_rows: int | None = None, db_root: str | None = None
) -> tuple[list[tuple] | None, list[str], str]:
    """Run one query. Returns (rows, column names, error); rows is None on an error.

    With max_rows, at most max_rows + 1 rows are fetched, so the caller can tell
    "more rows exist" from the count without reading a huge result.
    """
    try:
        path = resolve_db_path(db_id, db_root)
    except (RuntimeError, FileNotFoundError) as exc:
        return None, [], f"Error: {exc}"
    conn = _connect_readonly(path, timeout)
    timer = threading.Timer(timeout, conn.interrupt)
    try:
        timer.start()
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows + 1) if max_rows is not None else cursor.fetchall()
        return [tuple(r) for r in rows], columns, ""
    except sqlite3.OperationalError as exc:
        message = str(exc)
        if "interrupted" in message:
            return None, [], f"Error: query timed out after {timeout:g} seconds"
        return None, [], f"Error: {message}"
    except (sqlite3.Error, ValueError, OverflowError) as exc:
        return None, [], f"Error: {exc}"
    finally:
        timer.cancel()
        conn.close()


def _format_value(value: Any, max_chars: int = 200) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        text = f"<{len(value)} bytes>"
    elif isinstance(value, float):
        text = repr(value)
    else:
        text = str(value)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def format_result(rows: list[tuple], columns: list[str], max_rows: int, max_chars: int) -> str:
    """The observation the model reads: a header line, then up to max_rows rows."""
    if not columns:
        return "Query executed; it returned no columns."
    lines = [" | ".join(columns)]
    for row in rows[:max_rows]:
        lines.append(" | ".join(_format_value(v) for v in row))
    if not rows:
        lines.append("(0 rows)")
    elif len(rows) > max_rows:
        lines.append(f"... (more rows; showing the first {max_rows})")
    else:
        lines.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (output truncated to {max_chars} characters)"
    return text


async def execute_sql(db_id: str, sql: str) -> str:
    """The tool: run the query read-only with the tool timeout and format the result."""
    sql = sql.strip()
    if not sql:
        return "Error: empty query"
    async with _semaphore():
        rows, columns, error = await asyncio.to_thread(
            run_query, db_id, sql, TOOL_CONFIGS["exec_timeout"], TOOL_CONFIGS["max_result_rows"]
        )
    if rows is None:
        return error
    return format_result(rows, columns, TOOL_CONFIGS["max_result_rows"], TOOL_CONFIGS["max_result_chars"])


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def _normalize_row(row: tuple) -> tuple:
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


def _has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, re.IGNORECASE) is not None


def results_match(
    db_id: str, pred_sql: str, gold_sql: str, timeout: float | None = None, db_root: str | None = None
) -> tuple[bool, str]:
    """Execute both queries and compare their results. Returns (match, error text)."""
    timeout = TOOL_CONFIGS["score_timeout"] if timeout is None else timeout
    gold_rows, _, gold_error = run_query(db_id, gold_sql, timeout, db_root=db_root)
    if gold_rows is None:
        return False, f"gold query failed: {gold_error}"
    pred_rows, _, pred_error = run_query(db_id, pred_sql, timeout, db_root=db_root)
    if pred_rows is None:
        return False, pred_error
    gold_norm = [_normalize_row(r) for r in gold_rows]
    pred_norm = [_normalize_row(r) for r in pred_rows]
    if _has_order_by(gold_sql):
        return gold_norm == pred_norm, ""
    return Counter(gold_norm) == Counter(pred_norm), ""


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# A fenced code block: the rest of the opening line ("sql", "sql code", "") and the body.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_SQL_TAG_RE = re.compile(r"^\s*(?:sqlite|sql)\b", re.IGNORECASE)
_SQL_START_RE = re.compile(r"^\s*(?:select|with)\b", re.IGNORECASE)


def extract_fenced_sql(text: str) -> str:
    """The last SQL code block of `text`: tagged sql/sqlite (whatever else is on the
    opening line), or starting with SELECT/WITH. '' if there is none; blocks of other
    kinds (```json, ```python, expected output) are never the answer."""
    sql = [body for info, body in _FENCE_RE.findall(text) if _SQL_TAG_RE.match(info) or _SQL_START_RE.match(body)]
    return sql[-1].strip().rstrip(";").strip() if sql else ""


def _clean_sql(body: str) -> str:
    """An <answer> body: the query, unwrapped from a code block if it is in one."""
    return extract_fenced_sql(body) or body.strip().rstrip(";").strip()


def extract_final_sql(text: str) -> str:
    """The final query: the last <answer> block if the model used the tags, else the
    last ```sql block of the final assistant turn (after the last tool response);
    '' if neither. A bare block in a turn without a tool call is the answer."""
    answers = _ANSWER_RE.findall(text)
    if answers:
        return _clean_sql(answers[-1])
    return extract_fenced_sql(text.rsplit("</tool_response>", 1)[-1])


def score_response(response: str, db_id: str, gold_sql: str, db_root: str | None = None) -> dict[str, Any]:
    """+1 when the final query's result equals the gold query's, -1 otherwise."""
    pred_sql = extract_final_sql(response)
    if not pred_sql:
        return {"pred": "", "score": -1.0, "acc": False, "error": "no final sql block"}
    match, error = results_match(db_id, pred_sql, gold_sql, db_root=db_root)
    return {"pred": pred_sql, "score": 1.0 if match else -1.0, "acc": bool(match), "error": error}


# --------------------------------------------------------------------------------------
# Schema text for the prompt (used by scripts/data/prepare_sql.py)
# --------------------------------------------------------------------------------------


def build_schema_text(db_path: str, sample_rows: int = 3, max_value_chars: int = 40, max_chars: int = 12000) -> str:
    """CREATE TABLE statements plus the first rows of every table, as comments.

    Sample rows are dropped when the text would exceed max_chars; the statements
    are kept whole, truncated with a note only if they alone exceed it.
    """
    conn = _connect_readonly(db_path, 30.0)
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        creates, samples = [], []
        for name, create in tables:
            if not create:
                continue
            creates.append(re.sub(r"\s+", " ", create).strip().rstrip(";") + ";")
            try:
                cursor = conn.execute(f'SELECT * FROM "{name}" LIMIT {int(sample_rows)}')
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
            except sqlite3.Error:
                rows, columns = [], []
            if rows:
                lines = [f"-- first {len(rows)} rows of {name}:", "-- " + " | ".join(columns)]
                lines += ["-- " + " | ".join(_format_value(v, max_value_chars) for v in row) for row in rows]
                samples.append("\n".join(lines))
            else:
                samples.append("")
    finally:
        conn.close()

    def _assemble(with_samples: bool) -> str:
        parts = []
        for create, sample in zip(creates, samples, strict=True):
            parts.append(create + ("\n" + sample if with_samples and sample else ""))
        return "\n\n".join(parts)

    text = _assemble(True)
    if len(text) > max_chars:
        text = _assemble(False)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n-- (schema truncated)"
    return text


def build_user_prompt(schema_text: str, question: str, evidence: str | None = None) -> str:
    prompt = f"Database schema (SQLite):\n{schema_text}\n\nQuestion: {question.strip()}"
    if evidence and evidence.strip():
        prompt += f"\nHint: {evidence.strip()}"
    return prompt


# --------------------------------------------------------------------------------------
# Chat rendering and action parsing (ReTool's conventions, kept byte-compatible with
# generate_with_retool.py so that training and eval build identical token streams)
# --------------------------------------------------------------------------------------


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
    pattern = re.compile(r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_end\|>)<\|im_end\|>", re.DOTALL)
    messages: list[dict[str, str]] = []
    for role, content in pattern.findall(prompt):
        text = content.strip()
        if not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def normalize_prompt_messages(prompt: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """The user-side messages of a prompt: a rendered transcript (what slime's
    --apply-chat-template hands over), a message list, or a plain string."""
    if prompt is None:
        return []
    if isinstance(prompt, str) and "<|im_start|>" in prompt:
        parsed = _parse_chat_transcript(prompt)
        if parsed:
            # The data file carries no system prompt of its own; a template default
            # (Qwen adds none for Instruct-2507) would be replaced by ours anyway.
            return [m for m in parsed if m["role"] != "system"]
    if isinstance(prompt, list):
        normalized = []
        for message in prompt:
            if not isinstance(message, dict):
                content = str(message)
                if content.strip():
                    normalized.append({"role": "user", "content": content})
                continue
            content = _stringify_message_content(message.get("content", ""))
            if content.strip():
                normalized.append({"role": message.get("role", "user"), "content": content})
        return normalized
    content = str(prompt).strip()  # as _parse_chat_transcript strips: eval's raw prompt renders as training's templated one
    return [{"role": "user", "content": content}] if content else []


def full_system_prompt() -> str:
    tool_lines = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in TOOL_SPECS)
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_SYSTEM_PROMPT.replace('__TOOLS__', tool_lines)}"


def _render_messages(messages: list[dict[str, Any]]) -> list[str]:
    rendered = []
    for message in messages:
        role = message["role"]
        if role in ("system", "user"):
            rendered.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>")
        elif role == "assistant":
            text = ["<|im_start|>assistant", message.get("content", "")]
            for tool_call in message.get("tool_calls", []):
                text += ["<tool_call>", json.dumps(tool_call["function"], ensure_ascii=False), "</tool_call>"]
            text.append("<|im_end|>")
            rendered.append("\n".join(part for part in text if part != ""))
        elif role == "tool":
            rendered.append(f"<|im_start|>user\n<tool_response>\n{message['content']}\n</tool_response><|im_end|>")
    return rendered


def format_conversation_with_tools(prompt: str | list[dict[str, Any]]) -> str:
    """The text the first /generate call is tokenized from: our system prompt (with
    the tool signatures), the prompt's own messages, and an open assistant turn."""
    messages = [{"role": "system", "content": full_system_prompt()}] + normalize_prompt_messages(prompt)
    return "\n".join(_render_messages(messages) + ["<|im_start|>assistant\n"])


def render_recorded_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_render_messages(messages))


def render_tool_message(content: str, turn: str) -> str:
    """The observation as the user turn after `turn`, the model's text. A turn that
    stopped at a stop string is still open and gets closed here; one the model
    ended with its own <|im_end|> is not closed twice."""
    close = "" if turn.endswith("<|im_end|>") else "<|im_end|>"
    return f"{close}\n<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_first_tool_call(prediction: str) -> dict[str, Any] | None:
    match = _TOOL_CALL_RE.search(prediction)
    if not match:
        return None
    try:
        # strict=False accepts raw newlines inside the query string and a pretty-printed
        # object alike; ReTool's trick of escaping the newlines breaks the latter.
        data = json.loads(match.group(1), strict=False)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def postprocess_responses(response: str) -> str:
    """The recorded form of a turn: cut after the first complete tool call, or after
    the first closed answer (anything the model writes past it is ignored)."""
    cut = len(response)
    match = _TOOL_CALL_RE.search(response)
    if match:
        cut = min(cut, match.end())
    end = response.find("</answer>")
    if end != -1:
        cut = min(cut, end + len("</answer>"))
    return response[:cut]


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """('answer', sql) | ('sql', query) | (None, ''), for one turn of model text:
    a closed <answer> is the answer; else a sql_executor call is a query; else a turn
    that attempts a tool call is invalid (it does not parse, names another tool, has no
    query); else the turn is the answer, with its last SQL code block as the query
    ('' when there is none, which scores -1)."""
    if "</answer>" in prediction:
        return "answer", extract_final_sql(prediction)
    if "<tool_call>" not in prediction:
        return "answer", extract_fenced_sql(prediction)
    tool_call = extract_first_tool_call(prediction)
    if tool_call and tool_call.get("name") == TOOL_NAME:
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
        sql = arguments.get("sql", "") if isinstance(arguments, dict) else ""
        if isinstance(sql, str) and sql.strip():
            return "sql", sql.strip()
    return None, ""


def build_assistant_message(prediction: str) -> dict[str, Any] | None:
    """A structured assistant message for the recorded transcript."""
    sanitized = postprocess_responses(prediction)
    tool_call = extract_first_tool_call(sanitized)
    if tool_call is None:
        content = sanitized.rstrip()
        return {"role": "assistant", "content": content} if content else None
    match = _TOOL_CALL_RE.search(sanitized)
    content = sanitized[: match.start()].rstrip() if match else sanitized.rstrip()
    return {"role": "assistant", "content": content, "tool_calls": [{"type": "function", "function": tool_call}]}


async def execute_predictions(prediction: str, db_id: str) -> tuple[str, bool, dict[str, Any] | None]:
    """Act on the model's turn: run the query, accept the answer, or complain about a
    tool call that cannot be run. Returns (next observation text, done, tool message)."""
    action, content = postprocess_predictions(prediction)
    if action == "answer":
        return "", True, None
    result = await execute_sql(db_id, content) if action == "sql" else INVALID_ACTION_MESSAGE
    tool_message = {"role": "tool", "content": result}
    return render_tool_message(result, prediction), False, tool_message


def merged_stop_strings(existing_stop) -> list[str]:
    stops: list[str] = []
    if isinstance(existing_stop, str):
        stops.append(existing_stop)
    elif isinstance(existing_stop, list):
        stops.extend(str(item) for item in existing_stop)
    for stop in STOP_STRINGS:
        if stop not in stops:
            stops.append(stop)
    return stops
