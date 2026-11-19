"""Evaluate an HF checkpoint on a text-to-SQL set against a running SGLang server,
building each trajectory the way training builds it (train/src/generate_with_sql.py:
same system prompt, tool format, stop strings and observation rendering, all from
train/src/sql_tools.py) and scoring the final query by execution match.

scripts/eval/sql.sh starts the server and calls this per dataset. The dataset is a
jsonl from scripts/data/prepare_sql.py (prompt / label / metadata.db_id); SQL_DB_ROOT
must point at its databases/ directory. The summary has the shape of the math eval's
(accuracy, timeout_stats, num_samples) plus "env": "sql", so collect_results.py
tabulates it next to the others.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "train" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import sql_tools  # noqa: E402  (slime-free)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="traces per question")
    parser.add_argument("--model-path", required=True, help="HF model path (tokenizer; logged in the summary)")
    parser.add_argument("--tokenizer-path", default=None, help="defaults to --model-path")
    parser.add_argument("--dataset", required=True, help="jsonl written by scripts/data/prepare_sql.py")
    parser.add_argument("--dataset-name", default=None, help="label in the summary (default: the file's stem)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-wait-timeout", type=int, default=60)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="max new tokens per turn")
    parser.add_argument(
        "--max-context-len", type=int, default=9216,
        help="prompt + response budget per trace, as --rollout-max-context-len in training",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--output", default=None, help="per-question results jsonl")
    parser.add_argument("--summary-output", default=None, help="summary json")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--sample-timeout", type=int, default=600, help="seconds per trace, all turns")
    parser.add_argument("--print-turns", action="store_true")
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="re-run only the timed-out traces of an existing --output (math eval's semantics)",
    )
    args = parser.parse_args()
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    if args.num_samples < 1:
        parser.error("-n/--num-samples must be at least 1")
    return args


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The jsonl as written by prepare_sql.py: one {prompt, label, metadata} per line."""
    rows: list[dict[str, Any]] = []
    with open(args.dataset, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if args.limit is not None and len(rows) >= args.limit:
                break
    return rows


CONNECT_TIMEOUT_S = 30
CONNECT_ATTEMPTS = 5


async def _open_connection(host: str, port: int):
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=CONNECT_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError) as exc:
            if attempt == CONNECT_ATTEMPTS:
                raise RuntimeError(f"HTTP connect to {host}:{port} failed {CONNECT_ATTEMPTS} times ({exc})") from exc
            await asyncio.sleep(2 ** (attempt - 1))


async def _post_json_async(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    host, port, path = parsed.hostname or "127.0.0.1", parsed.port or 80, parsed.path or "/"
    body = json.dumps(payload).encode("utf-8")
    http_request = (
        f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode() + body
    reader, writer = await _open_connection(host, port)
    try:
        try:
            writer.write(http_request)
            await asyncio.wait_for(writer.drain(), timeout=10)
            response_bytes = await asyncio.wait_for(reader.read(-1), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"HTTP request to {host}:{port}{path} timed out after {timeout}s") from exc
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=5)
        except Exception:
            pass
    sep = response_bytes.find(b"\r\n\r\n")
    if sep == -1:
        raise ValueError(f"Malformed HTTP response: {response_bytes[:200]!r}")
    return json.loads(response_bytes[sep + 4 :])


def _wait_for_port(host: str, port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"SGLang server on {host}:{port} not reachable after {timeout}s: {last_error}")


async def _generate_one_with_tools(args, tokenizer, prompt: Any, db_id: str, debug_tag: str = "") -> dict[str, Any]:
    """One trajectory, token-append-only, exactly as generate_with_sql.generate."""
    trace_start = time.time()
    progress = {"current_turn": -1, "current_stage": "init", "stage_started_at": trace_start}
    response_ids: list[int] = []
    response_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    tool_call_count = 0
    stopped_due_to_max_tokens = False
    timed_out = False
    try:
        max_turns = sql_tools.TOOL_CONFIGS["max_turns"]
        max_tool_calls = sql_tools.TOOL_CONFIGS["max_tool_calls"]
        prompt_ids = tokenizer(sql_tools.format_conversation_with_tools(prompt), add_special_tokens=False)["input_ids"]
        max_context_len = args.max_context_len - 16
        for turn_index in range(max_turns):
            input_ids = prompt_ids + response_ids
            turn_max_new = min(args.max_new_tokens, sql_tools.TOOL_CONFIGS["max_tokens_per_turn"], max_context_len - len(input_ids))
            if turn_max_new <= 0:
                stopped_due_to_max_tokens = True
                break
            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": turn_max_new,
                    "stop": sql_tools.merged_stop_strings(None),
                    "no_stop_trim": True,
                },
                "return_logprob": True,
            }
            progress.update(current_turn=turn_index, current_stage="gen", stage_started_at=time.time())
            output = await _post_json_async(f"http://{args.host}:{args.port}/generate", payload, args.request_timeout)
            cur_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = tokenizer.decode(cur_ids, skip_special_tokens=False)
            finish_reason = output["meta_info"]["finish_reason"]["type"]
            if args.debug_trace:
                print(f"{debug_tag} turn={turn_index} resp_tokens={len(cur_ids)} finish={finish_reason}", flush=True)
            response_ids += cur_ids
            response_parts.append(cur_response)
            turn_record: dict[str, Any] = {"turn_index": turn_index, "assistant": cur_response}
            if finish_reason == "length":
                turn_record["max_tokens_reached"] = True
                turns.append(turn_record)
                stopped_due_to_max_tokens = True
                break
            if finish_reason == "abort":
                turn_record["aborted"] = True
                turns.append(turn_record)
                break
            progress.update(current_stage="tool", stage_started_at=time.time())
            next_obs, done, tool_message = await sql_tools.execute_predictions(cur_response, db_id)
            turn_record["done"] = done
            if done:
                turns.append(turn_record)
                break
            obs_ids = tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) + len(response_ids) + len(obs_ids) > max_context_len:
                turn_record["max_tokens_reached"] = True
                turns.append(turn_record)
                stopped_due_to_max_tokens = True
                break
            response_ids += obs_ids
            response_parts.append(next_obs)
            if "<tool_response>" in next_obs:
                tool_call_count += 1
            turn_record["tool_observation"] = next_obs
            if tool_message is not None:
                turn_record["tool_message"] = tool_message.get("content", "")
            turns.append(turn_record)
            if tool_call_count >= max_tool_calls:
                break
    except asyncio.CancelledError:
        timed_out = True
        if debug_tag:
            print(
                f"{debug_tag} CANCELLED stage={progress['current_stage']} turn={progress['current_turn']} "
                f"elapsed={time.time() - trace_start:.1f}s tool_calls={tool_call_count}",
                flush=True,
            )
    return {
        "response": "".join(response_parts),
        "turns": turns,
        "tool_call_count": tool_call_count,
        "total_trace_tokens": len(response_ids),
        "stopped_due_to_max_tokens": stopped_due_to_max_tokens,
        "timed_out": timed_out,
        "timeout_stage": progress.get("current_stage") if timed_out else None,
        "timeout_turn": progress.get("current_turn") if timed_out else None,
    }


def _print_trace_turns(example_index: int, trace: dict[str, Any]) -> None:
    print(f"\n=== example {example_index} trace {trace['trace_index']} ===")
    print(f"tool_calls={trace['tool_call_count']} total_tokens={trace['total_trace_tokens']} score={trace['score']:.1f} acc={int(trace['acc'])} pred={trace['pred']!r} {trace.get('error', '')}")
    for turn in trace.get("turns", []):
        print(f"\n--- turn {turn['turn_index']} assistant ---")
        print(turn.get("assistant", ""))
        if "tool_message" in turn:
            print("\n--- tool message ---")
            print(turn["tool_message"])


def _load_resume_records(output_path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not output_path.exists():
        return records
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec.get("index"), int):
                records[rec["index"]] = rec
    return records


_TIMEOUT_RESULT = {
    "response": "", "turns": [], "tool_call_count": 0, "total_trace_tokens": 0,
    "stopped_due_to_max_tokens": False, "timed_out": True, "timeout_stage": "unknown", "timeout_turn": -1,
}


def main() -> None:
    args = parse_args()
    if not sql_tools.TOOL_CONFIGS["db_root"]:
        raise SystemExit("SQL_DB_ROOT is not set (prepare_sql.py's databases/ directory)")
    tokenizer = _load_tokenizer(args.tokenizer_path or args.model_path)
    rows = _load_rows(args)
    dataset_name = args.dataset_name or Path(args.dataset).stem
    output_path = Path(args.output) if args.output else None

    resume_records: dict[int, dict[str, Any]] = {}
    if args.resume and output_path is not None:
        resume_records = _load_resume_records(output_path)
        print(f"[resume] loaded {len(resume_records)} previous results from {output_path}", flush=True)
        if output_path.exists():
            output_path.replace(output_path.with_suffix(output_path.suffix + ".bak"))

    print(f"Connecting to SGLang server at http://{args.host}:{args.port}")
    _wait_for_port(args.host, args.port, timeout=args.server_wait_timeout)

    async def _run_eval() -> dict[str, Any]:
        num_examples = len(rows)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        output_file = output_path.open("w", encoding="utf-8") if output_path else None
        output_lock = asyncio.Lock()
        done_examples = 0
        num_correct = 0.0
        total_timeouts = 0
        examples_with_any_timeout = 0
        all_timeout_examples: list[dict[str, Any]] = []
        no_answer = 0
        total_traces = 0

        async def _safe_generate(ex_index: int, sample_idx: int, prompt: str, db_id: str) -> dict[str, Any]:
            async with semaphore:
                tag = f"[ex{ex_index} s{sample_idx}]"
                try:
                    return await asyncio.wait_for(
                        _generate_one_with_tools(args, tokenizer, prompt, db_id, debug_tag=tag), timeout=args.sample_timeout
                    )
                except asyncio.TimeoutError:
                    print(f"{tag} SAMPLE-TIMEOUT (hard) after {args.sample_timeout}s", flush=True)
                    return dict(_TIMEOUT_RESULT)
                except Exception as exc:
                    print(f"{tag} ERROR {type(exc).__name__}: {exc}", flush=True)
                    return dict(_TIMEOUT_RESULT, timeout_stage="error")

        async def _process_example(ex_index: int, row: dict[str, Any]) -> None:
            nonlocal done_examples, num_correct, total_timeouts, examples_with_any_timeout, no_answer, total_traces
            prompt = row["prompt"]
            label = str(row.get("label", ""))
            metadata = row.get("metadata") or {}
            db_id = str(metadata.get("db_id", ""))

            kept_traces: list[dict[str, Any]] = []
            indices_to_run = list(range(args.num_samples))
            prev = resume_records.get(ex_index) if args.resume else None
            if prev is not None:
                prev_by_index = {int(t["trace_index"]): t for t in prev.get("traces", []) if isinstance(t.get("trace_index"), int)}
                timed_out_indices = {i for i, t in prev_by_index.items() if t.get("timed_out")}
                if prev_by_index and len(timed_out_indices) == len(prev_by_index):
                    indices_to_run, kept_traces = list(range(args.num_samples)), []
                else:
                    rerun = set(timed_out_indices) | {i for i in range(args.num_samples) if i not in prev_by_index}
                    indices_to_run = sorted(rerun)
                    kept_traces = [prev_by_index[i] for i in sorted(prev_by_index) if i < args.num_samples and i not in rerun]
                    if not indices_to_run:
                        async with output_lock:
                            done_examples += 1
                            num_correct += float(prev.get("avg_acc_excl_timeout", 0.0) or 0.0)
                            prev_timeouts = int(prev.get("num_timeouts", 0) or 0)
                            total_timeouts += prev_timeouts
                            total_traces += len(prev_by_index)
                            if prev_timeouts:
                                examples_with_any_timeout += 1
                            if output_file is not None:
                                output_file.write(json.dumps(prev, ensure_ascii=False) + "\n")
                                output_file.flush()
                        return

            generations = await asyncio.gather(*[_safe_generate(ex_index, i, prompt, db_id) for i in indices_to_run])
            new_traces = []
            for sample_idx, generation in zip(indices_to_run, generations):
                trace = await asyncio.to_thread(sql_tools.score_response, generation["response"], db_id, label)
                trace.update(
                    trace_index=sample_idx,
                    response=generation["response"],
                    turns=generation["turns"],
                    tool_call_count=generation["tool_call_count"],
                    total_trace_tokens=generation["total_trace_tokens"],
                    stopped_due_to_max_tokens=generation["stopped_due_to_max_tokens"],
                    timed_out=bool(generation.get("timed_out")),
                    timeout_stage=generation.get("timeout_stage"),
                    timeout_turn=generation.get("timeout_turn"),
                )
                new_traces.append(trace)
                if args.print_turns:
                    _print_trace_turns(ex_index, trace)

            traces = sorted(kept_traces + new_traces, key=lambda t: int(t.get("trace_index", 0)))
            valid = [t for t in traces if not t["timed_out"]]
            num_timeouts = len(traces) - len(valid)
            all_timed_out = not valid
            avg_acc = sum(int(t["acc"]) for t in traces) / len(traces)
            avg_score = sum(float(t["score"]) for t in traces) / len(traces)
            avg_acc_excl = 0.0 if all_timed_out else sum(int(t["acc"]) for t in valid) / len(valid)
            avg_score_excl = 0.0 if all_timed_out else sum(float(t["score"]) for t in valid) / len(valid)
            result = {
                "index": ex_index,
                "db_id": db_id,
                "dataset": metadata.get("dataset"),
                "question_id": metadata.get("question_id"),
                "difficulty": metadata.get("difficulty"),
                "prompt": prompt,
                "label": label,
                "avg_score": avg_score,
                "avg_acc": avg_acc,
                "avg_acc_excl_timeout": avg_acc_excl,
                "avg_score_excl_timeout": avg_score_excl,
                "num_timeouts": num_timeouts,
                "all_timed_out": all_timed_out,
                "traces": traces,
            }
            async with output_lock:
                num_correct += avg_acc_excl
                done_examples += 1
                total_timeouts += num_timeouts
                total_traces += len(traces)
                no_answer += sum(1 for t in new_traces if not t.get("pred"))
                if num_timeouts:
                    examples_with_any_timeout += 1
                if all_timed_out:
                    all_timeout_examples.append({"index": ex_index, "db_id": db_id, "label": label})
                if output_file is not None:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                running = num_correct / done_examples if done_examples else 0.0
                print(f"[{done_examples}/{num_examples}] ex{ex_index} acc={avg_acc_excl:.3f} timeouts={num_timeouts}/{args.num_samples} running_acc={running:.4f}", flush=True)

        try:
            await asyncio.gather(*[_process_example(i, row) for i, row in enumerate(rows)])
        finally:
            if output_file is not None:
                output_file.close()

        total_samples = num_examples * args.num_samples
        return {
            "env": "sql",
            "num_examples": num_examples,
            "accuracy": num_correct / num_examples if num_examples else 0.0,
            "no_answer_traces": no_answer,
            "timeout_stats": {
                "total_samples": total_samples,
                "total_timeouts": total_timeouts,
                "timeout_rate": total_timeouts / total_samples if total_samples else 0.0,
                "examples_with_any_timeout": examples_with_any_timeout,
                "examples_all_timeout": len(all_timeout_examples),
                "sample_timeout_seconds": args.sample_timeout,
                "request_timeout_seconds": args.request_timeout,
            },
            "all_timeout_examples": all_timeout_examples,
            "model_path": args.model_path,
            "dataset": dataset_name,
            "dataset_file": str(args.dataset),
            "db_root": sql_tools.TOOL_CONFIGS["db_root"],
            "num_samples": args.num_samples,
            "max_tool_calls": sql_tools.TOOL_CONFIGS["max_tool_calls"],
            "max_context_len": args.max_context_len,
            "temperature": args.temperature,
        }

    summary = asyncio.run(_run_eval())
    print(json.dumps(summary, indent=2))
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
