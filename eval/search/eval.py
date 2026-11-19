"""Evaluate a Search-R1 model against an SGLang server.

Mirrors eval/math/eval.py: assumes an SGLang server AND a local retrieval
server are running. scripts/eval/search.sh starts both
automatically.

Run with:
    bash scripts/eval/search.sh
or directly:
    python eval/search/eval.py --model-path ... --dataset baohao/search-test
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import random
import socket
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# local_search_server.py lives in train/src and is shared
# with the training-side generate_with_search.py. Appended, not inserted, so that
# the eval-local copies of any other module keep priority over the training ones.
_TRAIN_SRC = REPO_ROOT / "train" / "src"
if str(_TRAIN_SRC) not in sys.path:
    sys.path.append(str(_TRAIN_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Search-R1 model against an already-running SGLang server "
            "(scripts/eval/search.sh starts it automatically) and a local "
            "retrieval server (Search-R1's retrieval_launch.sh)."
        )
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing --output JSONL: re-run only the timed-out or missing samples "
            "per prompt (keeping non-timeout samples), and re-run all samples for prompts "
            "whose samples were all timed out. The original output file is backed up to "
            "<output>.bak before being overwritten."
        ),
    )
    parser.add_argument("--model-path", required=True, help="HF model path (used for the chat template + summary)")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path. Defaults to --model-path.")
    parser.add_argument("--dataset", required=True, help="HF dataset name (e.g. baohao/search-test) or local parquet/jsonl")
    parser.add_argument("--split", default="test", help="HF dataset split (only used when --dataset is a HF name)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--max-examples-per-source", type=int, default=2500,
                        help="Cap each data_source at N examples, sampled with --sample-seed. "
                             "0 disables the cap. Applied before --limit.")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for --max-examples-per-source. Same seed = same subset.")
    parser.add_argument("--data-source-filter", default=None,
                        help="Comma-separated subset of data_source values to keep (e.g. 'nq,hotpotqa').")
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="Traces per prompt (>=1)")

    parser.add_argument("--host", default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=30000, help="SGLang server port")
    parser.add_argument("--server-wait-timeout", type=int, default=60,
                        help="Seconds to wait for the SGLang port to accept connections")
    parser.add_argument("--request-timeout", type=int, default=1800, help="HTTP timeout per /generate call")

    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens per turn")
    parser.add_argument("--max-tokens-per-turn", type=int, default=4096,
                        help="Max new tokens for a single turn; mirrors SEARCH_R1_MAX_TOKENS_PER_TURN "
                             "in the search training scripts. Must match training, or the eval lets "
                             "one turn run past what the policy was trained to produce.")
    parser.add_argument("--max-context-len", type=int, default=9216,
                        help="Hard cap on input_tokens + max_new_tokens per /generate call; "
                             "matches ROLLOUT_MAX_CONTEXT_LEN in the search training scripts")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--max-turns", type=int, default=8, help="Max search rounds per trace")
    parser.add_argument("--topk", type=int, default=3, help="topk for the retrieval server")
    parser.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve",
                        help="URL of the local retrieval server, or comma-separated URLs for round-robin")
    parser.add_argument("--search-backend", choices=["local"], default="local")

    parser.add_argument("--max-concurrent", type=int, default=8,
                        help="Max concurrent in-flight samples")
    parser.add_argument("--sample-timeout", type=int, default=600,
                        help="Hard wall-clock budget per trace (all turns combined)")

    parser.add_argument("--output", default=None, help="JSONL output for per-example results")
    parser.add_argument("--summary-output", default=None, help="JSON output for the final summary")
    parser.add_argument("--print-turns", action="store_true", help="Print per-turn assistant/observation text")
    parser.add_argument("--debug-trace", action="store_true",
                        help="Print per-turn START/GEN/SEARCH timings")

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("-n/--num-samples must be at least 1")
    return args


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_tokenizer(tokenizer_path: str):
    transformers = importlib.import_module("transformers")
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _load_runtime():
    """Import the eval-side helpers (search/postprocess/execute_predictions)."""
    return importlib.import_module("eval_generate_with_search")


def _load_grader():
    """Return qa_em_format.compute_score_em."""
    module = importlib.import_module("qa_em_format")
    return module.compute_score_em


def _load_dataset_records(args: argparse.Namespace):
    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        if dataset_path.suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
        elif dataset_path.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        else:
            dataset = load_dataset(str(dataset_path), split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.data_source_filter:
        wanted = {s.strip() for s in args.data_source_filter.split(",") if s.strip()}
        dataset = dataset.filter(lambda r: r.get("data_source") in wanted)

    dataset = _subsample_per_source(dataset, args.max_examples_per_source, args.sample_seed)

    if args.limit is not None:
        limit = min(args.limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def _subsample_per_source(dataset, cap: int, seed: int):
    """Cap every data_source at `cap` examples, deterministically.

    baohao/search-test merges seven QA sets whose sizes differ by two orders of
    magnitude (Bamboogle has 125 questions, PopQA over 14k). Evaluating all of
    them at --num-samples 4 costs more than it buys: past a couple of thousand
    questions the standard error on a per-source accuracy is already well under a
    point, so the extra rollouts only add wall-clock. The cap is per source, not
    global, because a global one would silently re-weight the mix toward whatever
    the big sets are.

    Sampling is seeded per source, so the same (cap, seed) always selects the
    same questions -- across checkpoints, across arms, and across a --resume of a
    crashed run, which keys its records on position in this filtered dataset.
    Selected indices keep their original order for the same reason.
    """
    if not cap or cap <= 0 or len(dataset) <= cap:
        return dataset

    sources = dataset["data_source"] if "data_source" in dataset.column_names else [None] * len(dataset)
    by_source: dict[Any, list[int]] = {}
    for i, src in enumerate(sources):
        by_source.setdefault(src, []).append(i)

    keep: list[int] = []
    for src, indices in by_source.items():
        if len(indices) <= cap:
            keep.extend(indices)
            continue
        rng = random.Random(f"{seed}:{src}")
        picked = rng.sample(indices, cap)
        keep.extend(picked)
        print(f"[subsample] {src}: {len(indices)} -> {cap} (seed {seed})", flush=True)
    if len(keep) == len(dataset):
        return dataset
    keep.sort()
    return dataset.select(keep)


# ---------------------------------------------------------------------------
# Prompt / extraction helpers
# ---------------------------------------------------------------------------


def _row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the chat messages for a Search-R1 verl-style row."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
            for m in prompt
            if isinstance(m, dict)
        ]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise ValueError(f"Unsupported prompt format in row: {type(prompt)}")


def _row_ground_truth(row: dict[str, Any]) -> dict[str, list[str]]:
    """Extract the ground_truth dict {target: [...]} from a verl-style row."""
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if isinstance(gt, dict) and "target" in gt:
            target = gt["target"]
            if isinstance(target, str):
                target = [target]
            return {"target": list(target)}
    # Fallbacks for ad-hoc datasets.
    for key in ("answer", "answers", "golden_answers", "gt"):
        if key in row:
            value = row[key]
            if isinstance(value, str):
                return {"target": [value]}
            if isinstance(value, list):
                return {"target": [str(v) for v in value]}
    raise ValueError("Could not extract ground_truth from row")


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply the model's chat template (returning text)."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Async HTTP / server helpers (lightweight clone of retool/eval.py)
# ---------------------------------------------------------------------------


# SGLang's HTTP front end stops accepting new connections for seconds at a time under
# load, and so does this client's own event loop while it parses large responses; a
# single 10 s attempt threw the whole trace away on every such stall. Nothing has been
# sent before the connection exists, so retrying the connect cannot duplicate a request.
CONNECT_TIMEOUT_S = 30
CONNECT_ATTEMPTS = 5


async def _open_connection(host: str, port: int):
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=CONNECT_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError) as exc:  # timed out, refused, reset
            if attempt == CONNECT_ATTEMPTS:
                raise RuntimeError(
                    f"HTTP connect to {host}:{port} failed {CONNECT_ATTEMPTS} times, "
                    f"{CONNECT_TIMEOUT_S}s each ({type(exc).__name__}: {exc})"
                ) from exc
            await asyncio.sleep(2 ** (attempt - 1))  # 1, 2, 4, 8 s


async def _post_json_async(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = json.dumps(payload).encode("utf-8")
    req = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body

    reader, writer = await _open_connection(host, port)
    try:
        writer.write(req)
        await asyncio.wait_for(writer.drain(), timeout=10)
        response_bytes = await asyncio.wait_for(reader.read(-1), timeout=timeout)
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=5)
        except Exception:
            pass

    sep = response_bytes.find(b"\r\n\r\n")
    if sep == -1:
        raise ValueError(f"Malformed HTTP response (no header separator): {response_bytes[:200]}")
    return json.loads(response_bytes[sep + 4 :])


async def _post_generate(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/generate"
    return await _post_json_async(url, payload, args.request_timeout)


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
    raise TimeoutError(
        f"Timed out waiting for SGLang server on {host}:{port} after {timeout}s. "
        f"Is the server running? Last socket error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Multi-turn rollout
# ---------------------------------------------------------------------------


async def _generate_one_with_search(
    args: argparse.Namespace,
    tokenizer: Any,
    base_prompt: str,
    runtime: Any,
    debug_tag: str = "",
) -> dict[str, Any]:
    trace_start = time.time()
    response_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    search_count = 0
    timed_out = False
    finish_reason: str | None = None

    progress: dict[str, Any] = {"current_turn": -1, "current_stage": "init",
                                 "stage_started_at": trace_start}

    try:
        # The token stream training builds (generate_with_search.py): the prompt
        # tokenized once, then the model's own tokens and the rendered observations
        # appended as they come; every /generate call carries exactly that stream.
        prompt_ids = tokenizer(base_prompt, add_special_tokens=False)["input_ids"]
        response_ids: list[int] = []
        max_context_len = args.max_context_len - 16  # the buffer training keeps
        for turn_index in range(args.max_turns):
            input_ids = prompt_ids + response_ids
            # Three terms, matching training's generate_with_search.py. The per-turn
            # cap is the one that is easy to forget: without it a single turn may
            # generate twice what the policy was ever allowed to generate in
            # training, which costs wall time on repetition loops and evaluates
            # the policy off-distribution.
            turn_max_new = min(args.max_new_tokens, max_context_len - len(input_ids), args.max_tokens_per_turn)
            if turn_max_new <= 0:
                if args.debug_trace:
                    print(
                        f"{debug_tag} turn={turn_index} CONTEXT-FULL input_tokens={len(input_ids)} "
                        f"max_context_len={args.max_context_len} — stopping",
                        flush=True,
                    )
                break

            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": turn_max_new,
                    # The same stop set training uses (generate_with_search.py:
                    # _merged_stop_strings), so a turn ends where the policy was
                    # trained for it to end.
                    "stop": ["</tool_call>", "</answer>"],
                    "no_stop_trim": True,
                },
                "return_logprob": True,  # the token ids come with the log-probs, as in training
            }
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} GEN-> input_tokens={len(input_ids)} "
                    f"elapsed={time.time() - trace_start:.1f}s",
                    flush=True,
                )
            gen_t0 = time.time()
            progress["current_turn"] = turn_index
            progress["current_stage"] = "gen"
            progress["stage_started_at"] = gen_t0
            output = await _post_generate(args, payload)
            raw_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            raw_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = runtime.postprocess_responses(tokenizer.decode(raw_ids, skip_special_tokens=False))
            cur_ids, _ = runtime._trim_response_token_prefix(tokenizer, raw_ids, raw_log_probs, cur_response)
            finish_reason = output["meta_info"]["finish_reason"]["type"]
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} GEN<- gen_s={time.time() - gen_t0:.1f} "
                    f"resp_tokens={len(cur_ids)} finish={finish_reason}",
                    flush=True,
                )

            response_ids += cur_ids
            response_parts.append(cur_response)
            turn_record = {"turn_index": turn_index, "assistant": cur_response}

            if finish_reason == "abort":
                turn_record["aborted"] = True
                turns.append(turn_record)
                break

            action, _ = runtime.postprocess_predictions(cur_response)
            if finish_reason == "length" and action not in {"search", "answer", "boxed_answer"}:
                turns.append(turn_record)
                break

            search_t0 = time.time()
            progress["current_stage"] = "search"
            progress["stage_started_at"] = search_t0
            next_obs, done = await runtime.execute_predictions(cur_response)
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} SEARCH<- search_s={time.time() - search_t0:.1f} "
                    f"action={action} done={done} obs_chars={len(next_obs)}",
                    flush=True,
                )
            turn_record["done"] = done
            if done:
                turns.append(turn_record)
                break

            obs_ids = tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) + len(response_ids) + len(obs_ids) > max_context_len:
                # As in training: an observation that does not fit ends the trajectory.
                turns.append(turn_record)
                break
            response_ids += obs_ids
            response_parts.append(next_obs)
            # Increment search_count only when <tool_response> is present in the observation (successful search)
            if "<tool_response>" in next_obs:
                search_count += 1
            turn_record["observation"] = next_obs
            turns.append(turn_record)
    except asyncio.CancelledError:
        timed_out = True
        if debug_tag:
            stage = progress.get("current_stage", "init")
            stage_dt = time.time() - progress.get("stage_started_at", trace_start)
            print(
                f"{debug_tag} CANCELLED stage={stage} turn={progress.get('current_turn', -1)} "
                f"stage_s={stage_dt:.1f} elapsed={time.time() - trace_start:.1f}s "
                f"completed_turns={len(turns)} searches={search_count}",
                flush=True,
            )

    return {
        "response": "".join(response_parts),
        "turns": turns,
        "search_count": search_count,
        "timed_out": timed_out,
        "finish_reason": finish_reason,
        "timeout_stage": progress.get("current_stage") if timed_out else None,
        "timeout_turn": progress.get("current_turn") if timed_out else None,
    }


# ---------------------------------------------------------------------------
# Scoring + per-trace helpers
# ---------------------------------------------------------------------------


def _score_response(grader: Any, prompt: str, ground_truth: dict[str, Any], response: str,
                    format_score: float) -> dict[str, Any]:
    # Same split as the trainer's reward_func: the transcript for the format and
    # retrieval checks, the model's own output for the answer. The prompt itself
    # contains \boxed{$Answer} and \boxed{Beijing}, so reading the answer out of
    # prompt + response scores a trajectory that never answered as "Beijing".
    score = grader(
        solution_str=prompt + response,
        response_str=response,
        ground_truth=ground_truth,
        format_score=format_score,
    )
    score = float(score)
    return {
        "score": score,
        "acc": score >= 1.0,
        "response": response,
    }


def _print_trace_turns(example_index: int, trace: dict[str, Any]) -> None:
    print(f"\n=== example {example_index} trace {trace['trace_index']} ===")
    print(
        f"searches={trace['search_count']} score={trace['score']:.3f} "
        f"acc={int(trace['acc'])} timed_out={trace['timed_out']}"
    )
    for turn in trace.get("turns", []):
        print(f"\n--- turn {turn['turn_index']} assistant ---")
        print(turn.get("assistant", ""))
        if "observation" in turn:
            print("\n--- observation ---")
            print(turn["observation"])


def _tool_response_contents(observation: str) -> list[str]:
    contents = []
    start_tag = "<tool_response>"
    end_tag = "</tool_response>"
    start = 0
    while True:
        start = observation.find(start_tag, start)
        if start == -1:
            break
        start += len(start_tag)
        end = observation.find(end_tag, start)
        if end == -1:
            break
        contents.append(observation[start:end].strip())
        start = end + len(end_tag)
    return contents


def _count_search_outcomes(traces: list[dict[str, Any]]) -> tuple[int, int]:
    empty_results = 0
    errors = 0
    for trace in traces:
        for turn in trace.get("turns", []):
            observation = turn.get("observation", "")
            if not isinstance(observation, str):
                continue
            for content in _tool_response_contents(observation):
                if not content:
                    empty_results += 1
                elif content.startswith("[ERROR]"):
                    errors += 1
    return empty_results, errors


# ---------------------------------------------------------------------------
# Main eval driver
# ---------------------------------------------------------------------------


async def _preflight_search(runtime: Any) -> None:
    """Issue one real retrieval call before the eval starts, and abort if it fails.

    A broken retriever does not crash the eval on its own: execute_predictions
    catches the exception and writes "[ERROR] ..." into the transcript as if it
    were a tool result. The run then completes normally and reports a plausible
    closed-book accuracy, with the only evidence buried in search_errors. Fail
    loudly here instead.
    """
    cfg = runtime.SEARCH_R1_CONFIGS
    backend = cfg["search_backend"]
    where = cfg["local"]["search_url"]
    probe = "who wrote pride and prejudice"
    try:
        result = await runtime.search(probe)
    except Exception as e:
        raise SystemExit(
            f"[preflight] retrieval FAILED (backend={backend}, url={where}) for the probe "
            f"query {probe!r}: {type(e).__name__}: {e}\n"
            "[preflight] Every search would come back as '[ERROR] ...' and the eval would "
            "silently measure closed-book accuracy. Aborting."
        ) from e
    if not result or not result.strip():
        raise SystemExit(
            f"[preflight] retrieval returned nothing (backend={backend}, url={where}) for the "
            f"probe query {probe!r}. Aborting rather than measuring closed-book accuracy."
        )
    print(f"[preflight] retrieval OK: {len(result)} chars for {probe!r}", flush=True)


def _apply_runtime_overrides(args: argparse.Namespace, runtime: Any) -> None:
    """Push CLI overrides into the runtime's SEARCH_R1_CONFIGS."""
    cfg = runtime.SEARCH_R1_CONFIGS
    cfg["max_turns"] = args.max_turns
    cfg["topk"] = args.topk
    cfg["search_backend"] = args.search_backend
    cfg["local"]["search_url"] = args.search_url


def main() -> None:
    def _load_resume_records(output_path: Path) -> dict[int, dict[str, Any]]:
        """Load an existing eval JSONL into {example_index: result_record}. Last one wins."""
        records: dict[int, dict[str, Any]] = {}
        if not output_path or not output_path.exists():
            return records
        with output_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[resume] skipping malformed line {line_no} in {output_path}: {exc}", flush=True)
                    continue
                idx = rec.get("index")
                if isinstance(idx, int):
                    records[idx] = rec
        return records
    args = parse_args()
    tokenizer = _load_tokenizer(args.tokenizer_path or args.model_path)
    runtime = _load_runtime()
    grader = _load_grader()
    _apply_runtime_overrides(args, runtime)
    asyncio.run(_preflight_search(runtime))

    dataset = _load_dataset_records(args)
    output_path = Path(args.output) if args.output else None


    resume_records: dict[int, dict[str, Any]] = {}
    if getattr(args, "resume", False):
        assert output_path is not None
        resume_records = _load_resume_records(output_path)
        print(f"[resume] loaded {len(resume_records)} previous results from {output_path}", flush=True)
        if output_path.exists():
            backup_path = output_path.with_suffix(output_path.suffix + ".bak")
            output_path.replace(backup_path)
            print(f"[resume] backed up previous output to {backup_path}", flush=True)

    print(
        f"Connecting to SGLang server at http://{args.host}:{args.port} "
        f"(waiting up to {args.server_wait_timeout}s).",
        flush=True,
    )
    _wait_for_port(args.host, args.port, timeout=args.server_wait_timeout)
    print(f"SGLang server is reachable at http://{args.host}:{args.port}", flush=True)

    async def _run_eval() -> dict[str, Any]:
        num_examples = len(dataset)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        output_file = output_path.open("w", encoding="utf-8") if output_path else None
        output_lock = asyncio.Lock()

        rows = [(i, dataset[i]) for i in range(num_examples)]

        done_examples = 0
        num_correct = 0.0
        total_score = 0.0
        total_timeouts = 0
        examples_with_any_timeout = 0
        total_empty_search_results = 0
        total_search_errors = 0

        async def _safe_generate(ex_index: int, sample_idx: int, prompt_text: str) -> dict[str, Any]:
            tag = f"[ex{ex_index} s{sample_idx}]"
            async with semaphore:
                t0 = time.time()
                try:
                    return await asyncio.wait_for(
                        _generate_one_with_search(args, tokenizer, prompt_text, runtime, debug_tag=tag),
                        timeout=args.sample_timeout,
                    )
                except asyncio.TimeoutError:
                    print(
                        f"{tag} SAMPLE-TIMEOUT (hard) after {args.sample_timeout}s "
                        f"(wall={time.time() - t0:.1f}s)",
                        flush=True,
                    )
                    return {
                        "response": "",
                        "turns": [],
                        "search_count": 0,
                        "timed_out": True,
                        "finish_reason": None,
                        "timeout_stage": "unknown",
                        "timeout_turn": -1,
                    }
                except Exception as exc:
                    print(
                        f"{tag} ERROR {type(exc).__name__}: {exc} (wall={time.time() - t0:.1f}s)",
                        flush=True,
                    )
                    return {
                        "response": "",
                        "turns": [],
                        "search_count": 0,
                        "timed_out": True,
                        "finish_reason": None,
                        "timeout_stage": "error",
                        "timeout_turn": -1,
                    }

        async def _process_example(ex_index: int, row: dict[str, Any]) -> None:
            nonlocal done_examples, num_correct, total_score
            nonlocal total_timeouts, examples_with_any_timeout
            nonlocal total_empty_search_results, total_search_errors

            messages = _row_messages(row)
            ground_truth = _row_ground_truth(row)
            prompt_text = runtime.format_conversation_with_tools(
                prompt=messages,
                system_prompt=None,
                messages=None
            )

            # Resume logic: figure out which sample indices need to be (re)run and which existing traces to keep as-is.
            kept_traces: list[dict[str, Any]] = []
            indices_to_run: list[int] = list(range(args.num_samples))
            resume_rerun_note = ""
            prev = resume_records.get(ex_index) if getattr(args, "resume", False) else None
            if prev is not None:
                prev_traces = prev.get("traces", []) or []
                prev_by_index: dict[int, dict[str, Any]] = {}
                for trace in prev_traces:
                    trace_index = trace.get("trace_index")
                    if not isinstance(trace_index, int):
                        continue
                    if 0 <= trace_index < args.num_samples:
                        prev_by_index[int(trace_index)] = trace
                timed_out_indices = {
                    i for i, t in prev_by_index.items() if t.get("timed_out", False)
                }
                search_error_indices = {
                    i for i, t in prev_by_index.items() if _count_search_outcomes([t])[1] > 0
                }
                rerun_set = set(timed_out_indices) | set(search_error_indices)
                if timed_out_indices or search_error_indices:
                    resume_rerun_note = (
                        f"resume_rerun_timeouts={len(timed_out_indices)} "
                        f"resume_rerun_search_errors={len(search_error_indices)} "
                    )
                all_rerun_prev = (
                    len(prev_by_index) > 0 and len(rerun_set) == len(prev_by_index)
                )
                if all_rerun_prev:
                    indices_to_run = list(range(args.num_samples))
                    kept_traces = []
                else:
                    present_indices = set(prev_by_index.keys())
                    for i in range(args.num_samples):
                        if i not in present_indices:
                            rerun_set.add(i)
                    indices_to_run = sorted(rerun_set)
                    kept_traces = [
                        prev_by_index[i]
                        for i in sorted(present_indices)
                        if i < args.num_samples and i not in rerun_set
                    ]
                    if not indices_to_run:
                        # Nothing to do for this example — emit the previous record verbatim.
                        async with output_lock:
                            done_examples += 1
                            num_timeouts_prev = int(prev.get("num_timeouts", 0) or 0)
                            num_correct += float(prev.get("avg_acc_excl_timeout", 0.0) or 0.0)
                            total_score += float(prev.get("avg_score_excl_timeout", 0.0) or 0.0)
                            total_timeouts += num_timeouts_prev
                            empty_prev = prev.get("empty_search_results")
                            errors_prev = prev.get("search_errors")
                            if empty_prev is None or errors_prev is None:
                                empty_prev, errors_prev = _count_search_outcomes(prev.get("traces", []) or [])
                            empty_prev = int(empty_prev or 0)
                            errors_prev = int(errors_prev or 0)
                            prev["empty_search_results"] = empty_prev
                            prev["search_errors"] = errors_prev
                            total_empty_search_results += empty_prev
                            total_search_errors += errors_prev
                            if num_timeouts_prev > 0:
                                examples_with_any_timeout += 1
                            if output_file is not None:
                                output_file.write(json.dumps(prev, ensure_ascii=False) + "\n")
                                output_file.flush()
                            running_acc = num_correct / done_examples if done_examples else 0.0
                            print(
                                f"[{done_examples}/{num_examples}] ex{ex_index} [resume:keep] "
                                f"timeouts={num_timeouts_prev}/{args.num_samples} "
                                f"empty_search={empty_prev} search_errors={errors_prev} "
                                f"running_empty_search={total_empty_search_results} "
                                f"running_search_errors={total_search_errors} "
                                f"running_acc={running_acc:.4f}",
                                flush=True,
                            )
                        return

            generations = await asyncio.gather(*[
                _safe_generate(ex_index, i, prompt_text) for i in indices_to_run
            ])

            new_traces: list[dict[str, Any]] = []
            for sample_idx, generation in zip(indices_to_run, generations):
                trace = _score_response(
                    grader, prompt_text, ground_truth, generation["response"],
                    runtime.SEARCH_R1_CONFIGS["format_score"],
                )
                trace["trace_index"] = sample_idx
                trace["turns"] = generation["turns"]
                trace["search_count"] = generation["search_count"]
                trace["timed_out"] = generation["timed_out"]
                trace["finish_reason"] = generation.get("finish_reason")
                if trace["timed_out"]:
                    trace["timeout_stage"] = generation.get("timeout_stage")
                    trace["timeout_turn"] = generation.get("timeout_turn")
                new_traces.append(trace)
                if args.print_turns:
                    _print_trace_turns(ex_index, trace)

            traces = sorted(
                kept_traces + new_traces,
                key=lambda t: int(t.get("trace_index", 0)),
            )

            valid_traces = [t for t in traces if not t["timed_out"]]
            num_timeouts = len(traces) - len(valid_traces)
            all_timed_out = len(valid_traces) == 0
            empty_search_results, search_errors = _count_search_outcomes(traces)

            avg_score = sum(t["score"] for t in traces) / len(traces)
            avg_acc = sum(int(t["acc"]) for t in traces) / len(traces)
            if all_timed_out:
                avg_acc_excl_timeout = 0.0
                avg_score_excl_timeout = 0.0
            else:
                avg_acc_excl_timeout = sum(int(t["acc"]) for t in valid_traces) / len(valid_traces)
                avg_score_excl_timeout = sum(t["score"] for t in valid_traces) / len(valid_traces)

            result = {
                "index": ex_index,
                "data_source": row.get("data_source"),
                "messages": messages,
                "rendered_prompt": prompt_text,
                "ground_truth": ground_truth,
                "avg_score": avg_score,
                "avg_acc": avg_acc,
                "avg_acc_excl_timeout": avg_acc_excl_timeout,
                "avg_score_excl_timeout": avg_score_excl_timeout,
                "num_timeouts": num_timeouts,
                "empty_search_results": empty_search_results,
                "search_errors": search_errors,
                "all_timed_out": all_timed_out,
                "traces": traces,
            }

            async with output_lock:
                num_correct += avg_acc_excl_timeout
                total_score += avg_score_excl_timeout
                done_examples += 1
                total_timeouts += num_timeouts
                total_empty_search_results += empty_search_results
                total_search_errors += search_errors
                if num_timeouts > 0:
                    examples_with_any_timeout += 1
                if output_file is not None:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                running_acc = num_correct / done_examples if done_examples else 0.0
                print(
                    f"[{done_examples}/{num_examples}] ex{ex_index} "
                    f"data_source={row.get('data_source')} "
                    f"{resume_rerun_note}"
                    f"avg_acc_excl_timeout={avg_acc_excl_timeout:.3f} "
                    f"timeouts={num_timeouts}/{args.num_samples} "
                    f"empty_search={empty_search_results} "
                    f"search_errors={search_errors} "
                    f"running_empty_search={total_empty_search_results} "
                    f"running_search_errors={total_search_errors} "
                    f"running_acc={running_acc:.4f}",
                    flush=True,
                )

        try:
            await asyncio.gather(*[_process_example(idx, row) for idx, row in rows])
        finally:
            if output_file is not None:
                output_file.close()

        # Per-data-source breakdown.
        per_source: dict[str, dict[str, float]] = {}
        if output_path is not None and output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = rec.get("data_source") or "unknown"
                    bucket = per_source.setdefault(src, {"n": 0, "sum_acc": 0.0, "sum_score": 0.0})
                    bucket["n"] += 1
                    bucket["sum_acc"] += float(rec.get("avg_acc_excl_timeout", 0.0) or 0.0)
                    bucket["sum_score"] += float(rec.get("avg_score_excl_timeout", 0.0) or 0.0)
        per_source_summary = {
            src: {
                "num_examples": int(b["n"]),
                "accuracy": b["sum_acc"] / b["n"] if b["n"] else 0.0,
                "average_score": b["sum_score"] / b["n"] if b["n"] else 0.0,
            }
            for src, b in per_source.items()
        }

        total_samples = num_examples * args.num_samples
        return {
            "num_examples": num_examples,
            "accuracy": num_correct / num_examples if num_examples else 0.0,
            "average_score": total_score / num_examples if num_examples else 0.0,
            "per_data_source": per_source_summary,
            "timeout_stats": {
                "total_samples": total_samples,
                "total_timeouts": total_timeouts,
                "timeout_rate": total_timeouts / total_samples if total_samples else 0.0,
                "examples_with_any_timeout": examples_with_any_timeout,
                "sample_timeout_seconds": args.sample_timeout,
                "request_timeout_seconds": args.request_timeout,
            },
            "search_stats": {
                "empty_search_results": total_empty_search_results,
                "search_errors": total_search_errors,
            },
            "model_path": args.model_path,
            "dataset": args.dataset,
            "split": args.split,
            "num_samples": args.num_samples,
            "search_backend": args.search_backend,
            "search_url": args.search_url,
            "topk": args.topk,
            "max_turns": args.max_turns,
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
