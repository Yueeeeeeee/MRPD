#!/usr/bin/env python3
"""Tabulate every eval that has already been run, math and search alike.

scripts/eval/math.sh prints a table of the summaries *that invocation* produced,
which is no help once the results are spread over several runs, several
checkpoints and several days of scrollback. This walks a directory tree, finds
every eval summary, and prints all of them in one table.

    python3 scripts/eval/collect_results.py /workspace/runs

The two environments write different shapes, and both are picked up:

    math    <run>/<iter_XXXXXXX>/<dataset>/summary.json    one per benchmark
    search  <run>/<iter_XXXXXXX>/eval_search_r1_summary.json
                                             one per checkpoint, seven QA sets
                                             inside its `per_data_source` field
    sql     <run>/<iter_XXXXXXX>/<dataset>/summary.json    math's shape, "env": "sql" inside

A summary is classified by its content, not its filename: anything carrying
`per_data_source` is expanded into one row per source. Anything else with a summary three levels deep still works
-- the two directories above it are read as <checkpoint>/<dataset>.

MEAN rows are per (run, checkpoint, environment): a checkpoint trained on mixed
data can carry both a math and a search eval, and averaging across them would be
meaningless.
"""

import argparse
import json
import os
import re
import sys

# The order each environment reports its benchmarks in. Anything not listed
# sorts alphabetically after them, rather than being dropped.
BENCH_ORDER = {
    "math": ["aime24", "aime25", "amc23", "minerva_math", "olympiadbench", "math500"],
    "search": ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"],
    "sql": ["spider_dev", "spider_test", "bird_minidev"],
}
ENV_ORDER = ["math", "search", "sql"]


def natural_key(name):
    """iter_0000099 sorts before iter_0000499, and both before a bare name."""
    m = re.search(r"(\d+)\s*$", name)
    return (0, int(m.group(1)), "") if m else (1, 0, name)


def env_key(env):
    return (ENV_ORDER.index(env), "") if env in ENV_ORDER else (len(ENV_ORDER), env)


def bench_key(env, name):
    order = BENCH_ORDER.get(env, [])
    return (order.index(name), "") if name in order else (len(order), name)


def find_summaries(roots):
    """Yield (search_root, path) for every eval summary under the given roots."""
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            yield os.path.dirname(os.path.dirname(os.path.dirname(root))), root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # rollout_debug tensors and hf/ weight dirs hold no summaries and are
            # the two big ones; not descending into them keeps the walk quick.
            dirnames[:] = [d for d in dirnames if d not in ("hf", "rollout_debug")]
            for name in sorted(filenames):
                # "summary.json" is math's; "*_summary.json" is search's
                # eval_search_r1_summary.json and anything shaped like it.
                if name == "summary.json" or name.endswith("_summary.json"):
                    yield root, os.path.join(dirpath, name)


def split_run_label(parts, root, depth):
    """Read <run>/<checkpoint> off the path components above the summary.

    `depth` is how many components the summary itself occupies below the
    checkpoint directory: 2 for math (<dataset>/summary.json), 1 for search
    (eval_search_r1_summary.json sits in the checkpoint directory itself).
    """
    if len(parts) <= depth:
        return os.path.basename(root), ""
    checkpoint = parts[-(depth + 1)]
    run = os.sep.join(parts[: -(depth + 1)])
    return (run or os.path.basename(root)), checkpoint


def collect(roots):
    rows = []
    for root, path in find_summaries(roots):
        parts = os.path.relpath(path, root).split(os.sep)

        try:
            with open(path) as fh:
                summary = json.load(fh)
            if not isinstance(summary, dict):
                raise ValueError("not an object")
        except (OSError, ValueError) as exc:
            run, checkpoint = split_run_label(parts, root, 2)
            rows.append({"run": run, "ckpt": checkpoint, "env": "?",
                         "bench": parts[-2] if len(parts) > 1 else "?",
                         "acc": None, "timeout": None, "n": None,
                         "note": f"unreadable: {exc.__class__.__name__}"})
            continue

        per_source = summary.get("per_data_source")
        timeout = (summary.get("timeout_stats") or {}).get("timeout_rate")
        n = summary.get("num_samples")

        if isinstance(per_source, dict) and per_source:
            # Search: one file, one row per QA set. timeout_stats is only kept
            # for the checkpoint as a whole, so every row repeats it.
            run, checkpoint = split_run_label(parts, root, 1)
            # A retrieval failure does not fail the eval: execute_predictions
            # catches it and writes "[ERROR] ..." into the transcript as if it
            # were a tool result, so the run completes and reports whatever the
            # model could answer closed-book. search_errors is the only trace of
            # that in the summary, which is why it is surfaced rather than left
            # in the file.
            errors = (summary.get("search_stats") or {}).get("search_errors") or 0
            for source, stats in per_source.items():
                rows.append({
                    "run": run, "ckpt": checkpoint, "env": "search", "bench": source,
                    "acc": (stats or {}).get("accuracy"), "timeout": timeout, "n": n,
                    "note": "", "search_errors": errors,
                })
        else:
            # Math (or SQL, which writes the same shape and names itself): one file
            # per benchmark, named by its parent directory.
            run, checkpoint = split_run_label(parts, root, 2)
            rows.append({
                "run": run, "ckpt": checkpoint, "env": summary.get("env") or "math",
                "bench": parts[-2] if len(parts) > 1 else "?",
                "acc": summary.get("accuracy"), "timeout": timeout, "n": n,
                "note": "",
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["runs"],
                    help="directories to search (default: runs)")
    ap.add_argument("--run", action="append", default=[],
                    help="only runs whose label contains this substring; repeatable")
    ap.add_argument("--bench", action="append", default=[],
                    help="only these benchmarks; repeatable")
    ap.add_argument("--env", choices=ENV_ORDER,
                    help="only one environment's results")
    ap.add_argument("--no-mean", action="store_true", help="suppress the per-checkpoint MEAN rows")
    args = ap.parse_args()

    rows = collect(args.roots or ["runs"])
    if args.run:
        rows = [r for r in rows if any(f in r["run"] for f in args.run)]
    if args.bench:
        rows = [r for r in rows if r["bench"] in args.bench]
    if args.env:
        rows = [r for r in rows if r["env"] == args.env]

    if not rows:
        print(f"No eval summary found under: {', '.join(args.roots or ['runs'])}")
        print("An eval only writes one after it finishes a benchmark, so this also")
        print("means 'every eval so far died before it produced a number'.")
        return 1

    rows.sort(key=lambda r: (r["run"], natural_key(r["ckpt"]), env_key(r["env"]),
                             bench_key(r["env"], r["bench"])))

    show_env = len({r["env"] for r in rows}) > 1
    w_run = max(max(len(r["run"]) for r in rows), len("run"))
    w_ckpt = max(max(len(r["ckpt"]) for r in rows), len("checkpoint"))
    w_env = max(max(len(r["env"]) for r in rows), len("env")) if show_env else 0
    w_bench = max(max(len(r["bench"]) for r in rows), len("MEAN of 99"), len("benchmark"))
    width = w_run + w_ckpt + w_env + w_bench + 28 + (2 if show_env else 0)

    def line(run, ckpt, env, bench, tail):
        cells = [f"{run:<{w_run}}", f"{ckpt:<{w_ckpt}}"]
        if show_env:
            cells.append(f"{env:<{w_env}}")
        cells.append(f"{bench:<{w_bench}}")
        return "  ".join(cells) + "  " + tail

    print()
    print("=" * width)
    print(line("run", "checkpoint", "env", "benchmark",
               f"{'acc':>7} {'timeout':>8} {'n':>3}"))
    print("-" * width)

    groups = []  # (run, ckpt, env) in first-seen order, for the MEAN rows
    prev_group = None
    for r in rows:
        group = (r["run"], r["ckpt"], r["env"])
        if group != prev_group:
            if prev_group is not None:
                print()
            prev_group = group
        if group not in groups:
            groups.append(group)
        if r["acc"] is None:
            tail = f"{'--':>7} {'--':>8} {'--':>3}  ({r['note']})"
        else:
            timeout = f"{r['timeout']:>7.1%}" if r["timeout"] is not None else f"{'--':>7}"
            n = f"{r['n']:>3}" if r["n"] is not None else f"{'--':>3}"
            tail = f"{r['acc']:>7.4f} {timeout} {n}"
        print(line(r["run"], r["ckpt"], r["env"], r["bench"], tail))

    if not args.no_mean:
        print("-" * width)
        for run, ckpt, env in groups:
            vals = [r["acc"] for r in rows
                    if (r["run"], r["ckpt"], r["env"]) == (run, ckpt, env) and r["acc"] is not None]
            if len(vals) > 1:
                print(line(run, ckpt, env, f"MEAN of {len(vals)}", f"{sum(vals) / len(vals):>7.4f}"))

    print("=" * width)
    print(f"{len(rows)} result{'' if len(rows) == 1 else 's'} from "
          f"{len(groups)} checkpoint-environment pair{'' if len(groups) == 1 else 's'}.")
    print("acc is mean pass@1. Math counts timed-out samples out of a problem's")
    print("denominator, so a non-zero timeout column makes that accuracy optimistic;")
    print("search reports one timeout rate for the whole checkpoint, repeated on")
    print("every row. A MEAN row averages only the benchmarks present, so it is")
    print("comparable across checkpoints only when they saw the same set --")
    print("6 for math, 7 for search, 3 for sql.")

    # Loud, because the number it invalidates looks entirely reasonable.
    broken = sorted({(r["run"], r["ckpt"], r.get("search_errors", 0))
                     for r in rows if r.get("search_errors")})
    # Rows with different num_samples are not the same estimator: mean pass@1 over
    # four traces and a single trace answer the same question with different
    # variance, and averaging them is meaningless. This shows up the moment a
    # default changes mid-project, which is exactly when it is easy to miss.
    sample_counts = sorted({r["n"] for r in rows if r.get("n") is not None})
    if len(sample_counts) > 1:
        sys.stdout.flush()
        print(f"\nWARNING  num_samples differs across rows ({sample_counts}); those "
              f"accuracies are not directly comparable.", file=sys.stderr)

    if broken:
        sys.stdout.flush()   # stderr is unbuffered; keep the warnings under the table
        print()
        for run, ckpt, errors in broken:
            print(f"WARNING  {run}/{ckpt}: search_errors={errors} -- some turns got "
                  f"'[ERROR] ...' instead of documents.", file=sys.stderr)
        print("WARNING  Those traces answered closed-book; the accuracy above is not "
              "an open-book number.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
