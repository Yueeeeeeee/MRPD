#!/usr/bin/env python3
"""Tabulate pass@k from the traces of every math and SQL eval already run.

scripts/eval/collect_results.py reads the summaries (accuracy, i.e. avg@N). This reads the
traces behind them, <...>/<checkpoint>/<dataset>/results.jsonl, and prints the unbiased
pass@k estimate of every set for every k <= N, one table for everything found under the
given roots:

    python3 scripts/eval/collect_pass_at_k.py /workspace/runs
    python3 scripts/eval/collect_pass_at_k.py /workspace/runs --run dense_reverse_t299 --env math -k 1 4 8 16 32
    python3 scripts/eval/collect_pass_at_k.py /workspace/runs --csv pass_at_k.csv     # long format, for plots

Search is not covered: its eval takes one sample per question, so pass@1 is all there is
(collect_results.py has it). No GPU and no model: the traces are read as they are.

pass@k of a problem with n traces of which c are correct is 1 - C(n-c, k) / C(n, k), the
unbiased estimator of Chen et al. 2021; a set's pass@k is the mean over its problems, every
problem with the same weight, so pass@1 is the summary's accuracy. Timed-out traces are
left out of a problem's n, as the accuracy leaves them out of its mean (a problem whose
traces all timed out scores 0 at every k); a problem with fewer valid traces than k is
estimated at its own n, a lower bound, and the "short" column counts them. --timeouts wrong
counts a timed-out trace as incorrect instead, n staying the sample count.

MEAN rows are per (run, checkpoint, environment), one per distinct N among its sets: the
mean over the sets with at least that many samples, at the k those sets all have (math: the
six sets up to k = 4, the three 32-sample sets up to k = 32).
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import BENCH_ORDER, bench_key, env_key, natural_key, split_run_label  # noqa: E402

DEFAULT_KS = [1, 2, 4, 8, 16, 32]


def pass_at_k(n, c, k):
    """Unbiased pass@k: the chance that k of the n traces, drawn without replacement,
    contain at least one of the c correct ones. 1 when fewer than k traces are wrong."""
    if k < 1 or n < 1:
        return 0.0
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1.0 - k / i
    return 1.0 - p


def problem_counts(record, timeouts):
    """(n, c) of one results.jsonl record: the traces counted and the correct ones among them."""
    traces = record.get("traces") or []
    if timeouts == "exclude":
        traces = [t for t in traces if not t.get("timed_out", False)]
        return len(traces), sum(1 for t in traces if t.get("acc"))
    return len(traces), sum(1 for t in traces if t.get("acc") and not t.get("timed_out", False))


def set_pass_at_k(records, ks, timeouts):
    """pass@k of one set for every k in ks, averaged over its problems, with the number of
    problems and, per k, how many were estimated at a smaller n than k."""
    sums = {k: 0.0 for k in ks}
    short = {k: 0 for k in ks}
    problems = 0
    for rec in records:
        n, c = problem_counts(rec, timeouts)
        problems += 1
        for k in ks:
            if n < k:
                short[k] += 1
            sums[k] += pass_at_k(n, min(c, n), min(k, n))
    if not problems:
        return {}, 0, short
    return {k: sums[k] / problems for k in ks}, problems, short


def find_results(roots):
    """Yield (search_root, path) for every math / sql results.jsonl under the roots: the
    file beside a summary.json, which names the environment ("env": "sql"; math writes none)."""
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            yield os.path.dirname(os.path.dirname(os.path.dirname(root))), root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("hf", "rollout_debug")]
            if "results.jsonl" in filenames and "summary.json" in filenames:
                yield root, os.path.join(dirpath, "results.jsonl")


def read_records(path):
    """The problems of one results.jsonl. The evals rewrite the file on a resume, but should
    several lines share an index, the last one is the problem's record, as the evals themselves
    read it; a record without an index is kept as it is."""
    records, at = [], {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (isinstance(rec, dict) and "traces" in rec):
                continue
            idx = rec.get("index")
            if isinstance(idx, int) and idx in at:
                records[at[idx]] = rec
                continue
            if isinstance(idx, int):
                at[idx] = len(records)
            records.append(rec)
    return records


def collect(roots, ks, timeouts):
    rows = []
    for root, path in find_results(roots):
        parts = os.path.relpath(path, root).split(os.sep)
        run, checkpoint = split_run_label(parts, root, 2)
        bench = parts[-2] if len(parts) > 1 else "?"
        try:
            with open(os.path.join(os.path.dirname(path), "summary.json")) as fh:
                summary = json.load(fh)
            env = summary.get("env") or "math"
            num_samples = summary.get("num_samples")
        except (OSError, ValueError):
            env, num_samples = "math", None
        if env not in ("math", "sql"):
            continue
        records = read_records(path)
        if not records:
            continue
        if num_samples is None:
            num_samples = max(len(r.get("traces") or []) for r in records)
        use_ks = [k for k in ks if k <= num_samples]
        values, problems, short = set_pass_at_k(records, use_ks, timeouts)
        rows.append({"run": run, "ckpt": checkpoint, "env": env, "bench": bench,
                     "n": int(num_samples), "problems": problems, "pass": values,
                     "short": max(short.values()) if short else 0})
    return rows


def mean_rows(rows, ks):
    """One MEAN row per (run, checkpoint, env) and distinct N: the sets with at least N
    samples, averaged at every k <= N."""
    groups = {}
    for r in rows:
        groups.setdefault((r["run"], r["ckpt"], r["env"]), []).append(r)
    out = []
    for (run, ckpt, env), sets in groups.items():
        for n in sorted({r["n"] for r in sets}):
            members = [r for r in sets if r["n"] >= n]
            if len(sets) == 1 and len({r["n"] for r in sets}) == 1:
                pass  # a single set still gets its MEAN row, like collect_results
            values = {}
            for k in ks:
                if k > n:
                    continue
                vals = [r["pass"][k] for r in members if k in r["pass"]]
                if len(vals) == len(members):
                    values[k] = sum(vals) / len(vals)
            label = f"MEAN of {len(members)}" + (f" (N>={n})" if len({r['n'] for r in sets}) > 1 else "")
            out.append({"run": run, "ckpt": ckpt, "env": env, "bench": label, "n": n,
                        "problems": sum(r["problems"] for r in members), "pass": values,
                        "short": max(r["short"] for r in members), "is_mean": True})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["runs"], help="directories to search (default: runs)")
    ap.add_argument("--run", action="append", default=[], help="only runs whose label contains this substring; repeatable")
    ap.add_argument("--bench", action="append", default=[], help="only these benchmarks; repeatable")
    ap.add_argument("--env", choices=["math", "sql"], help="only one environment's results")
    ap.add_argument("-k", "--ks", type=int, nargs="+", default=DEFAULT_KS, help="the k to report (default: 1 2 4 8 16 32)")
    ap.add_argument("--timeouts", choices=["exclude", "wrong"], default="exclude",
                    help="timed-out traces: left out of n (default, the accuracy's rule) or counted as incorrect")
    ap.add_argument("--no-mean", action="store_true", help="suppress the MEAN rows")
    ap.add_argument("--csv", help="also write every row in long format (one line per set and k)")
    args = ap.parse_args()
    ks = sorted(set(k for k in args.ks if k >= 1))
    if not ks:
        print("no k >= 1 given")
        return 1

    rows = collect(args.roots or ["runs"], ks, args.timeouts)
    if args.run:
        rows = [r for r in rows if any(f in r["run"] for f in args.run)]
    if args.bench:
        rows = [r for r in rows if r["bench"] in args.bench]
    if args.env:
        rows = [r for r in rows if r["env"] == args.env]
    if not rows:
        print(f"No math / sql results.jsonl found under: {', '.join(args.roots or ['runs'])}")
        return 1

    rows.sort(key=lambda r: (r["run"], natural_key(r["ckpt"]), env_key(r["env"]), bench_key(r["env"], r["bench"])))
    means = [] if args.no_mean else mean_rows(rows, ks)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "checkpoint", "env", "benchmark", "num_samples", "problems", "k", "pass_at_k", "short"])
            for r in rows + means:
                for k in ks:
                    if k in r["pass"]:
                        w.writerow([r["run"], r["ckpt"], r["env"], r["bench"], r["n"], r["problems"], k,
                                    f"{r['pass'][k]:.6f}", r["short"]])

    show_env = len({r["env"] for r in rows}) > 1
    w_run = max(max(len(r["run"]) for r in rows), len("run"))
    w_ckpt = max(max(len(r["ckpt"]) for r in rows), len("checkpoint"))
    w_env = max(max(len(r["env"]) for r in rows), len("env")) if show_env else 0
    w_bench = max(max(len(r["bench"]) for r in rows + means), len("benchmark"))
    head_k = " ".join(f"{'pass@' + str(k):>8}" for k in ks)
    width = w_run + w_ckpt + w_env + w_bench + 8 + 9 * len(ks) + 8 + (2 if show_env else 0)

    def line(r):
        cells = [f"{r['run']:<{w_run}}", f"{r['ckpt']:<{w_ckpt}}"]
        if show_env:
            cells.append(f"{r['env']:<{w_env}}")
        cells.append(f"{r['bench']:<{w_bench}}")
        ks_cells = " ".join(f"{100 * r['pass'][k]:>8.2f}" if k in r["pass"] else f"{'':>8}" for k in ks)
        short = f"{r['short']:>5}" if r["short"] else f"{'':>5}"
        return "  ".join(cells) + f"  {r['n']:>3}  " + ks_cells + "  " + short

    print()
    print("=" * width)
    cells = [f"{'run':<{w_run}}", f"{'checkpoint':<{w_ckpt}}"]
    if show_env:
        cells.append(f"{'env':<{w_env}}")
    cells.append(f"{'benchmark':<{w_bench}}")
    print("  ".join(cells) + f"  {'N':>3}  " + head_k + "  short")
    print("-" * width)
    last = None
    for r in rows:
        key = (r["run"], r["ckpt"], r["env"])
        if last is not None and key != last:
            for m in means:
                if (m["run"], m["ckpt"], m["env"]) == last:
                    print(line(m))
            print("-" * width)
        print(line(r))
        last = key
    for m in means:
        if (m["run"], m["ckpt"], m["env"]) == last:
            print(line(m))
    print("=" * width)
    print(f"pass@k in percent, the unbiased estimate over N samples; timed-out traces {args.timeouts}d; "
          f"'short': problems with fewer valid traces than the largest k (estimated at their own n).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
