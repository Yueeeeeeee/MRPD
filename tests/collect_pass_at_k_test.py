#!/usr/bin/env python3
"""scripts/eval/collect_pass_at_k.py on a fake eval tree: the estimator against brute-force
enumeration, the timeout rules, pass@1 = the summary's accuracy, the labels, the MEAN rows, the CSV and a results.jsonl with one index on several
lines (the last one counts).

    python3 tests/collect_pass_at_k_test.py
"""

import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))
import collect_pass_at_k as cp  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "eval", "collect_pass_at_k.py")


def brute_force(n, c, k):
    """The share of k-subsets of n traces (c correct) that hold a correct one."""
    traces = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(1 for s in subsets if any(traces[i] for i in s)) / len(subsets)


def test_estimator():
    for n in range(1, 9):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                got, want = cp.pass_at_k(n, c, k), brute_force(n, c, k)
                assert abs(got - want) < 1e-12, (n, c, k, got, want)
    assert cp.pass_at_k(32, 1, 1) == 1 / 32
    assert cp.pass_at_k(32, 0, 32) == 0.0
    assert cp.pass_at_k(32, 1, 32) == 1.0
    print("estimator: ok")


def record(accs, timed_out=()):
    traces = [{"trace_index": i, "acc": bool(a), "score": 1.0 if a else 0.0, "timed_out": i in timed_out}
              for i, a in enumerate(accs)]
    valid = [t for t in traces if not t["timed_out"]]
    return {
        "traces": traces,
        "avg_acc_excl_timeout": (sum(t["acc"] for t in valid) / len(valid)) if valid else 0.0,
    }


def write_set(ckpt_dir, dataset, records, env=None):
    d = os.path.join(ckpt_dir, dataset)
    os.makedirs(d)
    with open(os.path.join(d, "results.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    n = max(len(r["traces"]) for r in records)
    summary = {"accuracy": sum(r["avg_acc_excl_timeout"] for r in records) / len(records),
               "num_samples": n, "timeout_stats": {"timeout_rate": 0.0}}
    if env:
        summary["env"] = env
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump(summary, fh)
    return summary["accuracy"]


def test_tree():
    with tempfile.TemporaryDirectory() as tmp:
        runs = os.path.join(tmp, "runs")
        run_a = os.path.join(runs, "mrpd", "iter_0000299")
        acc_aime = write_set(run_a, "aime24", [
            record([1, 0, 0, 0]),                 # c = 1 of 4
            record([0, 0, 0, 0]),                 # none
            record([1, 1, 0, 0], timed_out={3}),  # one timed out: n = 3, c = 2
            record([0, 0, 0, 0], timed_out={0, 1, 2, 3}),  # all timed out: 0 at every k
        ])
        acc_math500 = write_set(run_a, "math500", [record([1, 1, 0, 0]), record([0, 1, 0, 0])])
        grpo = os.path.join(runs, "grpo_3d", "iter_0000299")
        acc_sql = write_set(grpo, "spider_dev", [record([1, 0, 1, 0]), record([0, 0, 0, 1])], env="sql")
        # A search summary beside its jsonl must not be picked up (no results.jsonl there).
        with open(os.path.join(grpo, "eval_search_r1_summary.json"), "w") as fh:
            json.dump({"per_data_source": {"nq": {"accuracy": 0.5}}}, fh)

        rows = cp.collect([runs], [1, 2, 4, 8], "exclude")
        by = {(r["run"], r["bench"]): r for r in rows}
        assert set(by) == {("mrpd", "aime24"), ("mrpd", "math500"), ("grpo_3d", "spider_dev")}, set(by)
        assert by[("grpo_3d", "spider_dev")]["env"] == "sql" and by[("mrpd", "aime24")]["env"] == "math"
        assert all(r["ckpt"] == "iter_0000299" for r in rows)

        # pass@1 is the accuracy (timed-out traces left out, the all-timed-out problem 0).
        a = by[("mrpd", "aime24")]
        assert abs(a["pass"][1] - acc_aime) < 1e-12, (a["pass"][1], acc_aime)
        assert abs(by[("mrpd", "math500")]["pass"][1] - acc_math500) < 1e-12
        assert abs(by[("grpo_3d", "spider_dev")]["pass"][1] - acc_sql) < 1e-12
        # pass@4 of aime24: problem 1 -> 1, problem 2 -> 0, problem 3 (n = 3 < 4: at its own n) -> 1, problem 4 -> 0.
        assert abs(a["pass"][4] - 0.5) < 1e-12, a["pass"][4]
        assert a["short"] == 2, a["short"]          # problems 3 and 4 have fewer than 4 valid traces
        assert 8 not in a["pass"]                   # k above N is not reported
        # pass@2 of aime24 by hand: 0.5, 0, 1 - C(1,2)/C(3,2) = 1, 0 -> mean 0.375.
        assert abs(a["pass"][2] - 0.375) < 1e-12, a["pass"][2]

        # --timeouts wrong: n stays 4, the timed-out traces count as incorrect.
        rows_w = cp.collect([runs], [1, 4], "wrong")
        aw = {(r["run"], r["bench"]): r for r in rows_w}[("mrpd", "aime24")]
        assert abs(aw["pass"][1] - (0.25 + 0 + 0.5 + 0) / 4) < 1e-12, aw["pass"][1]
        assert abs(aw["pass"][4] - (1 + 0 + 1 + 0) / 4) < 1e-12
        assert aw["short"] == 0

        # MEAN rows: one per (run, checkpoint, env); math has one N here, so no "(N>=...)" tag.
        means = cp.mean_rows(rows, [1, 2, 4, 8])
        m = {(r["run"], r["env"]): r for r in means}
        assert m[("mrpd", "math")]["bench"] == "MEAN of 2", m[("mrpd", "math")]["bench"]
        want = (a["pass"][1] + by[("mrpd", "math500")]["pass"][1]) / 2
        assert abs(m[("mrpd", "math")]["pass"][1] - want) < 1e-12
        assert m[("grpo_3d", "sql")]["bench"] == "MEAN of 1"

        # Two distinct N in one environment: a MEAN over all sets up to the smaller N and one
        # over the 32-sample sets up to 32.
        write_set(run_a, "amc23", [record([1] + [0] * 31), record([0] * 32)])
        rows2 = cp.collect([runs], [1, 4, 32], "exclude")
        means2 = [r for r in cp.mean_rows(rows2, [1, 4, 32]) if r["run"] == "mrpd"]
        labels = {r["bench"]: r for r in means2}
        assert set(labels) == {"MEAN of 3 (N>=4)", "MEAN of 1 (N>=32)"}, set(labels)
        assert set(labels["MEAN of 3 (N>=4)"]["pass"]) == {1, 4}
        assert set(labels["MEAN of 1 (N>=32)"]["pass"]) == {1, 4, 32}
        assert abs(labels["MEAN of 1 (N>=32)"]["pass"][32] - 0.5) < 1e-12

        # The command line: the table, the filters and the CSV.
        out_csv = os.path.join(tmp, "p.csv")
        proc = subprocess.run([sys.executable, SCRIPT, runs, "--env", "math", "-k", "1", "4", "--csv", out_csv],
                              capture_output=True, text=True, check=True)
        assert "mrpd" in proc.stdout and "grpo_3d" not in proc.stdout, proc.stdout
        assert "MEAN of 3 (N>=4)" in proc.stdout and "pass@4" in proc.stdout, proc.stdout
        assert "actor" not in proc.stdout.split("pass@k in percent")[0]
        with open(out_csv) as fh:
            lines = list(csv.DictReader(fh))
        sets = {(l["benchmark"], l["k"]) for l in lines if not l["benchmark"].startswith("MEAN")}
        assert sets == {(b, k) for b in ("aime24", "math500", "amc23") for k in ("1", "4")}, sets
        aime1 = [l for l in lines if l["benchmark"] == "aime24" and l["k"] == "1"][0]
        assert abs(float(aime1["pass_at_k"]) - acc_aime) < 1e-6 and aime1["num_samples"] == "4"

        proc = subprocess.run([sys.executable, SCRIPT, runs, "--run", "nothing_here"], capture_output=True, text=True)
        assert proc.returncode == 1 and "No math / sql results.jsonl" in proc.stdout

        # Several lines with one index (the evals' resume rule: the last one is the record).
        dup = os.path.join(grpo, "bird_minidev")
        os.makedirs(dup)
        with open(os.path.join(dup, "results.jsonl"), "w") as fh:
            fh.write(json.dumps(dict(record([0, 0, 0, 0], timed_out={0, 1, 2, 3}), index=0)) + "\n")
            fh.write(json.dumps(dict(record([1, 0, 0, 0]), index=1)) + "\n")
            fh.write(json.dumps(dict(record([1, 1, 0, 0]), index=0)) + "\n")   # the re-run of problem 0
        with open(os.path.join(dup, "summary.json"), "w") as fh:
            json.dump({"accuracy": 0.375, "num_samples": 4, "env": "sql"}, fh)
        recs = cp.read_records(os.path.join(dup, "results.jsonl"))
        assert [r["index"] for r in recs] == [0, 1] and recs[0]["avg_acc_excl_timeout"] == 0.5
        bird = {(r["run"], r["bench"]): r for r in cp.collect([runs], [1, 4], "exclude")}[("grpo_3d", "bird_minidev")]
        assert bird["problems"] == 2 and abs(bird["pass"][1] - 0.375) < 1e-12 and bird["short"] == 0
    print("tree: ok")


if __name__ == "__main__":
    test_estimator()
    test_tree()
    print("all ok")
