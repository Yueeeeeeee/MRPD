"""tools/merge_rollout_logs.py on two tiny teacher dumps: the merged files' order, task tags,
score attributes, renumbering, rollout_id, the untouched reward, the manifest the launcher
reads, the default step count, the refusals (too many steps, an existing output, a
misaligned log-prob list, a block that is not one prompt group), and the two controls
(--groups-per-teacher, --shuffle-steps).
python3 tests/merge_rollout_logs_test.py (needs torch)."""
import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = f"{ROOT}/tools/merge_rollout_logs.py"
tmp = tempfile.mkdtemp()
N = 2


def dump(run, k, samples):
    os.makedirs(f"{tmp}/{run}/rollout_debug", exist_ok=True)
    torch.save({"rollout_id": k, "samples": samples}, f"{tmp}/{run}/rollout_debug/rollout_{k}.pt")


def sample(index, rl, reward, status="completed", metadata=None, group_index=None):
    return {"index": index, "group_index": index // N if group_index is None else group_index,
            "tokens": list(range(rl + 3)), "response_length": rl, "loss_mask": [1] * rl,
            "rollout_log_probs": [-0.5] * rl, "reward": reward, "status": status, "metadata": metadata,
            "rollout_id": None}


def run(*extra):
    return subprocess.run([sys.executable, TOOL, "--n-samples-per-prompt", str(N)] + list(extra), capture_output=True, text=True)


# teacher A: dict rewards (math's shape), no metadata, three steps of two groups; teacher B: float
# rewards (search's shape), a metadata dict, two steps
for k in range(3):
    dump("tA", k, [sample(k * 4 + i, 5 + i, {"score": 1.0 if i % 2 == 0 else -1.0, "pred": "x"},
                          status="truncated" if i == 3 else "completed") for i in range(4)])
for k in range(2):
    dump("tB", k, [sample(k * 4 + i, 3, -1.0 if i < 3 else 1.0, metadata={"x": 1, "step": k}) for i in range(4)])

out = f"{tmp}/merged"
res = run("--teacher", f"retool={tmp}/tA", "--teacher", f"search-r1={tmp}/tB", "--out", out, "--workers", "2")   # the pool path
assert res.returncode == 0, res.stderr
m = json.load(open(f"{out}/manifest.json"))
assert m["steps"] == 2 and m["samples_per_step"] == 8 and m["prompts_per_step"] == 4, m     # default: the steps every teacher has
assert m["n_samples_per_prompt"] == N and m["order"] == ["retool", "search-r1"], m
assert m["per_task_samples_per_step"] == {"retool": 4, "search-r1": 4}, m
assert m["groups_per_teacher"] is None and m["shuffle_seed"] is None and "step_order" not in m, m
assert sorted(os.listdir(out)) == ["manifest.json", "rollout_0.pt", "rollout_1.pt"], os.listdir(out)
for k in range(2):
    d = torch.load(f"{out}/rollout_{k}.pt", weights_only=False)
    s = d["samples"]
    assert d["rollout_id"] == k and len(s) == 8
    assert [x["metadata"]["task"] for x in s] == ["retool"] * 4 + ["search-r1"] * 4           # teacher order
    assert [x["index"] for x in s] == list(range(k * 8, k * 8 + 8)), [x["index"] for x in s]   # renumbered
    assert [x["group_index"] for x in s] == [i // N for i in range(k * 8, k * 8 + 8)]
    # rollout_id stays None: slime numbers every sample as its own rollout (its DP schedule counts
    # rollouts against the global batch; one shared id made the whole step a single rollout)
    assert all(x["rollout_id"] is None and x["_opd_task"] == x["metadata"]["task"] for x in s)
    assert [x["_opd_task_score"] for x in s] == [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0]
    assert [x["_opd_task_score_unshaped"] for x in s] == [x["_opd_task_score"] for x in s]
    assert s[0]["reward"] == {"score": 1.0, "pred": "x"} and s[4]["reward"] == -1.0               # untouched
    assert s[4]["metadata"] == {"x": 1, "step": k, "task": "search-r1"} and s[3]["status"] == "truncated"
    assert s[0]["rollout_log_probs"] == [-0.5] * 5 and s[0]["loss_mask"] == [1] * 5
assert "retool: 4 per step, mean score +0.000, truncated 25.0%" in res.stdout, res.stdout
assert "search-r1: 4 per step, mean score -0.500, truncated 0.0%" in res.stdout, res.stdout
# what the launcher reads off the manifest
line = subprocess.check_output([sys.executable, "-c", 'import json, sys\nm = json.load(open(sys.argv[1]))\n'
                                'print(m["steps"], m["prompts_per_step"], m["n_samples_per_prompt"], ",".join(m["order"]))',
                                f"{out}/manifest.json"], text=True).split()
assert line == ["2", "4", "2", "retool,search-r1"], line

# refusals
res = run("--teacher", f"retool={tmp}/tA", "--teacher", f"search-r1={tmp}/tB", "--out", f"{tmp}/m3", "--steps", "3")
assert res.returncode != 0 and "not enough consecutive rollout files" in res.stderr, res.stderr
res = run("--teacher", f"retool={tmp}/tA", "--out", out)
assert res.returncode != 0 and "exists" in res.stderr, res.stderr
bad = sample(1, 4, 1.0)
bad["rollout_log_probs"] = bad["rollout_log_probs"][:-1]
dump("tC", 0, [sample(0, 4, 1.0), bad])
res = run("--teacher", f"sql={tmp}/tC", "--out", f"{tmp}/bad")
assert res.returncode != 0 and "rollout_log_probs of length 3 for a response of 4" in res.stderr, res.stderr
dump("tD", 0, [sample(0, 4, 1.0), sample(1, 4, 1.0, group_index=7)])
res = run("--teacher", f"sql={tmp}/tD", "--out", f"{tmp}/bad2")
assert res.returncode != 0 and "not a whole prompt group" in res.stderr, res.stderr
# the controls: a subset of groups per teacher (the first G groups of every step), shuffled steps
res = run("--teacher", f"retool={tmp}/tA", "--teacher", f"search-r1={tmp}/tB", "--out", f"{tmp}/g1", "--groups-per-teacher", "1")
assert res.returncode == 0, res.stderr
m = json.load(open(f"{tmp}/g1/manifest.json"))
assert m["groups_per_teacher"] == 1 and m["samples_per_step"] == 4 and m["prompts_per_step"] == 2 and m["steps"] == 2, m
assert m["per_task_samples_per_step"] == {"retool": 2, "search-r1": 2}, m
for k in range(2):
    s = torch.load(f"{tmp}/g1/rollout_{k}.pt", weights_only=False)["samples"]
    assert [x["_opd_task"] for x in s] == ["retool"] * 2 + ["search-r1"] * 2
    assert [x["_opd_task_score"] for x in s] == [1.0, -1.0, -1.0, -1.0], [x["_opd_task_score"] for x in s]   # the first group of each
    assert [x["index"] for x in s] == list(range(k * 4, k * 4 + 4)) and [x["group_index"] for x in s] == [k * 2, k * 2, k * 2 + 1, k * 2 + 1]
assert ", the first 1 groups per teacher" in res.stdout, res.stdout
res = run("--teacher", f"retool={tmp}/tA", "--out", f"{tmp}/g3", "--groups-per-teacher", "3")
assert res.returncode != 0 and "2 groups, --groups-per-teacher asks for 3" in res.stderr, res.stderr
res = run("--teacher", f"retool={tmp}/tA", "--teacher", f"search-r1={tmp}/tB", "--out", f"{tmp}/shuf", "--shuffle-steps", "7", "--workers", "1")
assert res.returncode == 0, res.stderr
m = json.load(open(f"{tmp}/shuf/manifest.json"))
assert m["shuffle_seed"] == 7 and sorted(m["step_order"]) == [0, 1] and m["steps"] == 2 and m["groups_per_teacher"] is None, m
for k in range(2):
    s = torch.load(f"{tmp}/shuf/rollout_{k}.pt", weights_only=False)["samples"]
    assert all(x["metadata"]["step"] == m["step_order"][k] for x in s[4:]), "merged step k holds the teachers' step order[k]"
    assert [x["index"] for x in s] == list(range(k * 8, k * 8 + 8))                              # renumbered by the merged step
res2 = run("--teacher", f"retool={tmp}/tA", "--teacher", f"search-r1={tmp}/tB", "--out", f"{tmp}/shuf2", "--shuffle-steps", "7")
assert json.load(open(f"{tmp}/shuf2/manifest.json"))["step_order"] == m["step_order"], "the seed fixes the order"
assert ", steps shuffled with seed 7" in res.stdout, res.stdout
res = run("--teacher", f"retool={tmp}/tA", "--out", f"{tmp}/both", "--groups-per-teacher", "1", "--shuffle-steps", "3")
m = json.load(open(f"{tmp}/both/manifest.json"))
assert res.returncode == 0 and m["groups_per_teacher"] == 1 and m["shuffle_seed"] == 3 and sorted(m["step_order"]) == [0, 1, 2], m
# what the launchers read off the manifest for the run name
tag = subprocess.check_output([sys.executable, "-c", 'import json, sys\nm = json.load(open(sys.argv[1]))\n'
                               'tag = ("-g%d" % m["groups_per_teacher"] if m.get("groups_per_teacher") else "") + ("-shuf%d" % m["shuffle_seed"] if m.get("shuffle_seed") is not None else "")\n'
                               'print(tag or "none")', f"{tmp}/both/manifest.json"], text=True).strip()
assert tag == "-g1-shuf3", tag

# one teacher alone: a single-domain replay, three steps
res = run("--teacher", f"retool={tmp}/tA", "--out", f"{tmp}/math")
assert res.returncode == 0, res.stderr
m = json.load(open(f"{tmp}/math/manifest.json"))
assert m["steps"] == 3 and m["samples_per_step"] == 4 and m["order"] == ["retool"], m

print("merge_rollout_logs.py: order, tags, scores, renumbering, manifest, default steps, refusals, groups subset, shuffled steps, single teacher: OK")
