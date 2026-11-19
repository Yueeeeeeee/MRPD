"""python3 tests/replay_rft_test.py -- generate_multi_domain's REPLAY_LOSS=rft and RAFT=1 selection: which samples the SFT
loss keeps (the correct ones, at most RFT_MAX_PER_PROMPT per prompt; the experts' logged samples under REPLAY_LOSS=rft, the
student's own rollouts under RAFT=1, through the same hook) and the knobs' validation; the functions are lifted from the file
as grpo_reward_norm_test.py does (the module itself imports sglang / slime)."""
import ast, os
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train/src/generate_multi_domain.py")).read()
tree = ast.parse(src)
want = {"_replay_loss", "_rft_keep", "_raft"}
ns = {"os": os}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "gmd", "exec"), ns)
assert want <= set(ns), want - set(ns)
keep, loss = ns["_rft_keep"], ns["_replay_loss"]
T, F = True, False
R = [1, -1, -1, -1,   -1, -1, -1, -1,   1, 1, 1, -1]
assert keep(R, 4) == [T, F, F, F,  F, F, F, F,  T, T, T, F]          # every correct sample (the default)
assert keep(R, 4, 1) == [T, F, F, F,  F, F, F, F,  T, F, F, F]       # at most one per prompt
assert keep(R, 4, 2) == [T, F, F, F,  F, F, F, F,  T, T, F, F]
assert keep(R, 4, 8) == keep(R, 4)
# the cap counts correct samples, not positions: a wrong sample before the correct ones does not use it up
assert keep([-1, 1, 1, -1], 4, 1) == [F, T, F, F]
# a success is a positive reward (zero is not one); a trailing partial group is handled
assert keep([0.0, 0.5, 1.0, -1.0, 1.0], 4, None) == [F, T, T, F, T]
assert keep([], 4) == []
for bad in (0, -1):
    try:
        keep(R, 4, bad); raise SystemExit(f"cap {bad} accepted")
    except AssertionError:
        pass
os.environ.pop("REPLAY_LOSS", None); assert loss() == ""
os.environ["REPLAY_LOSS"] = "rft"; assert loss() == "rft"
os.environ["REPLAY_LOSS"] = " SFT "; assert loss() == "sft"
os.environ["REPLAY_LOSS"] = "grpo"
try:
    loss(); raise SystemExit("REPLAY_LOSS=grpo accepted")
except ValueError:
    pass
os.environ.pop("REPLAY_LOSS", None)
raft = ns["_raft"]
os.environ.pop("RAFT", None); assert raft() is False
os.environ["RAFT"] = "0"; assert raft() is False
os.environ["RAFT"] = " 1 "; assert raft() is True
for bad in ("yes", "2", "true"):
    os.environ["RAFT"] = bad
    try:
        raft(); raise SystemExit(f"RAFT={bad} accepted")
    except ValueError:
        pass
os.environ.pop("RAFT", None)
print("replay RFT: every correct sample kept by default, at most RFT_MAX_PER_PROMPT per prompt counted over the correct ones, zero is not a success; REPLAY_LOSS and RAFT validated: OK")

# --- the hook end to end: post_process_rewards on fake samples, the helpers lifted with it, slime's Sample and wandb stubbed ---
import sys, types
want2 = {"_metadata", "_get_task", "_use_task_reward", "_normalize_rewards", "post_process_rewards"}
names2 = {"TASKS", "METRIC_PREFIX", "TASK_ALIASES", "_SEEN_TASKS"}
class Sample:
    class Status:
        TRUNCATED = "truncated"
import torch
from typing import Any
ns.update({"Sample": Sample, "torch": torch, "Any": Any, "sys": sys, "_teacher_enabled": lambda args: False})
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want2:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "gmd", "exec"), ns)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if isinstance(target, ast.Name) and (target.id in names2 or target.id.startswith("_LAST_")):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "gmd", "exec"), ns)
assert want2 <= set(ns) and names2 <= set(ns), (want2 - set(ns), names2 - set(ns))
ppr = ns["post_process_rewards"]
logged = {}
fake_wandb = types.ModuleType("wandb"); fake_wandb.run = object(); fake_wandb.log = lambda m: logged.update(m)
sys.modules["wandb"] = fake_wandb
TASKS3 = ["retool"] * 4 + ["search-r1"] * 4 + ["sql"] * 4
def batch():
    return [types.SimpleNamespace(_opd_task_score=float(r), _opd_task=t, metadata={"task": t}, loss_mask=[1] * 5, response_length=5,
                                  effective_response_length=5, status=None, remove_sample=False, tool_call_count=0)
            for r, t in zip(R, TASKS3, strict=True)]
args = types.SimpleNamespace(advantage_estimator="grpo", n_samples_per_prompt=4, rollout_batch_size=3, rewards_normalization=True,
                             grpo_std_normalization=True, use_opd=False)
def run(**env):
    for k in ("REPLAY_LOSS", "RAFT", "RFT_MAX_PER_PROMPT", "MULTI_DOMAIN_USE_TASK_REWARD"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    logged.clear()
    s = batch()
    raw, train = ppr(args, s)
    return s, raw, train, dict(logged)
s0, raw0, train0, m0 = run()                                   # the GRPO replay arm: nothing removed, no RFT metric
assert raw0 == [float(x) for x in R] and not any(x.remove_sample for x in s0) and "rollout/rft_kept_ratio" not in m0
s1, raw1, train1, m1 = run(REPLAY_LOSS="rft")                  # RFT: the wrong samples flagged, rewards untouched
assert [x.remove_sample for x in s1] == [not k for k in keep(R, 4)], [x.remove_sample for x in s1]
assert raw1 == raw0 and train1 == train0
assert abs(m1["rollout/rft_kept_ratio"] - 4 / 12) < 1e-9 and abs(m1["rollout/retool_rft_kept_ratio"] - 1 / 4) < 1e-9
assert m1["rollout/search_r1_rft_kept_ratio"] == 0.0 and abs(m1["rollout/sql_rft_kept_ratio"] - 3 / 4) < 1e-9
s2, _, _, m2 = run(REPLAY_LOSS="rft", RFT_MAX_PER_PROMPT=1)    # at most one per prompt
assert [i for i, x in enumerate(s2) if not x.remove_sample] == [0, 8] and abs(m2["rollout/rft_kept_ratio"] - 2 / 12) < 1e-9
s3, _, _, m3 = run(REPLAY_LOSS="sft")                          # SFT on everything: the launcher's loss type does the work, nothing flagged
assert not any(x.remove_sample for x in s3) and "rollout/rft_kept_ratio" not in m3
try:
    run(REPLAY_LOSS="rft", MULTI_DOMAIN_USE_TASK_REWARD=0); raise SystemExit("rft with the task reward off accepted")
except ValueError as e:
    assert "MULTI_DOMAIN_USE_TASK_REWARD=0" in str(e), e
print("post_process_rewards under REPLAY_LOSS: rft flags remove_sample on the wrong samples (and beyond the cap), rewards and GRPO normalisation untouched, kept ratios logged overall and per domain; sft / unset flag nothing; rft without the task reward refused: OK")

# RAFT=1: the same selection on the student's own rollouts (the hook does not know where the samples came from)
s4, raw4, train4, m4 = run(RAFT=1)
assert [x.remove_sample for x in s4] == [x.remove_sample for x in s1] and raw4 == raw0 and train4 == train0
assert m4["rollout/rft_kept_ratio"] == m1["rollout/rft_kept_ratio"] and m4["rollout/sql_rft_kept_ratio"] == m1["rollout/sql_rft_kept_ratio"]
s5, _, _, m5 = run(RAFT=1, RFT_MAX_PER_PROMPT=1)                # RAFT's best-of-K: one correct sample per prompt
assert [i for i, x in enumerate(s5) if not x.remove_sample] == [0, 8]
s6, _, _, m6 = run(RAFT=0)                                      # RAFT=0 is the GRPO arm
assert not any(x.remove_sample for x in s6) and "rollout/rft_kept_ratio" not in m6
for env, needle in ((dict(RAFT=1, MULTI_DOMAIN_USE_TASK_REWARD=0), "MULTI_DOMAIN_USE_TASK_REWARD=0"),
                    (dict(RAFT=1, REPLAY_LOSS="rft"), "not both"), (dict(RAFT=1, REPLAY_LOSS="sft"), "not both")):
    try:
        run(**env); raise SystemExit(f"{env} accepted")
    except ValueError as e:
        assert needle in str(e), e
print("post_process_rewards under RAFT=1: the same samples flagged as REPLAY_LOSS=rft, the cap at 1 is RAFT's best-of-K, rewards untouched; RAFT=0 flags nothing; RAFT without the task reward or with REPLAY_LOSS refused: OK")
