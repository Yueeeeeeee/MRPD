"""python3 tests/grpo_reward_norm_test.py -- generate_multi_domain._normalize_rewards: the functions are lifted from the file (the
module itself imports sglang / slime) and run against torch."""
import ast, os, types
import torch
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train/src/generate_multi_domain.py")).read()
tree = ast.parse(src)
want = {"_use_task_reward", "_normalize_rewards"}
ns = {"torch": torch, "os": os}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "gmd", "exec"), ns)
assert want <= set(ns), want - set(ns)
norm = ns["_normalize_rewards"]
def args(est="grpo", n=4, b=3, std=True, opd=True):
    return types.SimpleNamespace(advantage_estimator=est, n_samples_per_prompt=n, rollout_batch_size=b,
                                 rewards_normalization=True, grpo_std_normalization=std, use_opd=opd)
R = [1, -1, -1, -1,   -1, -1, -1, -1,   1, 1, 1, -1]
def setenv(**kw):
    os.environ.pop("MULTI_DOMAIN_USE_TASK_REWARD", None)
    os.environ.update({k: str(v) for k, v in kw.items()})
def close(a, b): return all(abs(x - y) < 1e-5 for x, y in zip(a, b)) and len(a) == len(b)
# GRPO: centred within the prompt's group of 4 and divided by the group's unbiased std + 1e-6
setenv(MULTI_DOMAIN_USE_TASK_REWARD=1)
t = torch.tensor(R, dtype=torch.float).reshape(-1, 4); t = t - t.mean(-1, keepdim=True); t = t / (t.std(-1, keepdim=True) + 1e-6)
out = norm(args(), [float(x) for x in R])
assert close(out, t.flatten().tolist()), out
# every group is centred; a group whose samples all fail (or all succeed) gets zero advantages
assert all(abs(sum(out[4*g:4*g+4])) < 1e-4 for g in range(3)), out
assert out[4:8] == [0.0] * 4, out
# without the std normalisation: centred only
c = torch.tensor(R, dtype=torch.float).reshape(-1, 4); c = c - c.mean(-1, keepdim=True)
assert close(norm(args(std=False), [float(x) for x in R]), c.flatten().tolist())
# an estimator without group normalisation: the rewards are returned as they are
assert norm(args(est="ppo"), [float(x) for x in R]) == [float(x) for x in R]
# the task reward off (distillation only): zeros with --use-opd, refused without it
setenv(MULTI_DOMAIN_USE_TASK_REWARD=0)
assert norm(args(), [float(x) for x in R]) == [0.0] * 12
try:
    norm(args(opd=False), [float(x) for x in R]); raise SystemExit("zero reward accepted without --use-opd")
except ValueError as e:
    assert "MULTI_DOMAIN_USE_TASK_REWARD=0" in str(e), e
print("GRPO reward normalisation: centred and divided by the group std, all-fail groups zero, plain centring without std, other estimators untouched, zero reward only with OPD: OK")
