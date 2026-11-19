"""The dense patch's loss.py on stubs: the patched get_log_probs_and_entropy (teacher side
`with_topk`, student side `topk_ids`) and policy_loss_function end to end, against autograd on the
full log-softmax. The submodule's pristine loss.py is copied to a temp dir and patched with the
decouple and dense patches' loss.py hunks (git apply --include); megatron and slime's runtime are
stubbed; slime's ppo_utils and cp_utils are the real ones. Context parallel 1, tensor parallel 1
(the tensor-parallel path is tests/teacher_dense_kl_test.py's two-thread check).
python3 tests/dense_loss_patch_test.py (needs torch and git)."""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
from argparse import Namespace

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIME = f"{ROOT}/third_party/slime"
REL = "slime/backends/megatron_utils/loss.py"
sys.path.insert(0, f"{ROOT}/train/src")

# ---- the patched loss.py in a temp tree ---------------------------------------------------------
tmp = tempfile.mkdtemp()
os.makedirs(f"{tmp}/slime/backends/megatron_utils")
shutil.copy(f"{SLIME}/{REL}", f"{tmp}/{REL}")
for p in ("slime-opd-decouple-student-logprobs", "slime-opd-dense-kl"):
    subprocess.run(["git", "apply", f"--include={REL}", f"{ROOT}/patches/{p}.patch"], cwd=tmp, check=True)
src = open(f"{tmp}/{REL}").read()
assert "slime-opd-dense-kl-fused" in src and "slime-opd-dense-kl:" not in src, "the fused version, not the first"

# ---- stubs: megatron mpu, slime's runtime; the real ppo_utils and cp_utils ----------------------
mpu = types.SimpleNamespace(
    get_context_parallel_world_size=lambda: 1,
    get_context_parallel_rank=lambda: 0,
    get_tensor_model_parallel_group=lambda: None,
    get_tensor_model_parallel_rank=lambda: 0,
    get_tensor_model_parallel_world_size=lambda: 1,
)
megatron = types.ModuleType("megatron")
core = types.ModuleType("megatron.core")
core.mpu = mpu
sys.modules["megatron"], sys.modules["megatron.core"] = megatron, core
slime = types.ModuleType("slime")
su = types.ModuleType("slime.utils")
lu = types.ModuleType("slime.utils.logging_utils")
lu.log = lambda *a, **k: None
du = types.ModuleType("slime.utils.distributed_utils")
du.distributed_masked_whiten = None
mi = types.ModuleType("slime.utils.misc")
mi.load_function = lambda p: None
ty = types.ModuleType("slime.utils.types")
ty.RolloutBatch = dict
sb = types.ModuleType("slime.backends")
sbm = types.ModuleType("slime.backends.megatron_utils")
sbm.__path__ = []
sys.modules.update({"slime": slime, "slime.utils": su, "slime.utils.logging_utils": lu, "slime.utils.distributed_utils": du,
                    "slime.utils.misc": mi, "slime.utils.types": ty, "slime.backends": sb, "slime.backends.megatron_utils": sbm})


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pu = load("slime.utils.ppo_utils", f"{SLIME}/slime/utils/ppo_utils.py")
cpu = load("slime.backends.megatron_utils.cp_utils", f"{SLIME}/slime/backends/megatron_utils/cp_utils.py")
loss_mod = load("slime.backends.megatron_utils.loss", f"{tmp}/{REL}")
import teacher_dense_kl as m  # noqa: E402


def close(a, b, tol=1e-5, what=""):
    err = (a - b).abs().max().item() if a.numel() else 0.0
    assert err <= tol, f"{what}: max abs diff {err} > {tol}"


# ---- the micro-batch ----------------------------------------------------------------------------
torch.manual_seed(0)
V, K = 48, 6
totals, resps = [7, 10, 5], [5, 6, 3]
T = sum(totals) + 2  # padding rows at the end
tokens = [torch.randint(0, V, (t,)) for t in totals]
student = torch.randn(T, V) * 2
teacher = torch.randn(T, V) * 2
rows = m.response_rows(totals, resps)
args = Namespace(
    rollout_temperature=1.0, log_probs_chunk_size=-1, entropy_coef=0.0, allgather_cp=False, rollout_top_p=1.0,
    use_rollout_logprobs=True, advantage_estimator="grpo", eps_clip=0.2, eps_clip_high=0.2, use_opsm=False,
    get_mismatch_metrics=False, use_tis=False, custom_pg_loss_reducer_function_path=None, calculate_per_token_loss=False,
    use_kl_loss=False, opd_dense_kl="reverse", opd_dense_kl_coef=0.7, opd_teacher_topk=K,
)
common = dict(args=args, unconcat_tokens=tokens, total_lengths=totals, response_lengths=resps)


def ref_log_probs(x):
    """log-softmax at the logged response tokens, per sample, as slime defines them (row r predicts token r + 1)."""
    return [F.log_softmax(x[a:b], -1).gather(1, tok[t - r :][:, None]).squeeze(1) for (a, b), tok, t, r in zip(rows, tokens, totals, resps)]


# ---- 1. the teacher side: with_topk ---------------------------------------------------------------
_, res_t = loss_mod.get_log_probs_and_entropy(teacher[None], with_entropy=False, with_topk=K, **common)
ref_t_lp, ref_t_ids = F.log_softmax(teacher, -1).topk(K, dim=-1)
assert len(res_t["topk_ids"]) == len(res_t["topk_log_probs"]) == len(res_t["log_probs"]) == 3
for (a, b), ids, lp, lp_tok, r_tok in zip(rows, res_t["topk_ids"], res_t["topk_log_probs"], res_t["log_probs"], ref_log_probs(teacher)):
    assert ids.dtype == torch.int32 and ids.shape == (b - a, K) and torch.equal(ids.long(), ref_t_ids[a:b]), "teacher top-k ids"
    close(lp, ref_t_lp[a:b], what="teacher top-k log-probs")
    close(lp_tok, r_tok, what="teacher log-probs at the logged tokens")
assert "entropy" not in res_t
print("1 teacher side (with_topk): ids, log-probs, slime's own log-probs: OK")

# ---- 2. the student side: topk_ids -----------------------------------------------------------------
x = student.clone().requires_grad_(True)
_, res_s = loss_mod.get_log_probs_and_entropy(x[None], with_entropy=True, topk_ids=res_t["topk_ids"], **common)
assert set(res_s) == {"log_probs", "entropy", "topk_log_probs"}, set(res_s)
w = [torch.randn(r, K) for r in resps]
for (a, b), lp_tok, r_tok, tk, ids, ent in zip(rows, res_s["log_probs"], ref_log_probs(student), res_s["topk_log_probs"], res_t["topk_ids"], res_s["entropy"]):
    close(lp_tok, r_tok, what="student log-probs at the logged tokens")
    ls = F.log_softmax(student[a:b], -1)
    close(tk, ls.gather(1, ids.long()), what="student log-probs at the teacher's ids")
    close(ent, -(ls.exp() * ls).sum(-1), what="entropy")
    assert tk.requires_grad and not ent.requires_grad, "the top-k log-probs carry the gradient; the entropy (coef 0) does not"
(g,) = torch.autograd.grad(sum((wi * tk).sum() for wi, tk in zip(w, res_s["topk_log_probs"])) + sum(lp.sum() for lp in res_s["log_probs"]), x)
x_ref = student.clone().requires_grad_(True)
(g_ref,) = torch.autograd.grad(
    sum((wi * F.log_softmax(x_ref[a:b], -1).gather(1, ids.long())).sum() for wi, (a, b), ids in zip(w, rows, res_t["topk_ids"]))
    + sum(t.sum() for t in ref_log_probs(x_ref)),
    x_ref,
)
close(g, g_ref, what="gradient through the patched function")
assert g[-2:].abs().max() == 0 and g[0].abs().max() == 0, "no gradient on the padding rows / a prompt row that predicts a prompt token"
try:
    loss_mod.get_log_probs_and_entropy(x[None], with_entropy=True, with_topk=K, topk_ids=res_t["topk_ids"], **common)
    raise SystemExit("both sides at once were not refused")
except AssertionError:
    pass
print("2 student side (topk_ids): log-probs, top-k log-probs, entropy, gradient = autograd; both sides refused: OK")

# ---- 3. policy_loss_function end to end ---------------------------------------------------------------
adv = [torch.randn(r) for r in resps]
rollout_lp = [t.detach() - 0.3 * torch.randn(r) for t, r in zip(ref_log_probs(student), resps)]  # the dump's log-probs, near the student's
masks = [torch.ones(r) for r in resps]
masks[1][-2:] = 0  # two masked tokens
batch = {
    "advantages": adv, "rollout_log_probs": rollout_lp, "loss_masks": masks, "response_lengths": resps, "total_lengths": totals,
    "unconcat_tokens": tokens, "teacher_log_probs": res_t["log_probs"], "teacher_topk_ids": res_t["topk_ids"],
    "teacher_topk_log_probs": res_t["topk_log_probs"],
}
ssm = cpu.get_sum_of_sample_mean(totals, resps, masks)
x = student.clone().requires_grad_(True)
total, rep = loss_mod.policy_loss_function(args, batch, x[None], ssm)
# the reference: slime's clipped loss on the reference log-probs, plus the (k+1)-atom reverse KL
x_ref = student.clone().requires_grad_(True)
lp_r = torch.cat(ref_log_probs(x_ref))
pg, clipfrac = pu.compute_policy_loss(torch.cat(rollout_lp) - lp_r, torch.cat(adv), 0.2, 0.2)
kl_r = torch.cat([m.dense_kl(F.log_softmax(x_ref[a:b], -1).gather(1, ids.long()), tlp, "reverse")
                  for (a, b), ids, tlp in zip(rows, res_t["topk_ids"], res_t["topk_log_probs"])])
loss_ref = ssm(pg) + 0.7 * ssm(kl_r)
close(total, loss_ref, what="loss = pg + 0.7 * dense")
close(rep["loss"], loss_ref, what="reported loss")
close(rep["pg_loss"], ssm(pg), what="pg_loss")
close(rep["pg_clipfrac"], ssm(clipfrac), what="pg_clipfrac")
close(rep["teacher_dense_kl"], ssm(kl_r), what="teacher_dense_kl (before its coefficient)")
hit = torch.cat([(ids.long() == tok[t - r :][:, None]).any(-1).float() for ids, tok, t, r in zip(res_t["topk_ids"], tokens, totals, resps)])
close(rep["teacher_topk_hit"], ssm(hit), what="teacher_topk_hit")
assert 0 < rep["teacher_topk_hit"] < 1, rep["teacher_topk_hit"]
assert rep["teacher_dense_err_student"] < 1e-6 and rep["teacher_dense_err_teacher"] < 1e-5, (rep["teacher_dense_err_student"], rep["teacher_dense_err_teacher"])
(g,) = torch.autograd.grad(total, x)
(g_ref,) = torch.autograd.grad(loss_ref, x_ref)
close(g, g_ref, what="gradient of the whole loss")
assert g[-2:].abs().max() == 0, "padding rows"
print("3 policy_loss_function: loss, pg metrics, the four dense metrics, gradient = autograd reference: OK")

# ---- 4. the knob off: slime's own path, no dense outputs; on without the top-k: refused -------------
args.opd_dense_kl = None
x0 = student.clone().requires_grad_(True)
total0, rep0 = loss_mod.policy_loss_function(args, batch, x0[None], ssm)
close(total0, ssm(pg), what="loss without the term")
assert not any(k.startswith("teacher_") for k in rep0), sorted(rep0)
(g0,) = torch.autograd.grad(total0, x0)
x0r = student.clone().requires_grad_(True)
(g0r,) = torch.autograd.grad(ssm(pu.compute_policy_loss(torch.cat(rollout_lp) - torch.cat(ref_log_probs(x0r)), torch.cat(adv), 0.2, 0.2)[0]), x0r)
close(g0, g0r, what="gradient without the term")
args.opd_dense_kl = "reverse"
b2 = dict(batch)
del b2["teacher_topk_ids"]
try:
    loss_mod.policy_loss_function(args, b2, student.clone().requires_grad_(True)[None], ssm)
    raise SystemExit("a batch without the teacher's top-k was not refused")
except ValueError as e:
    assert "teacher_topk_ids" in str(e), e
print("4 knob off = slime's own path; knob on without the top-k refused: OK")

shutil.rmtree(tmp, ignore_errors=True)

# ---- 5. compute_advantages_and_returns: no sampled-token term and no opd_reverse_kl when the student's ----
# standalone forward was skipped (the term at 0; actor.py's _student_log_probs_consumed); with the forward,
# or with the term on, as before (the fallback to the rollout log-probs included)
loss_mod.mpu.is_pipeline_last_stage = lambda: True


def adv_args(coef):
    return Namespace(advantage_estimator="grpo", kl_coef=0, use_rollout_logprobs=True, custom_advantage_function_path=None,
                     normalize_advantages=False, use_opd=True, opd_kl_coef=coef, opd_type="megatron", kl_loss_type="low_var_kl")


def adv_data(with_student):
    r = [3, 2]
    d = {"rewards": [1.0, -1.0], "response_lengths": r, "total_lengths": [5, 4], "loss_masks": [torch.ones(n, dtype=torch.int) for n in r],
         "rollout_log_probs": [torch.randn(n) for n in r], "teacher_log_probs": [torch.randn(n) for n in r]}
    if with_student:
        d["log_probs"] = [torch.randn(n) for n in r]
    return d


d = adv_data(False)
loss_mod.compute_advantages_and_returns(adv_args(0.0), d)
assert "opd_reverse_kl" not in d, "no student forward, the term at 0: nothing applied, nothing stored"
assert all(torch.equal(a, torch.full((n,), w)) for a, n, w in zip(d["advantages"], [3, 2], [1.0, -1.0], strict=True))
d = adv_data(True)
loss_mod.compute_advantages_and_returns(adv_args(0.0), d)
assert all(torch.equal(k, s - t) for k, s, t in zip(d["opd_reverse_kl"], d["log_probs"], d["teacher_log_probs"], strict=True)), "the student's forward: stored"
assert all(torch.equal(a, torch.full((n,), w)) for a, n, w in zip(d["advantages"], [3, 2], [1.0, -1.0], strict=True)), "at 0: the advantages untouched"
d = adv_data(False)
loss_mod.compute_advantages_and_returns(adv_args(1.0), d)
assert all(torch.allclose(a, w - (s - t)) for a, w, s, t in zip(d["advantages"], [1.0, -1.0], d["rollout_log_probs"], d["teacher_log_probs"], strict=True)), "the term on without the forward: the fallback, as before"
print("5 compute_advantages_and_returns: the sampled-token term and opd_reverse_kl only when read or on: OK")

print("dense patch loss.py: teacher side, student side, policy loss end to end, knob off / refusal: OK")
