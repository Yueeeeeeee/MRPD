"""Unit tests of train/src/teacher_dense_kl.py (the dense teacher term of the distillation arms):
the teacher's top-k, the fused log-prob / top-k function of the policy loss (values and gradients
against autograd on the full log-softmax, both entropy modes), the (k+1)-atom KLs and their bounds,
the gradient identities behind the k1 / k2 / k3 estimators, the micro-batch glue with its
diagnostics, and the tensor-parallel path run on two threads through a fake group against the
single-shard numbers.
python3 tests/teacher_dense_kl_test.py (needs torch; no megatron, no GPU)."""
import math
import os
import sys
import threading

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/train/src")
import teacher_dense_kl as m  # noqa: E402

torch.manual_seed(0)
TP1 = m.TPGroup()
assert (TP1.rank, TP1.size) == (0, 1)


def close(a, b, tol=1e-4, what=""):
    err = (a - b).abs().max().item() if a.numel() else 0.0
    assert err <= tol, f"{what}: max abs diff {err} > {tol}"


# ---- 1. teacher top-k (one shard) ---------------------------------------------------------------
T, V, K = 7, 40, 5
logits = torch.randn(T, V) * 3
ids, lp = m.topk_log_probs(logits, K, TP1, chunk_rows=3)
ref_lp, ref_ids = F.log_softmax(logits, -1).topk(K, dim=-1)
assert ids.dtype == torch.int32 and torch.equal(ids.long(), ref_ids), "top-k ids"
close(lp, ref_lp, 1e-5, "top-k log-probs")
close(m.tail_log_prob(lp), torch.log1p(-ref_lp.exp().sum(-1)), 1e-5, "tail")
e_ids, e_lp = m.topk_log_probs(logits[:0], K, TP1)
assert e_ids.shape == (0, K) and e_lp.shape == (0, K)
ids_bf, lp_bf = m.topk_log_probs(logits.to(torch.bfloat16), K, TP1, chunk_rows=3)  # a bf16 shard is upcast per chunk
assert lp_bf.dtype == torch.float32 and ids_bf.shape == (T, K)
print("1 teacher top-k: OK")

# ---- 2. the fused function: log-prob at the target, log-probs at the ids, entropy; the gradients --
target = torch.randint(0, V, (T,))
ids_l = ids.long().clone()
ids_l[0, 2] = target[0]  # one row where the target is among the ids, so both scatters hit one column
w_lp, w_k, w_e = torch.randn(T), torch.randn(T, K), torch.randn(T)


def reference(x, with_entropy_grad):
    ls = F.log_softmax(x, -1)
    lp_ref = ls.gather(1, target[:, None]).squeeze(1)
    tk_ref = ls.gather(1, ids_l)
    ent_ref = -(ls.exp() * ls).sum(-1)
    loss = (w_lp * lp_ref).sum() + (w_k * tk_ref).sum() + ((w_e * ent_ref).sum() if with_entropy_grad else 0.0)
    return lp_ref, tk_ref, ent_ref, loss


for with_entropy_grad in (False, True):
    x = logits.clone().requires_grad_(True)
    lp_f, tk_f, ent_f = m.log_probs_and_topk(x, target, ids_l, TP1, with_entropy=True, with_entropy_grad=with_entropy_grad)
    assert lp_f.shape == (T, 1) and tk_f.shape == (T, K) and ent_f.shape == (T,), (lp_f.shape, tk_f.shape, ent_f.shape)
    assert ent_f.requires_grad == with_entropy_grad, "entropy differentiable only when asked for"
    lp_ref, tk_ref, ent_ref, loss_ref = reference(logits.clone().requires_grad_(True), with_entropy_grad)
    close(lp_f.squeeze(1), lp_ref, 1e-5, "fused log-prob at the target")
    close(tk_f, tk_ref, 1e-5, "fused log-probs at the ids")
    close(ent_f, ent_ref, 1e-5, "fused entropy")
    loss_f = (w_lp * lp_f.squeeze(1)).sum() + (w_k * tk_f).sum() + ((w_e * ent_f).sum() if with_entropy_grad else 0.0)
    (g_f,) = torch.autograd.grad(loss_f, x)
    x_ref = logits.clone().requires_grad_(True)
    (g_ref,) = torch.autograd.grad(reference(x_ref, with_entropy_grad)[3], x_ref)
    close(g_f, g_ref, 1e-5, f"fused gradient (entropy grad {with_entropy_grad})")
# the reference gather through autograd (student_log_probs_at) agrees too
x = logits.clone().requires_grad_(True)
got = m.student_log_probs_at(x, ids_l, TP1)
(g_got,) = torch.autograd.grad((w_k * got).sum(), x)
x2 = logits.clone().requires_grad_(True)
(g_ref2,) = torch.autograd.grad((w_k * F.log_softmax(x2, -1).gather(1, ids_l)).sum(), x2)
close(got, F.log_softmax(logits, -1).gather(1, ids_l), 1e-5, "reference gather")
close(g_got, g_ref2, 1e-5, "reference gradient")
# no entropy asked for: None back; the top-k output unused in the loss: its gradient is zero, the log-prob path intact
x = logits.clone().requires_grad_(True)
lp_f, tk_f, ent_f = m.log_probs_and_topk(x, target, ids_l, TP1)
assert ent_f is None
(g_f,) = torch.autograd.grad((w_lp * lp_f.squeeze(1)).sum(), x)
x_ref = logits.clone().requires_grad_(True)
(g_ref,) = torch.autograd.grad((w_lp * F.log_softmax(x_ref, -1).gather(1, target[:, None]).squeeze(1)).sum(), x_ref)
close(g_f, g_ref, 1e-5, "log-prob path alone")
lp_e, tk_e, ent_e = m.log_probs_and_topk(logits[:0], target[:0], ids_l[:0], TP1, with_entropy=True)
assert lp_e.shape == (0, 1) and tk_e.shape == (0, K) and ent_e.shape == (0,)
print("2 fused log-prob / top-k / entropy: values and gradients = autograd on the full log-softmax: OK")

# ---- 3. the (k+1)-atom KLs: exact at k = V, bounds and monotonicity below, JS properties ---------
# In double: the tail is 1 - sum of the kept atoms, and with fp32 atoms that difference carries the
# atoms' rounding (a 1e-5 tail from 1 - 0.99999 is only known to ~1%), which is what the real run
# lives with (the tail atom is a floor, TAIL_EPS, not a precise quantity); the identities themselves
# are exact, so they are checked in exact arithmetic.
ls_full, lt_full = F.log_softmax(torch.randn(T, V).double(), -1), F.log_softmax(torch.randn(T, V).double() * 2, -1)
ps, pt = ls_full.exp(), lt_full.exp()
exact = {"reverse": (ps * (ls_full - lt_full)).sum(-1), "forward": (pt * (lt_full - ls_full)).sum(-1)}
lm_full = torch.logaddexp(ls_full, lt_full) - math.log(2)
exact["js"] = 0.5 * ((ps * (ls_full - lm_full)).sum(-1) + (pt * (lt_full - lm_full)).sum(-1))
order = lt_full.argsort(-1, descending=True)  # the teacher's ranking, as topk_log_probs orders it
for kind in m.KINDS:
    full = m.dense_kl(ls_full.gather(1, order), lt_full.gather(1, order), kind)
    close(full, exact[kind], 1e-5, f"{kind} at k = V")  # the tail floor TAIL_EPS is all that is left
    prev = torch.zeros(T, dtype=torch.float64)
    for k in (1, 2, 5, 10, 20, 39):
        trunc = m.dense_kl(ls_full.gather(1, order[:, :k]), lt_full.gather(1, order[:, :k]), kind)
        assert (trunc >= -1e-12).all(), f"{kind} k={k} negative"
        assert (trunc <= exact[kind] + 1e-9).all(), f"{kind} k={k} above the exact value"
        assert (trunc >= prev - 1e-9).all(), f"{kind} k={k} below k smaller"
        prev = trunc
js_ab = m.dense_kl(ls_full.gather(1, order[:, :K]), lt_full.gather(1, order[:, :K]), "js")
js_ba = m.dense_kl(lt_full.gather(1, order[:, :K]), ls_full.gather(1, order[:, :K]), "js")
close(js_ab, js_ba, 1e-12, "JS symmetric")
assert (exact["js"] <= math.log(2) + 1e-12).all()
same = ls_full.gather(1, order[:, :K])
for kind in m.KINDS:
    assert m.dense_kl(same, same, kind).abs().max() < 1e-12, f"{kind} not zero on equal distributions"
print("3 KLs: exact at k = V, 0 <= truncated <= exact, monotone in k, JS symmetric and <= log 2: OK")

# ---- 4. the gradient identities (one prefix, k = V): what each estimator's expected gradient is --
theta = torch.randn(V, requires_grad=True)
lt = F.log_softmax(torch.randn(V) * 2, -1)
order1 = lt.argsort(descending=True)


def grad_of(loss):
    (g,) = torch.autograd.grad(loss, theta)
    return g


lp = F.log_softmax(theta, -1)
p = lp.detach().exp()
dense_rev = grad_of(m.dense_kl(lp[order1][None], lt[order1][None], "reverse").sum())
lp = F.log_softmax(theta, -1)
dense_fwd = grad_of(m.dense_kl(lp[order1][None], lt[order1][None], "forward").sum())
lp = F.log_softmax(theta, -1)
# k1 as the advantage (slime's term): E_y[(log pi - log pi_T)(y) grad log pi(y)], the score-function
# estimator of grad KL(pi || pi_T); the advantage is detached, the expectation taken exactly over y
k1_adv = grad_of((p * (lp.detach() - lt) * lp).sum())
close(k1_adv, dense_rev, 1e-5, "k1-as-advantage expected gradient = dense reverse KL gradient")
# k2 in the loss: E_y[grad 1/2 (log pi - log pi_T)^2(y)], the sampling distribution not differentiated
lp = F.log_softmax(theta, -1)
k2_loss = grad_of((p * 0.5 * (lp - lt) ** 2).sum())
close(k2_loss, dense_rev, 1e-5, "k2-in-loss expected gradient = dense reverse KL gradient")
# k3 in the loss (GRPO's low_var_kl): E_y[grad (exp(r) - r - 1)], r = log pi_T - log pi, same convention
lp = F.log_softmax(theta, -1)
r = lt - lp
k3_loss = grad_of((p * (r.exp() - r - 1)).sum())
close(k3_loss, dense_fwd, 1e-5, "k3-in-loss expected gradient = dense FORWARD KL gradient")
# and k1 in the loss: E_y[grad (log pi - log pi_T)(y)] = E_y[grad log pi(y)] = 0
lp = F.log_softmax(theta, -1)
k1_loss = grad_of((p * (lp - lt)).sum())
assert k1_loss.abs().max() < 1e-6, "k1-in-loss expected gradient is not zero"
print("4 gradient identities: k1-as-advantage = k2-in-loss = reverse, k3-in-loss = forward, k1-in-loss = 0: OK")

# ---- 5. the micro-batch glue, as the patched policy loss runs it: rows, the full layouts, diagnostics --
totals, resps = [6, 9], [4, 5]
Tm = sum(totals) + 3  # padding rows at the end, as slime's get_batch pads
logits_mb = torch.randn(Tm, V) * 2
teacher_mb = torch.randn(Tm, V) * 2
tokens = [torch.randint(0, V, (t,)) for t in totals]
rows = m.response_rows(totals, resps)
assert rows == [(1, 5), (9, 14)], rows
# slime's shifted targets: row r of a sample predicts its token r + 1; 0 on the padding rows
full_tokens = torch.zeros(Tm, dtype=torch.long)
off = 0
for tok, total in zip(tokens, totals):
    full_tokens[off : off + total - 1] = tok[1:total]
    off += total
# the teacher's pass: top-k per sample from its own logits at the response rows, and slime's own
# log-probs at the logged tokens (student and teacher)
t_ids, t_lp = m.topk_log_probs(teacher_mb, K, TP1)
topk_ids = [t_ids[a:b].clone() for a, b in rows]
topk_lp = [t_lp[a:b].clone() for a, b in rows]
s_ref = [F.log_softmax(logits_mb[a:b], -1).gather(1, tok[t - r :][:, None]).squeeze(1) for (a, b), tok, t, r in zip(rows, tokens, totals, resps)]
t_ref = [F.log_softmax(teacher_mb[a:b], -1).gather(1, tok[t - r :][:, None]).squeeze(1) for (a, b), tok, t, r in zip(rows, tokens, totals, resps)]
# the student side: the ids laid out like the logits, the fused function, the per-sample slices
ids_full = m.rows_to_full(topk_ids, totals, resps, Tm, dtype=torch.long)
assert ids_full.shape == (Tm, K) and ids_full.dtype == torch.long and ids_full[5:9].abs().sum() == 0 and ids_full[14:].abs().sum() == 0
assert torch.equal(ids_full[1:5], topk_ids[0].long()) and torch.equal(ids_full[9:14], topk_ids[1].long())
x = logits_mb.clone().requires_grad_(True)
lp_full, topk_full, ent_full = m.log_probs_and_topk(x, full_tokens, ids_full, TP1, with_entropy=True)
lp_list = [lp_full.squeeze(1)[a:b] for a, b in rows]
close(torch.cat(lp_list), torch.cat(s_ref), 1e-5, "the policy loss's log-probs at the logged tokens")
out = m.dense_kl_for_batch([topk_full[a:b] for a, b in rows], topk_ids, topk_lp, "reverse", tokens=tokens, total_lengths=totals,
                           response_lengths=resps, student_log_probs=lp_list, teacher_log_probs=t_ref)
assert out["kl"].shape == (sum(resps),) and out["hit"].shape == (sum(resps),)
direct = torch.cat([m.dense_kl(F.log_softmax(logits_mb[a:b], -1).gather(1, i.long()), l, "reverse")
                    for (a, b), i, l in zip(rows, topk_ids, topk_lp)])
close(out["kl"], direct, 1e-5, "glue kl = per-sample kl")
(g,) = torch.autograd.grad(out["kl"].sum(), x)
assert g[5:9].abs().max() == 0 and g[14:].abs().max() == 0 and g[0].abs().max() == 0, "gradient on non-response rows"
assert g[1:5].abs().max() > 0 and g[9:14].abs().max() > 0
x_ref = logits_mb.clone().requires_grad_(True)
(g_ref,) = torch.autograd.grad(torch.cat([m.dense_kl(F.log_softmax(x_ref[a:b], -1).gather(1, i.long()), l, "reverse")
                                          for (a, b), i, l in zip(rows, topk_ids, topk_lp)]).sum(), x_ref)
close(g, g_ref, 1e-5, "gradient of the term through the fused function = autograd")
assert out["err_student"].max() < 1e-6 and out["err_teacher"].max() < 1e-5, "diagnostics against the functions' own values"
hit_ref = torch.cat([(i.long() == tok[t - r :][:, None]).any(-1).float() for i, tok, t, r in zip(topk_ids, tokens, totals, resps)])
assert torch.equal(out["hit"], hit_ref) and 0 < hit_ref.mean() < 1, "hit fraction"
# k = V: every token hits; teacher = student: zero KL
v_ids, v_lp = m.topk_log_probs(logits_mb, V, TP1)
ids_v = m.rows_to_full([v_ids[a:b] for a, b in rows], totals, resps, Tm, dtype=torch.long)
_, topk_v, _ = m.log_probs_and_topk(logits_mb, full_tokens, ids_v, TP1)
out = m.dense_kl_for_batch([topk_v[a:b] for a, b in rows], [v_ids[a:b] for a, b in rows], [v_lp[a:b] for a, b in rows], "js",
                           tokens=tokens, total_lengths=totals, response_lengths=resps, student_log_probs=s_ref, teacher_log_probs=s_ref)
assert out["hit"].min() == 1 and out["kl"].abs().max() < 1e-5 and out["err_student"].max() < 1e-5
# temperature: slime divides the logits before the call; the term then sits at that temperature on both sides
t_ids2, t_lp2 = m.topk_log_probs(teacher_mb / 0.7, K, TP1)
ids_t = m.rows_to_full([t_ids2[a:b] for a, b in rows], totals, resps, Tm, dtype=torch.long)
_, topk_t, _ = m.log_probs_and_topk(logits_mb / 0.7, full_tokens, ids_t, TP1)
out_t = m.dense_kl_for_batch([topk_t[a:b] for a, b in rows], [t_ids2[a:b] for a, b in rows], [t_lp2[a:b] for a, b in rows], "forward")
direct_t = torch.cat([m.dense_kl(F.log_softmax(logits_mb[a:b] / 0.7, -1).gather(1, t_ids2[a:b].long()), t_lp2[a:b], "forward") for a, b in rows])
close(out_t["kl"], direct_t, 1e-5, "temperature")
assert "hit" not in out_t
print("5 micro-batch glue: rows, full layouts, fused log-probs, gradient only on response rows, diagnostics, k = V, temperature: OK")


# ---- 6. tensor parallel on two threads: the shards' numbers and gradients equal the single shard's --
class FakeTP(m.TPGroup):
    """`size` ranks on `size` threads; every collective meets at a barrier and sees all slots."""

    def __init__(self, rank, size, shared):
        self.group, self.rank, self.size, self.shared = None, rank, size, shared

    def _exchange(self, x):
        sh = self.shared
        sh["slots"][self.rank] = x.detach().clone()
        sh["barrier"].wait()
        out = [s.clone() for s in sh["slots"]]
        sh["barrier"].wait()
        return out

    def all_reduce_max_(self, x):
        x.copy_(torch.stack(self._exchange(x)).max(0).values)

    def all_reduce_sum_(self, x):
        x.copy_(torch.stack(self._exchange(x)).sum(0))

    def all_gather(self, x):
        return self._exchange(x)


TP = 2
assert V % TP == 0
shared = {"barrier": threading.Barrier(TP), "slots": [None] * TP}
vl = V // TP
results, errors = [None] * TP, []
w_full = torch.randn(Tm)  # a weight on the log-prob path, so the shared buffer carries both gradients
w_ref = torch.randn(Tm, K)  # a weight on the reference gather's output


def worker(rank):
    try:
        tp = FakeTP(rank, TP, shared)
        shard_t = teacher_mb[:, rank * vl : (rank + 1) * vl].contiguous()
        ids_r, lp_r = m.topk_log_probs(shard_t, K, tp, chunk_rows=5)
        ids_full_r = m.rows_to_full([ids_r[a:b] for a, b in rows], totals, resps, Tm, dtype=torch.long)
        x_r = logits_mb[:, rank * vl : (rank + 1) * vl].contiguous().requires_grad_(True)
        lp_f, topk_f, ent_f = m.log_probs_and_topk(x_r, full_tokens, ids_full_r, tp, with_entropy=True)
        out_r = m.dense_kl_for_batch([topk_f[a:b] for a, b in rows], [ids_r[a:b] for a, b in rows], [lp_r[a:b] for a, b in rows], "reverse",
                                     tokens=tokens, total_lengths=totals, response_lengths=resps,
                                     student_log_probs=[lp_f.squeeze(1)[a:b] for a, b in rows], teacher_log_probs=t_ref)
        (g_r,) = torch.autograd.grad((w_full * lp_f.squeeze(1)).sum() + out_r["kl"].sum(), x_r)
        # the reference gather under TP as well
        x_r2 = logits_mb[:, rank * vl : (rank + 1) * vl].contiguous().requires_grad_(True)
        ref_r = m.student_log_probs_at(x_r2, ids_full_r, tp)
        (g_ref_r,) = torch.autograd.grad((w_ref * ref_r).sum(), x_r2)
        results[rank] = (ids_r, lp_r, lp_f.detach(), topk_f.detach(), ent_f.detach(), out_r, g_r, ref_r.detach(), g_ref_r)
    except Exception as e:  # noqa: BLE001
        errors.append((rank, repr(e)))
        shared["barrier"].abort()


threads = [threading.Thread(target=worker, args=(r,)) for r in range(TP)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert not errors, errors
ids_1, lp_1 = m.topk_log_probs(teacher_mb, K, TP1)
ids_full_1 = m.rows_to_full([ids_1[a:b] for a, b in rows], totals, resps, Tm, dtype=torch.long)
x1 = logits_mb.clone().requires_grad_(True)
lp_1f, topk_1f, ent_1f = m.log_probs_and_topk(x1, full_tokens, ids_full_1, TP1, with_entropy=True)
out_1 = m.dense_kl_for_batch([topk_1f[a:b] for a, b in rows], [ids_1[a:b] for a, b in rows], [lp_1[a:b] for a, b in rows], "reverse",
                             tokens=tokens, total_lengths=totals, response_lengths=resps,
                             student_log_probs=[lp_1f.squeeze(1)[a:b] for a, b in rows], teacher_log_probs=t_ref)
(g_1,) = torch.autograd.grad((w_full * lp_1f.squeeze(1)).sum() + out_1["kl"].sum(), x1)
x1b = logits_mb.clone().requires_grad_(True)
ref_1 = m.student_log_probs_at(x1b, ids_full_1, TP1)
(g_ref_1,) = torch.autograd.grad((w_ref * ref_1).sum(), x1b)
for rank in range(TP):
    ids_r, lp_r, lp_f, topk_f, ent_f, out_r, g_r, ref_r, g_ref_r = results[rank]
    sl = slice(rank * vl, (rank + 1) * vl)
    assert torch.equal(ids_r, ids_1), f"rank {rank}: top-k ids differ from the single shard"
    close(lp_r, lp_1, 1e-5, f"rank {rank} top-k log-probs")
    close(lp_f, lp_1f, 1e-5, f"rank {rank} fused log-prob at the target")
    close(topk_f, topk_1f, 1e-5, f"rank {rank} fused log-probs at the ids")
    close(ent_f, ent_1f, 1e-5, f"rank {rank} fused entropy")
    close(out_r["kl"], out_1["kl"], 1e-5, f"rank {rank} dense kl")
    close(g_r, g_1[:, sl], 1e-5, f"rank {rank} gradient of its shard (log-prob path + dense term)")
    close(ref_r, ref_1, 1e-5, f"rank {rank} reference gather")
    close(g_ref_r, g_ref_1[:, sl], 1e-5, f"rank {rank} reference gradient of its shard")
    assert out_r["err_student"].max() < 1e-6 and out_r["err_teacher"].max() < 1e-5 and torch.equal(out_r["hit"], out_1["hit"])
print("6 tensor parallel (2 threads, fake group): top-k, fused values, kl, gradients per shard, diagnostics = single shard: OK")

print("teacher_dense_kl.py: top-k, fused log-prob/top-k, KLs, gradient identities, batch glue, TP path: OK")
