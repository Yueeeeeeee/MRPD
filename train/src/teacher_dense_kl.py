"""Dense (logit-level) teacher term for the distillation arms: the KL between the student's and
the teacher's next-token distributions at every logged position, on the teacher's top-k.

slime's own teacher term (apply_opd_kl_to_advantages, tinker-cookbook's train_on_policy) is the
one-sample estimate: at every response position the advantage is log pi_T(y_t) - log pi_theta(y_t)
at the token that was sampled, through the clipped PG loss, which in expectation over y_t ~ pi_theta
is the gradient of KL(pi_theta || pi_T) at that prefix. This module computes that KL (or the
forward KL, or JS) from the distributions themselves:

  teacher side, in the teacher's forward-only pass (no grad):  topk_log_probs
      the k largest teacher probabilities at every position, as global vocabulary ids and
      log-probs, over the tensor-parallel vocabulary shards;
  student side, in the policy loss (differentiable through this rank's shard):  log_probs_and_topk
      slime's vocabulary-parallel log-prob at the logged token and the student's log-probs at the
      teacher's k ids, out of one fused autograd function (_LogProbTopK); then the KL between the
      two (k+1)-atom distributions -- the k atoms plus one tail atom holding the rest of each
      side's mass (dense_kl, dense_kl_for_batch).

The (k+1)-atom KL is the KL between coarsenings of the two distributions: a lower bound of the
full-vocabulary KL, exact when k covers the vocabulary, >= 0, and monotone in k. It is
differentiated directly in the loss: no importance ratio and no clip (the quantity is exact at the
logged prefix; what the ratio corrected in the sampled-token term was the one-token estimate).

Tensor parallelism: the vocabulary is sharded over the TP group, rank r holding columns
[r * V_local, (r + 1) * V_local) of the logits. TPGroup wraps the collectives -- max / sum
all-reduces for the log-sum-exp, an all-gather for the top-k merge, and one differentiable sum
(sum_differentiable: all-reduce in the forward, identity in the backward, the rule of Megatron's
reduce_from_tensor_model_parallel_region: every TP rank computes the same loss, so every rank's
shard takes the gradient of that one loss unchanged). A size-1 group skips every collective;
tests/teacher_dense_kl_test.py runs the size-2 path on two threads through a fake group and
checks it against the single-shard numbers.

Memory: the policy loss's log-prob at the logged token and the log-probs at the k ids come out of
one autograd function, slime's _VocabParallelLogProbEntropy (slime/utils/ppo_utils.py) with the
extra gather: one [T, V_local] fp32 buffer per micro-batch, the softmax in the forward,
overwritten in place into the gradient in the backward, on the whole micro-batch at once (slime's
default; row chunks would hand autograd a second full buffer to concatenate the chunks' gradients
into). The term therefore costs the policy loss no full-vocabulary tensor beyond the one it has
without it. The first version (2026-10-31) took the student side through autograd on the logits
on top of slime's function (student_log_probs_at, kept as the reference the tests compare
against) -- a second full buffer plus chunk temporaries -- and ran out of memory on a
16.6k-token micro-batch (one max-length math sample). The teacher side (topk_log_probs, no grad)
works in row chunks.

Used by patches/slime-opd-dense-kl.patch (loss.py, actor.py); knobs TEACHER_DENSE_KL /
TEACHER_DENSE_KL_COEF / TEACHER_TOPK in scripts/train/opd_multi.sh.
"""
import math

import torch
import torch.distributed as dist

KINDS = ("reverse", "forward", "js")
TAIL_EPS = 1e-6  # the tail atom's probability is clamped here: below ~1e-7 fp32 cannot tell 1 - s from 1
CHUNK_ROWS = 2048  # rows of the logits per [rows, V_local] temporary of the teacher side


class TPGroup:
    """Collectives over the tensor-parallel group; size 1 makes every one a no-op.

    `group` is a torch.distributed process group (rank() / size(), as slime's
    _get_vocab_parallel_rank_size reads it), or None for the default group / a single process."""

    def __init__(self, group=None):
        self.group = group
        if group is not None and hasattr(group, "rank") and hasattr(group, "size"):
            self.rank, self.size = group.rank(), group.size()
        elif dist.is_available() and dist.is_initialized():
            self.rank, self.size = dist.get_rank(group=group), dist.get_world_size(group=group)
        else:
            self.rank, self.size = 0, 1

    def all_reduce_max_(self, x: torch.Tensor) -> None:
        if self.size > 1:
            dist.all_reduce(x, op=dist.ReduceOp.MAX, group=self.group)

    def all_reduce_sum_(self, x: torch.Tensor) -> None:
        if self.size > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)

    def all_gather(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.size == 1:
            return [x]
        out = [torch.empty_like(x) for _ in range(self.size)]
        dist.all_gather(out, x.contiguous(), group=self.group)
        return out

    def sum_differentiable(self, x: torch.Tensor) -> torch.Tensor:
        """Sum over the TP ranks, gradient passed through unchanged (the loss is replicated)."""
        return x if self.size == 1 else _SumOverTP.apply(x, self)


class _SumOverTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        y = x.clone()
        tp.all_reduce_sum_(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def log_sum_exp(logits: torch.Tensor, tp: TPGroup) -> torch.Tensor:
    """Row-wise log-sum-exp over the whole vocabulary of a [rows, V_local] shard; differentiable."""
    lse = torch.logsumexp(logits, dim=-1)
    if tp.size == 1:
        return lse
    m = lse.detach().clone()
    tp.all_reduce_max_(m)
    s = tp.sum_differentiable((lse - m).exp())
    return m + s.log()


@torch.no_grad()
def topk_log_probs(logits: torch.Tensor, k: int, tp: TPGroup, chunk_rows: int = CHUNK_ROWS):
    """The k largest probabilities of every row of a [T, V_local] shard, over the whole vocabulary.

    Returns (ids [T, k] int32: global vocabulary ids, descending probability; log_probs [T, k] fp32).
    k <= V_local is required (each rank contributes its own top-k before the merge)."""
    assert logits.dim() == 2, logits.shape
    v_local = logits.size(1)
    assert 1 <= k <= v_local, f"top-k of {k} on a vocabulary shard of {v_local}"
    if logits.size(0) == 0:
        return logits.new_zeros((0, k), dtype=torch.int32), logits.new_zeros((0, k), dtype=torch.float32)
    offset = tp.rank * v_local
    ids_out, lp_out = [], []
    for chunk in torch.split(logits, chunk_rows, dim=0):
        chunk = chunk.float()
        lse = log_sum_exp(chunk, tp)
        vals, idx = chunk.topk(k, dim=-1)
        idx = idx + offset
        if tp.size > 1:
            vals = torch.cat(tp.all_gather(vals), dim=-1)
            idx = torch.cat(tp.all_gather(idx), dim=-1)
            vals, pos = vals.topk(k, dim=-1)
            idx = idx.gather(-1, pos)
        ids_out.append(idx.to(torch.int32))
        lp_out.append(vals - lse[:, None])
    return torch.cat(ids_out), torch.cat(lp_out)


class _LogProbTopK(torch.autograd.Function):
    """slime's _VocabParallelLogProbEntropy plus the log-probs at k ids per row, one buffer.

    forward(logits [T, V_local] fp32, target [T] long, ids [T, k] long, tp, with_entropy, with_entropy_grad)
      -> log_prob [T, 1] at the target, topk_log_probs [T, k] at the ids, entropy [T] (empty unless
         with_entropy; a value only, not differentiable, unless with_entropy_grad);
    target and ids are global vocabulary ids. The shard's softmax is built in one [T, V_local]
    buffer (logits - max, exp_, div_); the two gathers are taken from the normalized logits
    before the exp_, zero off-shard, summed over TP. The backward overwrites that buffer with
      -p (g_lp + sum_j g_j) + onehot(target) g_lp + sum_j onehot(id_j) g_j
    on this rank's columns (plus the entropy gradient when asked for): the gradient of the
    replicated loss through this shard, no further reduction, as slime's function returns it."""

    @staticmethod
    def forward(ctx, logits, target, ids, tp, with_entropy, with_entropy_grad):
        with_entropy_grad = bool(with_entropy and with_entropy_grad)
        logits = logits.float()
        rows, v_local = logits.shape
        start = tp.rank * v_local
        t_off = (target < start) | (target >= start + v_local)
        t_local = (target - start).masked_fill(t_off, 0)
        i_off = (ids < start) | (ids >= start + v_local)
        i_local = (ids - start).masked_fill(i_off, 0)
        arange = torch.arange(rows, device=logits.device)

        m = logits.max(dim=-1, keepdim=True).values
        tp.all_reduce_max_(m)
        buf = logits - m  # the one [rows, V_local] buffer: normalized logits, exp, softmax, then the gradient
        pred = buf[arange, t_local].masked_fill(t_off, 0.0)  # [rows]
        at_ids = buf.gather(1, i_local).masked_fill_(i_off, 0.0)  # [rows, k]
        buf.exp_()
        s = buf.sum(dim=-1, keepdim=True)
        tp.all_reduce_sum_(s)
        buf.div_(s)  # the softmax over the whole vocabulary, this shard's columns
        log_s = s.log()  # [rows, 1]
        tp.all_reduce_sum_(pred)
        tp.all_reduce_sum_(at_ids)
        log_prob = pred.unsqueeze(-1) - log_s
        topk_lp = at_ids - log_s

        entropy = logits.new_zeros((0,))
        sxl = logits.new_zeros((0,))
        if with_entropy:
            # sum_v p_v logit_v without a [rows, V_local] product buffer on CUDA, as slime does it
            sxl = torch.einsum("ij,ij->i", buf, logits) if buf.is_cuda else (buf * logits).sum(dim=-1)
            tp.all_reduce_sum_(sxl)
            entropy = m.squeeze(-1) + log_s.squeeze(-1) - sxl
        if not with_entropy_grad:
            ctx.mark_non_differentiable(entropy)
        ctx.with_entropy_grad = with_entropy_grad
        empty = logits.new_zeros((0,))
        ctx.save_for_backward(buf, t_off, t_local, i_off, i_local, logits if with_entropy_grad else empty, sxl if with_entropy_grad else empty)
        return log_prob, topk_lp, entropy

    @staticmethod
    def backward(ctx, g_lp, g_topk, g_ent):
        p, t_off, t_local, i_off, i_local, logits, sxl = ctx.saved_tensors
        grad_ent = None
        if ctx.with_entropy_grad and g_ent is not None and g_ent.numel() > 0:
            grad_ent = sxl.unsqueeze(-1) - logits  # d entropy / d logit = p (E_p[logit] - logit)
            grad_ent.mul_(p).mul_(g_ent.reshape(-1, 1))
        g_lp = g_lp.reshape(-1, 1)
        g_sum = g_topk.sum(dim=-1, keepdim=True)
        p.mul_(-(g_lp + g_sum))  # the softmax terms of both log-probs, in place
        arange = torch.arange(p.size(0), device=p.device)
        p[arange, t_local] += g_lp.squeeze(-1) * (~t_off).to(p.dtype)
        p.scatter_add_(1, i_local, g_topk.masked_fill(i_off, 0.0))
        if grad_ent is not None:
            p.add_(grad_ent)
        return p, None, None, None, None, None


def log_probs_and_topk(logits, tokens, ids, tp: TPGroup, with_entropy: bool = False, with_entropy_grad: bool = False):
    """The policy loss's log-probs of a micro-batch: at the logged tokens and at the teacher's ids.

    logits [T, V_local] fp32 (grad), tokens [T] (the shifted targets: row r's next token), ids [T, k]
    (global ids) -> (log_prob [T, 1], topk_log_probs [T, k], entropy [T] or None): the shapes slime's
    calculate_log_probs_and_entropy returns, plus the [T, k]. One call on the whole micro-batch."""
    assert logits.dim() == 2 and ids.dim() == 2 and tokens.dim() == 1, (logits.shape, ids.shape, tokens.shape)
    assert ids.size(0) == logits.size(0) == tokens.size(0), (logits.shape, ids.shape, tokens.shape)
    if logits.size(0) == 0:
        return logits.new_zeros((0, 1)), logits.new_zeros((0, ids.size(1))), (logits.new_zeros((0,)) if with_entropy else None)
    log_prob, topk_lp, entropy = _LogProbTopK.apply(logits, tokens.long(), ids.long(), tp, with_entropy, with_entropy_grad)
    return log_prob, topk_lp, (entropy if with_entropy else None)


def student_log_probs_at(logits: torch.Tensor, ids: torch.Tensor, tp: TPGroup) -> torch.Tensor:
    """log pi_theta at the global ids [rows, k] from this rank's [rows, V_local] shard, through autograd.

    The reference the tests hold log_probs_and_topk against; not the training path (its backward
    builds a second full-vocabulary buffer, the reason the first version ran out of memory)."""
    v_local = logits.size(1)
    start = tp.rank * v_local
    local = (ids >= start) & (ids < start + v_local)
    li = (ids - start).clamp(0, v_local - 1)
    g = logits.gather(1, li) * local.to(logits.dtype)  # 0 off-shard, no gradient there
    g = tp.sum_differentiable(g)
    return g - log_sum_exp(logits, tp)[:, None]


def tail_log_prob(log_probs: torch.Tensor) -> torch.Tensor:
    """log of the mass outside the k atoms, [rows, k] -> [rows]; clamped at TAIL_EPS."""
    s = log_probs.exp().sum(dim=-1).clamp(max=1.0 - TAIL_EPS)
    return torch.log1p(-s)


def dense_kl(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, kind: str) -> torch.Tensor:
    """KL between the (k+1)-atom distributions, per row: [rows, k] x [rows, k] -> [rows].

    reverse: KL(student || teacher) -- what the sampled-token term estimates;
    forward: KL(teacher || student); js: Jensen-Shannon, symmetric, <= log 2."""
    ls = torch.cat([student_log_probs, tail_log_prob(student_log_probs)[:, None]], dim=-1)
    lt = torch.cat([teacher_log_probs, tail_log_prob(teacher_log_probs)[:, None]], dim=-1)
    if kind == "reverse":
        return (ls.exp() * (ls - lt)).sum(dim=-1)
    if kind == "forward":
        return (lt.exp() * (lt - ls)).sum(dim=-1)
    if kind == "js":
        lm = torch.logaddexp(ls, lt) - math.log(2.0)
        return 0.5 * ((ls.exp() * (ls - lm)).sum(dim=-1) + (lt.exp() * (lt - lm)).sum(dim=-1))
    raise ValueError(f"dense KL kind must be one of {KINDS}, got {kind!r}")


def response_rows(total_lengths, response_lengths) -> list[tuple[int, int]]:
    """Per sample, the [first, last) rows of the micro-batch logits that predict its response tokens
    (context parallel 1: the samples concatenated in order, row r predicting token r + 1)."""
    rows, offset = [], 0
    for total, resp in zip(total_lengths, response_lengths, strict=True):
        end = offset + total
        rows.append((end - resp - 1, end - 1))
        offset = end
    return rows


def rows_to_full(per_sample, total_lengths, response_lengths, T: int, dtype=None, fill=0) -> torch.Tensor:
    """The per-sample [L_i, ...] tensors placed at their response rows of a [T, ...] tensor whose
    other rows (the prompts', the padding) hold `fill` -- the teacher's ids laid out like the
    micro-batch logits for log_probs_and_topk."""
    rows = response_rows(total_lengths, response_lengths)
    assert per_sample and len(per_sample) == len(rows), (len(per_sample), len(rows))
    assert all(0 <= a <= b <= T for a, b in rows), (rows, T)
    first = per_sample[0]
    full = torch.full((T, *first.shape[1:]), fill, dtype=dtype or first.dtype, device=first.device)
    for (a, b), x in zip(rows, per_sample, strict=True):
        assert x.shape[0] == b - a and x.shape[1:] == first.shape[1:], (x.shape, b - a, first.shape)
        full[a:b] = x
    return full


def dense_kl_for_batch(
    student_topk_log_probs,
    teacher_topk_ids,
    teacher_topk_log_probs,
    kind: str,
    tokens=None,
    total_lengths=None,
    response_lengths=None,
    student_log_probs=None,
    teacher_log_probs=None,
) -> dict[str, torch.Tensor]:
    """The dense teacher term of one micro-batch from its per-sample [L_i, k] tensors: the student's
    log-probs at the teacher's ids (log_probs_and_topk, grad), the teacher's ids (int32) and log-probs
    (its forward). Returns per-token tensors of length sum(L_i), in sample order, for
    sum_of_sample_mean:
      kl            the (k+1)-atom KL of `kind` (differentiable);
    and, when tokens (per sample [total_i], the logged tokens), total_lengths, response_lengths,
    student_log_probs and teacher_log_probs (per sample [L_i], the log-probs at the logged tokens
    from the student's and the teacher's own functions) are given, the detached diagnostics
      hit           1 where the logged token is among the teacher's top-k,
      err_student   |the student's top-k log-prob - its log-prob at the logged token| (on hits, else 0),
      err_teacher   |the teacher's top-k log-prob - its log-prob at the logged token| (on hits, else 0):
    the gathers' index and mask paths against the two functions' own values at the same token
    (the teacher's two numbers come from different code, topk_log_probs vs slime's function)."""
    assert kind in KINDS, kind
    assert len(student_topk_log_probs) == len(teacher_topk_ids) == len(teacher_topk_log_probs)
    for s, i, t in zip(student_topk_log_probs, teacher_topk_ids, teacher_topk_log_probs, strict=True):
        assert s.shape == i.shape == t.shape and s.dim() == 2, (s.shape, i.shape, t.shape)
    slp = torch.cat(list(student_topk_log_probs)) if student_topk_log_probs else torch.zeros((0, 0))
    ids = torch.cat([i.long() for i in teacher_topk_ids]) if teacher_topk_ids else slp.new_zeros((0, 0), dtype=torch.long)
    tlp = torch.cat([t.to(torch.float32) for t in teacher_topk_log_probs]) if teacher_topk_log_probs else slp.new_zeros((0, 0))
    out = {"kl": dense_kl(slp, tlp, kind) if slp.numel() else slp.new_zeros((slp.size(0),))}
    if tokens is None:
        return out
    y = torch.cat([tok[total - resp :].long() for tok, total, resp in zip(tokens, total_lengths, response_lengths, strict=True)])
    assert y.size(0) == ids.size(0), (y.shape, ids.shape)
    match = ids == y[:, None]  # [sum L, k]
    found = match.any(dim=-1)
    pos = match.to(torch.float32).argmax(dim=-1)  # the column of the hit (0 where none, masked below)
    f = found.to(torch.float32)
    s_at = slp.detach().gather(1, pos[:, None]).squeeze(1)
    t_at = tlp.gather(1, pos[:, None]).squeeze(1)
    s_ref = torch.cat([x.detach().to(torch.float32) for x in student_log_probs])
    t_ref = torch.cat([x.detach().to(torch.float32) for x in teacher_log_probs])
    out["hit"] = f
    out["err_student"] = (s_at - s_ref).abs() * f
    out["err_teacher"] = (t_at - t_ref).abs() * f
    return out
