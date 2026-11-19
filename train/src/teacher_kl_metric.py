"""train/teacher_kl-<domain>: the student's KL to each domain's expert, by domain.

opd_multi.sh passes `log_teacher_kl` as slime's --rollout-data-postprocess-path. slime's own
train/opd_reverse_kl pools the domains of a step; this hook reports the same quantity per
domain: the mean over a domain's model tokens of the student's minus the expert's
log-probability at the trained token (the sampled-token estimate of the reverse KL), both
computed by the trainer. Every actor rank calls it; the all-reduce is over the data-parallel
group.
"""

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.utils import logging_utils


def _agreed_sources(source_names, dp_size, dp_group):
    """The sources of the whole batch, the same list on every data-parallel rank: a rank's
    own shard need not hold every source, and the collective below needs every rank to
    walk the same keys."""
    shards = [None] * dp_size
    dist.all_gather_object(shards, sorted(set(source_names)), group=dp_group)
    return sorted(set().union(*shards))


def _mean_ell_by_source(ell, masks, source_names) -> dict[str, float]:
    """The mean of ell over each source's model tokens (nat per token), over the whole
    data-parallel batch."""
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
    sources = _agreed_sources(source_names, dp_size, mpu.get_data_parallel_group_gloo(with_context_parallel=True))
    stats = torch.zeros(2 * len(sources), dtype=torch.float64, device=ell[0].device)
    for i, name in enumerate(source_names):
        k = sources.index(name)
        stats[2 * k] += ell[i].sum().double()  # ell is zero off the model tokens
        stats[2 * k + 1] += masks[i].sum().double()
    dist.all_reduce(stats, group=mpu.get_data_parallel_group(with_context_parallel=True))
    return {source: (stats[2 * k] / stats[2 * k + 1]).item() for k, source in enumerate(sources) if stats[2 * k + 1] > 0}


def log_teacher_kl(args, rollout_id, rollout_data) -> None:
    """Logs train/teacher_kl-<domain> from slime's opd_reverse_kl (student minus expert
    log-prob per response token), masked here to the model tokens."""
    reverse_kl, source_names = rollout_data.get("opd_reverse_kl"), rollout_data.get("source_names")
    if reverse_kl is None or not source_names:
        return
    masks = rollout_data["loss_masks"]
    ell = [k * m for k, m in zip(reverse_kl, masks, strict=True)]
    kl = _mean_ell_by_source(ell, masks, source_names)
    if (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
    ):
        metrics = {"train/teacher_kl_step": rollout_id + 1, **{f"train/teacher_kl-{s}": kl[s] for s in sorted(kl)}}
        logging_utils.log(args, metrics, step_key="train/teacher_kl_step")
