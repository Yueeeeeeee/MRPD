"""python3 tests/teacher_kl_metric_test.py -- train/src/teacher_kl_metric.py (train/teacher_kl-<domain>), single
process with megatron, torch.distributed and slime stubbed (needs torch)."""
import os, sys, types
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/train/src")

RANK = {"dp": 0}
mpu = types.SimpleNamespace(
    get_data_parallel_world_size=lambda with_context_parallel=False: 1,
    get_data_parallel_group=lambda with_context_parallel=False: None,
    get_data_parallel_group_gloo=lambda with_context_parallel=False: None,
    get_data_parallel_rank=lambda with_context_parallel=False: RANK["dp"],
    get_tensor_model_parallel_rank=lambda: 0,
    is_pipeline_last_stage=lambda: True,
)
megatron = types.ModuleType("megatron"); core = types.ModuleType("megatron.core"); core.mpu = mpu
sys.modules["megatron"] = megatron; sys.modules["megatron.core"] = core
import torch.distributed as dist
dist.all_reduce = lambda t, group=None, **kw: None
def _gather(out, obj, group=None): out[0] = obj
dist.all_gather_object = _gather
LOGGED = []
slime = types.ModuleType("slime"); su = types.ModuleType("slime.utils"); lu = types.ModuleType("slime.utils.logging_utils")
lu.log = lambda args, metrics, step_key: LOGGED.append((dict(metrics), step_key))
su.logging_utils = lu
sys.modules.update({"slime": slime, "slime.utils": su, "slime.utils.logging_utils": lu})

import teacher_kl_metric as m

def close(a, b): return abs(a - b) < 1e-6   # the inputs are float32
# three responses over two domains; a mask of 0 is a tool-output token, whose (large) value must not count
kl = [torch.tensor([0.2, 9.0, 0.4]), torch.tensor([1.0, 1.0]), torch.tensor([0.6, 9.0])]
masks = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])]
data = {"opd_reverse_kl": kl, "loss_masks": masks, "source_names": ["retool", "sql", "retool"]}
m.log_teacher_kl(None, 4, data)
metrics, key = LOGGED[-1]
assert key == "train/teacher_kl_step" and metrics["train/teacher_kl_step"] == 5, metrics
assert close(metrics["train/teacher_kl-retool"], (0.2 + 0.4 + 0.6) / 3), metrics   # the mean over the domain's model tokens
assert close(metrics["train/teacher_kl-sql"], 1.0), metrics
assert set(metrics) == {"train/teacher_kl_step", "train/teacher_kl-retool", "train/teacher_kl-sql"}, metrics
# a domain without a model token in the batch is left out rather than reported as nan
data2 = {"opd_reverse_kl": [torch.tensor([0.5]), torch.tensor([3.0])], "loss_masks": [torch.tensor([1.0]), torch.tensor([0.0])],
         "source_names": ["search-r1", "sql"]}
m.log_teacher_kl(None, 0, data2)
assert set(LOGGED[-1][0]) == {"train/teacher_kl_step", "train/teacher_kl-search-r1"}, LOGGED[-1]
# nothing to log without the teacher term or without sources; only the first rank logs
n = len(LOGGED)
m.log_teacher_kl(None, 0, {"loss_masks": masks, "source_names": ["retool"]})
m.log_teacher_kl(None, 0, {"opd_reverse_kl": kl, "loss_masks": masks, "source_names": []})
RANK["dp"] = 1
m.log_teacher_kl(None, 0, data)
assert len(LOGGED) == n, LOGGED[n:]
print("teacher_kl_metric.py: per-domain mean over model tokens, empty domains left out, silent without the term, rank 0 logs: OK")
