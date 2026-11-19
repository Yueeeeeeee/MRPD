"""The teacher forward of the slime patches on stubs. The submodule's pristine actor.py is copied to a
temp dir and patched with the restore, decouple, multi-teacher and dense patches' actor.py hunks (git
apply --include, the launcher's order); the functions under test are taken out of it by ast and run
with a fake compute_log_prob standing in for the forward (megatron and slime's runtime are not
imported; the packing, slime/utils/seqlen_balancing.py, is the real one).

1. _forward_teachers with --opd-teacher-loads: every sample's teacher_log_probs / teacher_topk_ids come
   from its own source's teacher; each teacher sees its own samples alone, in the batch's order, in
   micro-batches under --max-tokens-per-gpu (an oversized sample alone) or of --micro-batch-size;
   a source absent from the rank costs no forward; the per-teacher lists are gone afterwards.
2. The guards: a sample its teacher did not score (a schedule that drops one) is refused by
   _route_teacher_log_probs; a forward that returns one value too few by _scatter_rows; a source
   without a teacher as before.
3. One teacher (--opd-teacher-load): the batch's own iterator, prefix teacher_, as before.
4. The dense patch's skip of the student's standalone log-prob forward: _student_log_probs_consumed
   is false for the dense arm alone (slime's sampled-token term at 0, the ratio to the rollout
   log-probs, no custom advantage function, no reader among the metrics), and compute_log_prob then
   returns nothing for store_prefix "" while the teachers' forwards (with their top-k) still run.
5. The replay prefetch (slime-opd-replay-prefetch, rollout.py's RolloutManager): the next step's dump
   is read in a thread once a step is handed out and taken when that step is asked for; a failed
   read raises at its step and is dropped; nothing past the last step; a step out of order is read then.
python3 tests/teacher_forward_patch_test.py (needs git)."""
import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
from argparse import Namespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIME = f"{ROOT}/third_party/slime"
REL = "slime/backends/megatron_utils/actor.py"

# ---- the patched actor.py in a temp tree --------------------------------------------------------
tmp = tempfile.mkdtemp()
os.makedirs(f"{tmp}/slime/backends/megatron_utils")
shutil.copy(f"{SLIME}/{REL}", f"{tmp}/{REL}")
for p in ("slime-opd-restore-actor-after-teacher-load", "slime-opd-decouple-student-logprobs", "slime-opd-multi-teacher", "slime-opd-dense-kl"):
    subprocess.run(["git", "apply", f"--include={REL}", f"{ROOT}/patches/{p}.patch"], cwd=tmp, check=True)
src = open(f"{tmp}/{REL}").read()
assert "teacher therefore also scores samples that are not its own" not in src, "the full-batch version of the patch"

# ---- the real packing; stubs for the rest -------------------------------------------------------
spec = importlib.util.spec_from_file_location("slime.utils.seqlen_balancing", f"{SLIME}/slime/utils/seqlen_balancing.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)
sys.modules.update({"slime": types.ModuleType("slime"), "slime.utils": types.ModuleType("slime.utils"), "slime.utils.seqlen_balancing": sb})


class DataIterator:  # slime.backends.megatron_utils.data.DataIterator's interface, as forward_only reads it
    def __init__(self, rollout_data, micro_batch_indices):
        self.rollout_data, self.micro_batch_indices, self.offset = rollout_data, micro_batch_indices, 0


tree = ast.parse(src)
funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MegatronTrainRayActor")
methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
attrs = {t.id: n for n in cls.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: 1, get_virtual_pipeline_model_parallel_world_size=lambda: None)
ns = {"os": os, "mpu": mpu, "DataIterator": DataIterator, "RolloutBatch": dict}
for name in ("_teacher_source", "_teachers_present_in_batch", "_source_forward_schedule", "_scatter_rows", "_route_teacher_log_probs"):
    exec(ast.unparse(funcs[name]), ns)
exec(ast.unparse(attrs["_FORWARD_KEYS"]), ns)
exec(ast.unparse(methods["_forward_teachers"]), ns)


class Actor:
    """weights_backuper, args, _switch_model and a compute_log_prob that forwards nothing: it returns, per sample
    of the view it is handed, (prefix, tokens), in micro-batch order and then back in the view's order, as
    forward_only does under the dynamic batch size; it checks the token cap and what the view carries."""

    _FORWARD_KEYS = ns["_FORWARD_KEYS"]
    _forward_teachers = ns["_forward_teachers"]

    def __init__(self, tags, args):
        self.weights_backuper = types.SimpleNamespace(backup_tags=tags)
        self.args = args
        self.switched, self.calls = [], []

    def _switch_model(self, tag):
        self.switched.append(tag)

    def compute_log_prob(self, data_iterator, num_microbatches, store_prefix=""):
        it = data_iterator[0]
        assert len(num_microbatches) == 1 and num_microbatches[0] == len(it.micro_batch_indices), (num_microbatches, it.micro_batch_indices)
        view = it.rollout_data
        if store_prefix != "teacher_":  # by source: a view of the forward's keys; one teacher: the batch itself
            assert set(view) <= set(self._FORWARD_KEYS) and "tokens" in view and "source_names" not in view, sorted(view)
        values = []
        for mbs in it.micro_batch_indices:
            toks = [view["tokens"][j] for j in mbs]
            if self.args.use_dynamic_batch_size:
                assert sum(len(t) for t in toks) <= self.args.max_tokens_per_gpu or len(mbs) == 1, (store_prefix, mbs)
            else:
                assert len(mbs) <= self.args.micro_batch_size, (store_prefix, mbs)
            values += [(store_prefix, tuple(t)) for t in toks]
        back = [None] * len(values)
        for v, i in zip(values, sum(it.micro_batch_indices, []), strict=True):
            back[i] = v
        self.calls.append((store_prefix, data_iterator, [tuple(t) for t in view["tokens"]], [len(m) for m in it.micro_batch_indices]))
        return {f"{store_prefix}log_probs": back, f"{store_prefix}topk_ids": [("topk",) + v for v in back]}


def batch(sources, lengths, cap=8):
    tokens = [list(range(100 * i, 100 * i + n)) for i, n in enumerate(lengths)]
    mbs = sb.first_fit_pack(lengths, cap)
    data = {
        "tokens": tokens, "loss_masks": [[1] * n for n in lengths], "total_lengths": list(lengths),
        "response_lengths": [n - 1 for n in lengths], "source_names": list(sources), "rewards": [1.0] * len(lengths),
        "micro_batch_indices": mbs, "num_microbatches": [len(mbs)],
    }
    return data, [DataIterator(data, mbs)], [len(mbs)]


TAGS = ["actor", "teacher::retool", "teacher::search-r1", "teacher::sql"]
dyn = Namespace(use_routing_replay=False, use_dynamic_batch_size=True, max_tokens_per_gpu=8, micro_batch_size=1)
static = Namespace(use_routing_replay=False, use_dynamic_batch_size=False, max_tokens_per_gpu=None, micro_batch_size=2)
sources = ["retool", "sql", "search-r1", "retool", "sql", "retool", "search-r1", "sql", "retool", "search-r1"]
lengths = [5, 3, 4, 6, 2, 9, 4, 3, 5, 2]  # the 9 is above the cap of 8: alone in its micro-batch


def check_routed(data, sources):
    for i, s in enumerate(sources):
        assert data["teacher_log_probs"][i] == (f"teacher::{s}_", tuple(data["tokens"][i])), i
        assert data["teacher_topk_ids"][i] == ("topk", f"teacher::{s}_", tuple(data["tokens"][i])), i
    assert not [k for k in data if k.startswith("teacher::")], "the per-teacher lists are dropped"


# ---- 1. by source, dynamic and static micro-batches --------------------------------------------------
for args in (dyn, static):
    data, iters, nmb = batch(sources, lengths)
    actor = Actor(TAGS, args)
    actor._forward_teachers(data, iters, nmb)
    check_routed(data, sources)
    assert actor.switched == ["teacher::retool", "teacher::search-r1", "teacher::sql"], actor.switched
    for prefix, its, toks, sizes in actor.calls:
        s = prefix[len("teacher::"):-1]
        assert toks == [tuple(t) for t, src in zip(data["tokens"], sources, strict=True) if src == s], "its own samples, in the batch's order"
        assert sum(sizes) == sources.count(s) and its[0] is not iters[0], (prefix, sizes)
    if args is dyn:
        sizes = {p: sz for p, _, _, sz in actor.calls}
        # retool 5, 6, 9, 5 under 8: first fit opens a bin for each (the 9 alone, above the cap); search-r1 4, 4, 2: [4, 4], [2]; sql 3, 2, 3: one bin
        assert sizes["teacher::retool_"] == [1, 1, 1, 1] and sizes["teacher::search-r1_"] == [2, 1] and sizes["teacher::sql_"] == [3], sizes
    else:
        assert [sz for _, _, _, sz in actor.calls] == [[2, 2], [2, 1], [2, 1]], [sz for _, _, _, sz in actor.calls]
print("1 by source: every sample from its own teacher; its samples alone; the token cap / micro-batch size; dynamic and static: OK")

data, iters, nmb = batch([s for s in sources if s != "sql"], [n for s, n in zip(sources, lengths, strict=True) if s != "sql"])
actor = Actor(TAGS, dyn)
actor._forward_teachers(data, iters, nmb)
check_routed(data, data["source_names"])
assert actor.switched == ["teacher::retool", "teacher::search-r1"], actor.switched
print("1 a source absent from the rank: its teacher is not forwarded: OK")

# ---- 2. the guards -------------------------------------------------------------------------------------
good = ns["_source_forward_schedule"]


def dropping(source_names, total_lengths, source, max_tokens, mbs_size):
    idx, _ = good(source_names, total_lengths, source, max_tokens, mbs_size)
    idx = idx[:-1]
    return idx, sb.first_fit_pack([total_lengths[i] for i in idx], max_tokens)


ns["_source_forward_schedule"] = dropping
data, iters, nmb = batch(sources, lengths)
try:
    Actor(TAGS, dyn)._forward_teachers(data, iters, nmb)
    raise AssertionError("a sample its teacher did not score went through")
except ValueError as e:
    assert "was not scored by its teacher" in str(e), e
ns["_source_forward_schedule"] = good
print("2 a schedule that drops a sample: refused by the routing: OK")


class Short(Actor):
    def compute_log_prob(self, data_iterator, num_microbatches, store_prefix=""):
        out = super().compute_log_prob(data_iterator, num_microbatches, store_prefix)
        return {k: v[:-1] for k, v in out.items()}


data, iters, nmb = batch(sources, lengths)
try:
    Short(TAGS, dyn)._forward_teachers(data, iters, nmb)
    raise AssertionError("a forward one value short went through")
except AssertionError as e:
    assert "(" in str(e), e
print("2 a forward one value short: refused by the scatter: OK")

data, iters, nmb = batch(sources[:-1] + ["code"], lengths)
try:
    Actor(TAGS, dyn)._forward_teachers(data, iters, nmb)
    raise AssertionError("a source without a teacher went through")
except ValueError as e:
    assert "has no teacher" in str(e), e
print("2 a source without a teacher: refused as before: OK")

# ---- 3. one teacher ----------------------------------------------------------------------------------
data, iters, nmb = batch(sources, lengths)
actor = Actor(["actor", "teacher"], dyn)
actor._forward_teachers(data, iters, nmb)
assert actor.switched == ["teacher"] and len(actor.calls) == 1 and actor.calls[0][1] is iters, actor.calls
assert data["teacher_log_probs"] == [("teacher_", tuple(t)) for t in data["tokens"]]
print("3 one teacher: the batch's own iterator, prefix teacher_: OK")

# ---- 4. the student's standalone forward: skipped when nothing reads it (dense patch) ----------------
from functools import partial  # noqa: E402

ns["torch"] = types.SimpleNamespace(Tensor=object)  # the annotations alone
exec(ast.unparse(methods["_student_log_probs_consumed"]), ns)
exec(ast.unparse(methods["compute_log_prob"]), ns)
dense = dict(opd_dense_kl="reverse", opd_kl_coef=0, use_rollout_logprobs=True, custom_advantage_function_path=None,
             get_mismatch_metrics=False, use_tis=False, log_correct_samples=False, opd_teacher_topk=7)
consumed = ns["_student_log_probs_consumed"]
assert not consumed(types.SimpleNamespace(args=Namespace(**dense))), "the dense arm: nothing reads it"
for k, v in (("opd_dense_kl", None), ("opd_kl_coef", 1.0), ("use_rollout_logprobs", False), ("custom_advantage_function_path", "my_module.compute"),
             ("get_mismatch_metrics", True), ("use_tis", True), ("log_correct_samples", True)):
    assert consumed(types.SimpleNamespace(args=Namespace(**{**dense, k: v}))), f"read when {k}={v!r}"
sampled = {**dense, "opd_dense_kl": None, "opd_kl_coef": 1.0}
assert consumed(types.SimpleNamespace(args=Namespace(**sampled))), "the sampled-token arms keep it"
print("4 _student_log_probs_consumed: the dense arm alone skips it: OK")


class Forwarder:
    _student_log_probs_consumed = ns["_student_log_probs_consumed"]
    compute_log_prob = ns["compute_log_prob"]

    def __init__(self, args):
        self.args, self.model, self.forwards = args, "model", []


def forward_only(f, args, model, data_iterator, num_microbatches, store_prefix="", use_rollout_top_p_replay=False):
    Forwarder.last.forwards.append((store_prefix, f))
    return {f"{store_prefix}log_probs": ["forwarded"]}


class timer:
    def __init__(self, name):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def get_log_probs_and_entropy(*a, **k):
    pass


ns.update(forward_only=forward_only, timer=timer, get_log_probs_and_entropy=get_log_probs_and_entropy, partial=partial)
Forwarder.last = fw = Forwarder(Namespace(**dense))
assert fw.compute_log_prob(iters, nmb, store_prefix="") == {} and fw.forwards == [], "the student forward is skipped"
out = fw.compute_log_prob(iters, nmb, store_prefix="teacher::sql_")
assert out == {"teacher::sql_log_probs": ["forwarded"]} and fw.forwards[-1][0] == "teacher::sql_", out
assert isinstance(fw.forwards[-1][1], partial) and fw.forwards[-1][1].keywords == {"with_topk": 7}, "the teacher forward stores its top-k"
Forwarder.last = fw = Forwarder(Namespace(**{**dense, "custom_advantage_function_path": "x.compute"}))
assert fw.compute_log_prob(iters, nmb, store_prefix="") == {"log_probs": ["forwarded"]} and fw.forwards[-1][1] is get_log_probs_and_entropy, "read: forwarded, plain"
print("4 compute_log_prob: the student forward skipped in the dense arm, the teachers' and a read one run: OK")

# ---- 5. the replay prefetch (slime-opd-replay-prefetch, rollout.py): the next step's dump read ahead --------
import concurrent.futures  # noqa: E402

RREL = "slime/ray/rollout.py"
os.makedirs(f"{tmp}/slime/ray")
shutil.copy(f"{SLIME}/{RREL}", f"{tmp}/{RREL}")
subprocess.run(["git", "apply", f"--include={RREL}", f"{ROOT}/patches/slime-opd-replay-prefetch.patch"], cwd=tmp, check=True)
rtree = ast.parse(open(f"{tmp}/{RREL}").read())
rcls = next(n for n in rtree.body if isinstance(n, ast.ClassDef) and n.name == "RolloutManager")
rmethods = {n.name: n for n in rcls.body if isinstance(n, ast.FunctionDef)}
loads = []


def fake_load(path, weights_only=False):
    loads.append(os.path.basename(path))
    if "rollout_2." in path:
        raise RuntimeError("bad dump")
    return {"samples": [{"id": os.path.basename(path)}]}


rns = {"os": os, "concurrent": concurrent, "torch": types.SimpleNamespace(load=fake_load),
       "Sample": types.SimpleNamespace(from_dict=lambda d: ("S", d["id"]))}
for name in ("_load_debug_rollout_samples", "_debug_rollout_samples"):
    exec(ast.unparse(rmethods[name]), rns)
dumps = tempfile.mkdtemp()
for k in range(4):
    open(f"{dumps}/rollout_{k}.pt", "w").close()


class Manager:
    _load_debug_rollout_samples = rns["_load_debug_rollout_samples"]
    _debug_rollout_samples = rns["_debug_rollout_samples"]

    def __init__(self):
        self.args = Namespace(load_debug_rollout_data=f"{dumps}/rollout_{{rollout_id}}.pt", num_rollout=4)


def settle(m):  # wait for the read in flight, as the training of the step would
    if m._prefetched_debug_rollout is not None:
        concurrent.futures.wait([m._prefetched_debug_rollout[1]])


m = Manager()
assert m._debug_rollout_samples(0) == [("S", "rollout_0.pt")] and loads[0] == "rollout_0.pt", "step 0 read now"
settle(m)
assert loads == ["rollout_0.pt", "rollout_1.pt"] and m._prefetched_debug_rollout[0] == 1, "step 1 read ahead"
assert m._debug_rollout_samples(1) == [("S", "rollout_1.pt")], "step 1 taken from the read ahead"
settle(m)
assert loads == ["rollout_0.pt", "rollout_1.pt", "rollout_2.pt"], "not read again; step 2 read ahead"
try:
    m._debug_rollout_samples(2)
    raise AssertionError("a failed read ahead went through")
except RuntimeError as e:
    assert "bad dump" in str(e)
assert m._prefetched_debug_rollout is None, "the failed read ahead is dropped"
try:
    m._debug_rollout_samples(2)
    raise AssertionError("a retried bad step went through")
except RuntimeError:
    pass
assert loads[-1] == "rollout_2.pt" and loads.count("rollout_2.pt") == 2 and m._prefetched_debug_rollout is None, "a retry reads the step now"
assert m._debug_rollout_samples(3) == [("S", "rollout_3.pt")] and m._prefetched_debug_rollout is None, "the last step: nothing past the end"
loads.clear()
assert m._debug_rollout_samples(1) == [("S", "rollout_1.pt")] and loads[0] == "rollout_1.pt", "out of order (a resume): read now"
settle(m)
os.remove(f"{dumps}/rollout_1.pt")
m3 = Manager()
m3._debug_rollout_samples(0)
assert m3._prefetched_debug_rollout is None, "no file for the next step: not started"
shutil.rmtree(dumps)
print("5 replay prefetch: the next dump read ahead and taken, a failure surfaced at its step and dropped, nothing past the end, a resume read now: OK")

shutil.rmtree(tmp)
print("teacher_forward_patch_test: OK")
