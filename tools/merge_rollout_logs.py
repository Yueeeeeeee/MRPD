#!/usr/bin/env python3
"""Merge the teachers' rollout dumps into the per-step files a replay arm trains on.

    python3 tools/merge_rollout_logs.py --steps 300 --out /workspace/runs/replay_logs/t3d \
        --teacher retool=/workspace/runs/grpo_math \
        --teacher search-r1=/workspace/runs/grpo_search \
        --teacher sql=/workspace/runs/grpo_sql

Every launcher dumps each rollout as <SAVE_DIR>/rollout_debug/rollout_<k>.pt (slime's
--save-debug-rollout-data: the samples as the generate function returned them, before the
reward post-processing, so `reward` is the raw score and `rollout_log_probs` the sampling
engine's log-prob of every sampled token). slime's --load-debug-rollout-data reads one such
file per step back and trains on it with no rollout engine (grpo_multi.sh, REPLAY_LOGS=<dir>).

This writes <out>/rollout_<k>.pt for k < steps: the teachers' step-k samples concatenated in
the order the --teacher arguments are given, every sample
  - tagged with its task in metadata.task (what generate_multi_domain routes and groups by),
  - given the score attributes generate_multi_domain.post_process_rewards reads
    (_opd_task_score and _opd_task_score_unshaped: float(reward["score"]) for a dict reward,
    float(reward) otherwise; _opd_task), `reward` itself left as it was,
  - renumbered so that a prompt's n samples stay consecutive and no two teachers' groups
    collide: index = k * samples_per_step + position, group_index = index // n.
Sample.rollout_id is left as the dump has it (None): slime then numbers every sample as its own
rollout when it builds the training batch, exactly as in a run that samples -- its DP schedule
counts rollouts against --global-batch-size, so one shared id would leave it a single rollout.
Beside the files, manifest.json records the shape (steps, samples_per_step, prompts_per_step,
n_samples_per_prompt, the task order), which the launcher reads for its batch size, step count
and run name. One teacher alone gives a single-domain replay (the acceptance test: its own
logs back through the same loss).

Two controls of the replay arm, both recorded in the manifest and in the run name:
  --groups-per-teacher G   only the first G prompt groups of every teacher's step (G * n samples
                           per teacher per step; the sample budget of an online run at G = 24,
                           576 per step for three teachers); manifest groups_per_teacher, -g<G>.
  --shuffle-steps SEED     the steps in a seeded random order: merged step k holds every
                           teacher's step order[k] (one permutation for all teachers, so a step
                           stays a step of the teachers' training at one stage); the 300 steps
                           are the same set, only the order changes. manifest shuffle_seed and
                           step_order, -shuf<SEED>.

Checked on every sample, a violation stopping the run: rollout_log_probs present and as long
as the response, loss_mask as long as the response, a score in the reward, the n samples of a
group sharing one original group; and every step having the same number of samples per
teacher. The steps are independent, so --workers of them run at once (the time is pickling).
"""
import argparse
import json
import multiprocessing
import os
import random

import torch


class MergeError(Exception):
    """A dump that cannot be replayed; raised inside a worker, reported by the parent."""


def score_of(reward):
    if isinstance(reward, dict):
        for key in ("score", "reward"):
            if key in reward:
                return float(reward[key])
        raise MergeError(f"a reward dict without a score: keys {sorted(reward)}")
    if reward is None:
        raise MergeError("a sample without a reward")
    return float(reward)


def available_steps(run_dir):
    """How many consecutive rollout_<k>.pt files start at 0."""
    k = 0
    while os.path.exists(os.path.join(run_dir, "rollout_debug", f"rollout_{k}.pt")):
        k += 1
    return k


def load_step(run_dir, k):
    path = os.path.join(run_dir, "rollout_debug", f"rollout_{k}.pt")
    samples = torch.load(path, weights_only=False)["samples"]
    if not samples:
        raise MergeError(f"{path}: no samples")
    return samples


def check_groups(samples, n, where):
    if len(samples) % n:
        raise MergeError(f"{where}: {len(samples)} samples are not whole groups of {n}")
    for i in range(0, len(samples), n):
        block = samples[i : i + n]
        groups = {s.get("group_index", (s.get("index") or 0) // n) for s in block}
        if len(groups) != 1:
            raise MergeError(f"{where}: samples {i}..{i + n - 1} span groups {sorted(groups)}; not a whole prompt group")


def merge_step(job):
    """One step: the teachers' files of step `src` read, checked, tagged, renumbered and written
    as rollout_<k>.pt. Returns (k, samples, {task: samples}, {task: truncated}, {task: score sum})."""
    k, src, teachers, n, groups, out = job
    merged = []
    per_task, truncated, score_sum = {}, {}, {}
    for task, run_dir in teachers:
        where = f"{run_dir}/rollout_debug/rollout_{src}.pt"
        samples = load_step(run_dir, src)
        check_groups(samples, n, where)
        if groups is not None:
            if len(samples) < groups * n:
                raise MergeError(f"{where}: {len(samples) // n} groups, --groups-per-teacher asks for {groups}")
            samples = samples[: groups * n]
        truncated[task] = 0
        score_sum[task] = 0.0
        for s in samples:
            rl = s["response_length"]
            lp = s.get("rollout_log_probs")
            if lp is None or len(lp) != rl:
                raise MergeError(f"{where}: a sample with rollout_log_probs of length {None if lp is None else len(lp)} for a response of {rl}")
            lm = s.get("loss_mask")
            if lm is not None and len(lm) != rl:
                raise MergeError(f"{where}: a sample with a loss_mask of length {len(lm)} for a response of {rl}")
            score = score_of(s.get("reward"))
            metadata = dict(s.get("metadata") or {})
            metadata["task"] = task
            s["metadata"] = metadata
            s["_opd_task"] = task
            s["_opd_task_score"] = score
            s["_opd_task_score_unshaped"] = score
            score_sum[task] += score
            if s.get("status") == "truncated":
                truncated[task] += 1
            merged.append(s)
        per_task[task] = len(samples)
    total = len(merged)
    for i, s in enumerate(merged):
        s["index"] = k * total + i
        s["group_index"] = s["index"] // n
    torch.save({"rollout_id": k, "samples": merged}, os.path.join(out, f"rollout_{k}.pt"))
    return k, total, per_task, truncated, score_sum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", action="append", required=True, metavar="TASK=RUN_DIR",
                    help="retool / search-r1 / sql = the run whose rollout_debug/ to replay; order = batch order")
    ap.add_argument("--out", required=True, help="output directory (refused if it exists)")
    ap.add_argument("--steps", type=int, default=None, help="teacher steps to merge (default: all every teacher has)")
    ap.add_argument("--n-samples-per-prompt", type=int, default=8)
    ap.add_argument("--groups-per-teacher", type=int, default=None, metavar="G",
                    help="keep only the first G prompt groups of every teacher's step (default: all)")
    ap.add_argument("--shuffle-steps", type=int, default=None, metavar="SEED",
                    help="write the steps in a seeded random order (default: the logged order)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="steps merged in parallel")
    args = ap.parse_args()

    teachers = []
    for spec in args.teacher:
        task, run_dir = spec.split("=", 1)
        if task not in ("retool", "search-r1", "sql"):
            raise SystemExit(f"--teacher task must be retool, search-r1 or sql, got '{task}'")
        if not os.path.isdir(os.path.join(run_dir, "rollout_debug")):
            raise SystemExit(f"{run_dir}/rollout_debug is not a directory")
        teachers.append((task, run_dir))
    if len({t for t, _ in teachers}) != len(teachers):
        raise SystemExit("one --teacher per task")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists")

    have = {task: available_steps(run_dir) for task, run_dir in teachers}
    steps = args.steps if args.steps is not None else min(have.values())
    short = {task: k for task, k in have.items() if k < steps}
    if short:
        raise SystemExit(f"not enough consecutive rollout files from 0: {short} (asked for {steps})")
    n = args.n_samples_per_prompt
    groups = args.groups_per_teacher
    if groups is not None and groups < 1:
        raise SystemExit(f"--groups-per-teacher must be >= 1, got {groups}")
    order = list(range(steps))
    if args.shuffle_steps is not None:
        order = random.Random(args.shuffle_steps).sample(order, steps)
    os.makedirs(args.out)

    jobs = [(k, order[k], teachers, n, groups, args.out) for k in range(steps)]
    workers = max(1, min(args.workers, steps))
    pool = multiprocessing.Pool(workers) if workers > 1 else None
    results = pool.imap(merge_step, jobs) if pool else map(merge_step, jobs)
    samples_per_step, per_task = None, None
    truncated = {task: 0 for task, _ in teachers}
    score_sum = {task: 0.0 for task, _ in teachers}
    try:
        for k, total, pt, tr, ss in results:
            if samples_per_step is None:
                samples_per_step, per_task = total, pt
            elif total != samples_per_step or pt != per_task:
                raise MergeError(f"step {k}: {pt} samples per teacher, step 0 had {per_task}")
            for task in truncated:
                truncated[task] += tr[task]
                score_sum[task] += ss[task]
            if k % 25 == 0 or k == steps - 1:
                print(f"step {k}: {total} samples written", flush=True)
    except MergeError as e:
        if pool:
            pool.terminate()
        raise SystemExit(str(e)) from None
    if pool:
        pool.close()
        pool.join()

    manifest = {
        "teachers": {task: run_dir for task, run_dir in teachers},
        "order": [task for task, _ in teachers],
        "steps": steps,
        "n_samples_per_prompt": n,
        "samples_per_step": samples_per_step,
        "prompts_per_step": samples_per_step // n,
        "per_task_samples_per_step": per_task,
        "groups_per_teacher": groups,
        "shuffle_seed": args.shuffle_steps,
    }
    if args.shuffle_steps is not None:
        manifest["step_order"] = order
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    extra = (f", the first {groups} groups per teacher" if groups is not None else "") + (
        f", steps shuffled with seed {args.shuffle_steps}" if args.shuffle_steps is not None else ""
    )
    print(f"\n{steps} steps x {samples_per_step} samples ({manifest['prompts_per_step']} prompts x {n}){extra} -> {args.out}")
    for task, _ in teachers:
        total = per_task[task] * steps
        print(f"  {task}: {per_task[task]} per step, mean score {score_sum[task] / total:+.3f}, truncated {truncated[task] / total:.1%}")


if __name__ == "__main__":
    main()
