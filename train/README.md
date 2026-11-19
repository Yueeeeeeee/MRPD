# Training reference

The code in `train/src/` and the variables the launchers in `scripts/train/` read. The walkthrough
with complete commands is the [main README](../README.md#training).

Every launcher goes through slime's trainer with the hooks of this folder. A sample is routed by
`metadata.task`:

- `retool`: reasoning with a Python sandbox, graded by the boxed answer.
- `search-r1`: open-domain QA with the retriever, graded by normalized exact match.
- `sql`: text-to-SQL with a read-only SQLite tool, graded by execution.

Every domain scores a trajectory with +1 or -1.

## Layout

```text
train/
  src/
    generate_multi_domain.py     # task routing, rewards and their normalisation, per-domain metrics
    generate_with_retool.py      # reasoning rollout
    generate_with_search.py      # search rollout
    generate_with_sql.py         # SQL rollout
    sql_tools.py                 # schema text, the SQLite executor, execution-accuracy scoring
    tool_sandbox.py              # Python sandbox
    qa_em_format.py              # exact-match grader
    local_search_server.py       # retriever client
    mixed_data_source.py         # prompts handed out round-robin by domain
    teacher_dense_kl.py          # the dense distillation term (top-k + tail atom KL)
    teacher_kl_metric.py         # train/teacher_kl-<domain>

scripts/train/                   # launchers (they put src/ on PYTHONPATH)
  convert_{hf_to_torch_dist,torch_dist_to_hf}.sh
  cold_start.sh                  # SFT initialization
  teacher_grpo_{math,search,sql}.sh   # the single-domain experts; they write the training logs
  grpo_multi.sh                  # replay without the distillation term; online GRPO and RAFT
  opd_multi.sh                   # MRPD and its distillation variants; MOPD
```

slime hook names: `generate_multi_domain.{generate,reward_func,post_process_rewards}`,
`mixed_data_source.MixedTaskDataSource`, `teacher_kl_metric.log_teacher_kl`.

## Launchers

Every launcher validates its inputs on startup and names what is missing. `SAVE_DIR` is also the
directory a run resumes from: an existing directory resumes silently, so a new run needs a new
directory (`CLEAN_SAVE_DIR=1` wipes it first).

| Script | Required | Notes |
| --- | --- | --- |
| `convert_hf_to_torch_dist.sh` | `HF_CHECKPOINT`, `SAVE` | HF → torch_dist. `MODEL_CONFIG` defaults to Qwen3-4B's. |
| `convert_torch_dist_to_hf.sh` | `INPUT_DIR`, `OUTPUT_DIR`, `ORIGIN_HF_DIR` | Reads the architecture from `common.pt`. |
| `cold_start.sh` | `HF_CHECKPOINT`, `REF_LOAD`, `PROMPT_DATA`, `SAVE_DIR` | SFT, 3 epochs. |
| `teacher_grpo_math.sh` | `HF_CHECKPOINT`, `REF_LOAD`, `PROMPT_DATA`, `SAVE_DIR` | Expert of the reasoning domain. Every step's rollouts are written to `SAVE_DIR/rollout_debug/rollout_<k>.pt`: the training log. |
| `teacher_grpo_search.sh` | the same and a running retriever (or `START_RETRIEVER=1 DATA_DIR=...`) | Expert of the search domain. |
| `teacher_grpo_sql.sh` | the same and `SQL_DB_ROOT` | Expert of the SQL domain. |
| `grpo_multi.sh` | `HF_CHECKPOINT`, `REF_LOAD`, `RETOOL_PROMPT_DATA`, `SEARCH_R1_PROMPT_DATA`, `SAVE_DIR` (`SQL_PROMPT_DATA`, `SQL_DB_ROOT` for the third domain) | With `REPLAY_LOGS`: the replay with the policy-gradient term alone, or with `REPLAY_LOSS=sft\|rft` the likelihood of the logged responses. Without it: online GRPO, or with `RAFT=1` online RAFT. |
| `opd_multi.sh` | the same and `MATH_TEACHER_LOAD` / `SEARCH_R1_TEACHER_LOAD` / `SQL_TEACHER_LOAD` with their `*_CKPT_STEP` | With `REPLAY_LOGS`: MRPD and its distillation variants. Without it: MOPD. |

## Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_CHECKPOINT` | required | The initialization as an HF checkpoint (rollout engine and tokenizer). |
| `REF_LOAD` | required | The same weights as a torch_dist checkpoint (the trainer). |
| `SAVE_DIR` | required | Checkpoints, logs, rollout dumps; also the directory a run resumes from. |
| `PROMPT_DATA` | required by the single-domain launchers | The domain's training prompts. |
| `RETOOL_PROMPT_DATA`, `SEARCH_R1_PROMPT_DATA`, `SQL_PROMPT_DATA` | required by the multi-domain launchers (SQL optional) | One prompt file per domain; `SQL_PROMPT_DATA` switches the third domain on and needs `SQL_DB_ROOT`. |
| `MODEL` / `STUDENT` | `qwen3-4b` | Preset for the model config and parallelism (`STUDENT` in `opd_multi.sh`); `qwen3-4b` is the only preset. |
| `MODEL_CONFIG` | from the preset | The model-args file in `scripts/models/`. |
| `MEGATRON_ROOT` | `/root/Megatron-LM` | The Megatron checkout put on `PYTHONPATH`. |
| `REPLAY_LOGS` | unset | A directory written by `tools/merge_rollout_logs.py`. Step k trains on `rollout_k.pt` instead of sampling; the batch size and the number of steps are read from its `manifest.json`. |
| `MULTI_DOMAIN_USE_TASK_REWARD` | `1` in `grpo_multi.sh`, unset in `opd_multi.sh` | `1` keeps the task reward (the policy-gradient term); unset or `0` zeroes it, leaving the distillation term alone (`opd_multi.sh` only). |
| `TEACHER_DENSE_KL` | unset | `opd_multi.sh`: `reverse`, `forward` or `js` selects the dense distillation term; unset is the sampled-token term. |
| `TEACHER_DENSE_KL_COEF` | `1` | Weight of the dense term. |
| `TEACHER_TOPK` | `256` | Number of the expert's top tokens the dense term uses (plus one tail atom). |
| `OPD_KL_COEF` | `1.0` (`0` under `TEACHER_DENSE_KL`) | Weight of the sampled-token term. |
| `MATH_TEACHER_LOAD`, `SEARCH_R1_TEACHER_LOAD`, `SQL_TEACHER_LOAD` | required by `opd_multi.sh` | Each expert's run directory (the `SAVE_DIR` of its `teacher_grpo_*.sh`), not an `iter_*` subdirectory and not an HF directory. |
| `MATH_TEACHER_CKPT_STEP`, `SEARCH_R1_TEACHER_CKPT_STEP`, `SQL_TEACHER_CKPT_STEP` | required by `opd_multi.sh` | The expert iteration to distill, e.g. `299`. |
| `OPD_TEACHER_SOURCE` | `megatron` | The experts are weight sets inside the trainer. The replay and the dense term need this value. |
| `REPLAY_LOSS` | unset | `grpo_multi.sh` under `REPLAY_LOGS`: `sft` is slime's SFT loss on every logged sample, `rft` on the correct ones. |
| `RAFT` | `0` | `grpo_multi.sh` without `REPLAY_LOGS`: `1` trains with the SFT loss on the correct samples of the student's own rollouts, averaged over the step's tokens. |
| `RFT_MAX_PER_PROMPT` | unset (every correct sample) | With `REPLAY_LOSS=rft` or `RAFT=1`: at most this many correct samples per prompt stay in the loss. |
| `NUM_ROLLOUT` | `500` (the logged steps under `REPLAY_LOGS`) | Number of steps. |
| `SAVE_INTERVAL` | `50` | Steps between checkpoints (`iter_0000049`, `iter_0000099`, ...). |
| `LR` | `1e-6` | Learning rate, constant. |
| `MIXED_TASK_ORDER` | `retool,search-r1` (plus `sql` with the third domain) | The order prompts are handed out in, by domain. |
| `SEARCH_URL` | `http://127.0.0.1:8000/retrieve` | Retrieval endpoint(s); comma-separated endpoints are load balanced round-robin with failover. |
| `START_RETRIEVER` | `0` | The experts' launchers and `grpo_multi.sh`: `1` starts a retriever inside the container (needs `DATA_DIR`). `opd_multi.sh` never starts one; under `REPLAY_LOGS` no launcher asks for it. |
| `TOOL_SANDBOX_MAX_TURNS`, `TOOL_SANDBOX_MAX_TOOL_CALLS` | `16`, `16` | Limits of the reasoning domain. |
| `SEARCH_R1_MAX_TURNS` | `8` | Search turns per sample. |
| `SQL_MAX_TURNS`, `SQL_MAX_TOOL_CALLS` | `8`, `6` | Limits of the SQL domain. |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | Passed through to the training workers by `opd_multi.sh`; the dense runs of the paper use `expandable_segments:True`. |
| `USE_WANDB`, `WANDB_PROJECT`, `WANDB_GROUP`, `WANDB_KEY` | `1`, `agentic-opd`, derived from the knobs, unset | Logging. The run name (wandb group) is derived from the knobs, e.g. `grpo-opd-3d-replay-dense-reverse-t299-qwen3-4b`. |
| `AUTO_PATCH_SLIME` | `1` | `opd_multi.sh` applies the [slime patches](../patches/README.md) in place; `0` refuses instead. |

## Metrics

`post_process_rewards` logs per domain `rollout/<domain>_sample_count`, `_task_score_mean`,
`_train_reward_mean` and the tool-call counts; `rollout/raw_reward` is the mean reward of the step.
`opd_multi.sh` logs `train/teacher_kl-<domain>` (the sampled-token estimate of the student's KL to
the domain's expert) and, with the dense term, `train/teacher_dense_kl`, `train/teacher_topk_hit`
(the share of trained tokens inside the expert's top-k) and two self-checks,
`train/teacher_dense_err_teacher` (about 1e-5 when the tensor-parallel paths agree) and
`train/teacher_dense_err_student` (0 up to rounding). `train/train_rollout_logprob_abs_diff` and
`train/pg_clipfrac` are the training-inference mismatch and the clip fraction of the paper's Figure 6.

## Tests

No GPU is needed: megatron, `torch.distributed` and slime's runtime are stubbed. They need a Python
with torch and the `third_party/slime` submodule checked out.

```bash
python3 tests/teacher_dense_kl_test.py        # the dense term: top-k, the fused log-prob function, the KLs and their gradients
python3 tests/dense_loss_patch_test.py        # the patched loss path end to end
python3 tests/teacher_forward_patch_test.py   # the per-domain expert forward and the replay prefetch
python3 tests/teacher_kl_metric_test.py       # train/teacher_kl-<domain>
python3 tests/merge_rollout_logs_test.py      # tools/merge_rollout_logs.py
python3 tests/replay_rft_test.py              # which samples REPLAY_LOSS=rft and RAFT=1 keep
python3 tests/grpo_reward_norm_test.py        # GRPO's reward normalisation
python3 tests/collect_pass_at_k_test.py       # scripts/eval/collect_pass_at_k.py
REPLAY=1 TEACHER_DENSE_KL=reverse bash tests/opd_run_name.sh   # the run name a set of knobs gives
```
