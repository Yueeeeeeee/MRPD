# slime patches

MRPD runs on slime v0.3.1 (`third_party/slime`, a pinned submodule). `scripts/train/opd_multi.sh`
applies the patches below to that checkout in place before it starts (each is guarded by a marker
string, so applying twice is a no-op; `AUTO_PATCH_SLIME=0` refuses instead of patching). None of
them changes slime's behaviour unless the corresponding flag is set.

| Patch | What it does |
| --- | --- |
| `slime-opd-restore-actor-after-teacher-load.patch` | slime's `load_other_checkpoint()` leaves the active weight set on whatever it loaded last (an expert, or the reference); the patch switches back to the student at the end of initialisation. |
| `slime-opd-decouple-student-logprobs.patch` | Separates the two uses of the student's log-probabilities. The importance ratio's denominator is the logged sampling log-probability (`--use-rollout-logprobs`); the student side of the distillation term comes from the trainer's own forward pass, the same engine that computes the expert side. |
| `slime-opd-multi-teacher.patch` | One expert per domain (`--opd-teacher-loads retool=...,search-r1=...,sql=...`). Each expert forwards only its own domain's samples, packed into their own micro-batches and scattered back by index; samples are routed by `source_names`, and a sample its expert did not score is refused. |
| `slime-opd-dense-kl.patch` | `--opd-dense-kl {reverse,forward,js}` (`--opd-dense-kl-coef`, `--opd-teacher-topk`): the distillation term on the expert's next-token distribution. The experts' forward pass also stores the top-k ids and log-probabilities at every response position; the policy loss reads the student's log-probabilities at those ids out of its own log-probability function and differentiates the KL between the two (k+1)-atom distributions directly, with no ratio and no clip. The math is in `train/src/teacher_dense_kl.py`. Applied only when `TEACHER_DENSE_KL` is set. |
| `slime-opd-replay-prefetch.patch` | Under `--load-debug-rollout-data` (the replay), reads the next step's log file in a background thread while the current step trains. |

Tests of the patched code paths, on stubs (no GPU): `python3 tests/teacher_dense_kl_test.py`,
`python3 tests/dense_loss_patch_test.py`, `python3 tests/teacher_forward_patch_test.py`.

## Upstream status

Checked against `THUDM/slime@main` on 2026-08-28.

- **restore-actor**: reported upstream; two open PRs fix it
  ([#1903](https://github.com/THUDM/slime/pull/1903), [#2224](https://github.com/THUDM/slime/pull/2224)).
  The patch here has #2224's shape.
- **decouple-student-logprobs**: not reported. `--use-rollout-logprobs` selects one tensor that is
  used both as the importance ratio's denominator and as the student side of the distillation term.
  The ratio wants the sampling policy; the distillation term wants the same engine as the expert.
- The other three are specific to this repository.
