# MRPD: Multi-Expert Replay Policy Distillation

Code for **"Replay, Don't Resample: Off-Policy Consolidation of Long-Horizon RL Experts"**.

Frontier post-training trains one RL expert per domain and then consolidates the experts into one
model, usually by multi-teacher on-policy distillation (MOPD): the student samples in every
environment again and each expert scores its tokens. MRPD consolidates the experts **without sampling
anything**. Every RL step of an expert already produced a batch of rollouts with their rewards and
sampling log-probabilities. MRPD replays these training logs through the student, in the order the
experts learned from them, with two per-token terms:

- **Policy gradient (PG).** GRPO's clipped surrogate, recomputed on the logged trajectories and rewards,
  with the importance ratio taken to the logged sampling policy.
- **Distillation (KL).** The KL between the student's and the final expert's next-token distributions at
  every logged prefix, computed in closed form over the expert's top-k tokens and a tail atom, with no
  ratio and no clip.

The trainer is the only process: no environment, no tool call, no rollout engine. Following the experts'
order keeps the student close to the policies that sampled the logs, so the replay is off-policy in
origin but stays within the clip range throughout.

Results after 300 steps with Qwen3-4B on three agentic domains (reasoning with a Python sandbox, search
with a retriever, text-to-SQL), as domain averages over sixteen held-out sets:

| Method | Reasoning | Search | SQL | Overall |
| --- | --- | --- | --- | --- |
| SFT initialization | 47.3 | 22.4 | 14.9 | 28.2 |
| Online GRPO on the three domains | 50.3 | 39.7 | 67.4 | 52.5 |
| MOPD | 63.8 | 40.6 | 69.9 | 58.1 |
| MOPD + GRPO | 63.4 | 41.4 | 69.7 | 58.2 |
| **MRPD** | 63.7 | 42.0 | 70.1 | **58.6** |
| Single-domain GRPO experts (each on its own domain) | 65.2 | 43.0 | 69.6 | 59.3 |

On eight H100s a step of MRPD takes about 110 s against 350 s for MOPD, and the 300 steps take 9 hours.

## Repository layout

```text
scripts/
  train/       # cold_start.sh, teacher_grpo_{math,search,sql}.sh (the experts), grpo_multi.sh, opd_multi.sh, checkpoint converters
  eval/        # math.sh, search.sh, sql.sh, collect_results.py, collect_pass_at_k.py, start_retriever.sh
  data/        # prepare_sql.py (Spider / BIRD databases and prompt files)
  models/      # the Megatron model config of Qwen3-4B
train/         # rollout and reward code, the dense distillation term (src/); variable reference (README.md)
tools/         # merge_rollout_logs.py (the experts' logs -> replay files), checkpoint converters
patches/       # the slime patches the launcher applies (README.md)
eval/          # evaluation code: math/ (6 sets), search/ (7 QA sets and the retriever server), sql/ (Spider dev / test, BIRD mini-dev)
tests/         # unit tests on stubs, no GPU
docs/search.md # the search retriever in detail
third_party/slime   # the training framework (submodule, pinned to v0.3.1)
```

The pipeline has five stages:

| Stage | Script | Output |
| --- | --- | --- |
| 1. Initialization | `scripts/train/cold_start.sh` | the shared SFT checkpoint every model starts from |
| 2. Experts | `scripts/train/teacher_grpo_{math,search,sql}.sh` | one GRPO expert per domain, and its training log |
| 3. Merge the logs | `tools/merge_rollout_logs.py` | one replay file per step |
| 4. MRPD | `scripts/train/opd_multi.sh` with `REPLAY_LOGS` | the consolidated student |
| 5. Evaluation | `scripts/eval/{math,search,sql}.sh` | scores on the sixteen sets |

## Installation

Two places are used throughout, and every command below assumes them:

| | Path | What is there |
| --- | --- | --- |
| Host | `/opd_workspace` (`MRPD_STORAGE`) | checkpoints, data, runs, caches; the retriever's conda environment |
| Container | `/workspace/MRPD` | this checkout |
| Container | `/workspace/{ckpt,data,runs,cache}` | the host's `/opd_workspace/{ckpt,data,runs,cache}` |

Steps 1 to 3 run on the host, in one shell. Steps 4 to 6 run inside the container.

### 1. Clone and prepare the storage (host)

```bash
git clone --recurse-submodules https://github.com/Yueeeeeeee/MRPD.git
cd MRPD
export MRPD_ROOT="$(pwd)"

# host storage: checkpoints, data, caches, ray's /tmp
export MRPD_STORAGE=/opd_workspace
sudo mkdir -p "${MRPD_STORAGE}" && sudo chown "$(id -u):$(id -g)" "${MRPD_STORAGE}"
mkdir -p "${MRPD_STORAGE}"/{ckpt,data,runs,cache,tmp,envs}
df -h "${MRPD_STORAGE}"   # needs a few hundred GB free
```

`MRPD_STORAGE` is also the variable `scripts/eval/start_retriever.sh` reads.

### 2. The search retriever (host)

The search domain's tool is a dense retriever over Wikipedia (e5 over `wiki-18`). It is needed to
**train the search expert** and to **evaluate on search**; MRPD's own training never asks for it. It runs
on the host, outside Docker, from a conda environment on the persistent disk, and the container reaches
it through `--network host`. Background and troubleshooting: [`docs/search.md`](docs/search.md).

Conda:

```bash
curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p ${MRPD_STORAGE}/miniconda3
source ${MRPD_STORAGE}/miniconda3/etc/profile.d/conda.sh
```

The environment. Run **one** of the two blocks, by GPU:

```bash
# A100 (and other pre-Hopper cards): faiss from pip
conda create -y -p ${MRPD_STORAGE}/envs/retriever python=3.10
conda activate ${MRPD_STORAGE}/envs/retriever
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.46.3" datasets pyserini huggingface_hub uvicorn fastapi
pip install faiss-gpu-cu12
```

```bash
# H100: faiss from conda-forge (the pip wheel has no kernels for this card)
conda create -y -p ${MRPD_STORAGE}/envs/retriever -c conda-forge python=3.10 faiss-gpu=1.14.3
conda activate ${MRPD_STORAGE}/envs/retriever
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.46.3" datasets pyserini huggingface_hub uvicorn fastapi
```

Check that faiss runs a real kernel on this card (`get_num_gpus()` passes even when it cannot):

```bash
python -c "
import faiss, numpy as np
r = faiss.StandardGpuResources(); i = faiss.GpuIndexFlatL2(r, 8)
i.add(np.zeros((2, 8), 'float32')); i.search(np.zeros((1, 8), 'float32'), 1)
print('OK', faiss.__version__)"
```

The index and corpus (about 132 GB of disk, once), still in the retriever environment:

```bash
DATA_DIR=${MRPD_STORAGE}/data/search_data
mkdir -p $DATA_DIR
python ${MRPD_ROOT}/eval/search/local_dense_retriever/download.py --save_path $DATA_DIR
cat $DATA_DIR/part_* > $DATA_DIR/e5_Flat.index    # the index ships split in two
gzip -d $DATA_DIR/wiki-18.jsonl.gz                # the corpus ships gzipped
rm $DATA_DIR/part_*
```

Start the server in the background and wait until it answers (the first start loads the index, a few
minutes):

```bash
RETRIEVER_GPUS=0,1,2,3,4,5,6,7 nohup bash ${MRPD_ROOT}/scripts/eval/start_retriever.sh \
    > ${MRPD_STORAGE}/retriever.log 2>&1 &

QUERY='{"queries":["Who wrote Hamlet?"],"topk":3,"return_scores":true}'
until curl -sf -o /dev/null http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d "$QUERY"; do
    sleep 10
done
curl -s http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d "$QUERY" | head -c 300
```

The training scripts measure the GPU memory the retriever holds and size SGLang around it.

### 3. Start the container (host)

This repository pins slime **v0.3.1** (`a6272da0d4f3d0a08520c99a2f3b4f6c887960dc`), and Docker Hub
carries a release image tagged to match. Pull it by immutable digest and start it with the checkout and
the storage mounted. The bind mounts are not optional:

```bash
export SLIME_IMAGE='slimerl/slime:v0.3.1@sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7'
docker pull "${SLIME_IMAGE}"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MRPD_ROOT}:/workspace/MRPD" \
  -v "${MRPD_STORAGE}/ckpt:/workspace/ckpt" \
  -v "${MRPD_STORAGE}/data:/workspace/data" \
  -v "${MRPD_STORAGE}/runs:/workspace/runs" \
  -v "${MRPD_STORAGE}/cache:/workspace/cache" \
  -v "${MRPD_STORAGE}/tmp:/tmp" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e XDG_CACHE_HOME=/workspace/cache \
  -e TRITON_CACHE_DIR=/workspace/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions \
  -e START_RETRIEVER=0 \
  "${SLIME_IMAGE}" /bin/bash
```

| | |
| --- | --- |
| Image | `slimerl/slime:v0.3.1`, `linux/amd64`, 21.1 GB; do not use `latest`, which resolves to a later slime |
| CUDA | 12.9 (base image `slimerl/sglang:v0.5.15.post1-cu129`) |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087`, at `/root/Megatron-LM` |
| GPUs | Ampere (A100) through Hopper (H100); the launchers assume a node with eight |

`START_RETRIEVER=0` tells every script to use the retriever on the host. The `-e` cache variables keep
the library caches off the container filesystem, which is small and discarded on exit:

| Container path | Written by | Typical size |
| --- | --- | --- |
| `/workspace/ckpt` | HF downloads, torch_dist conversions | 10–150 GB |
| `/workspace/data` | datasets | 1–20 GB (plus 132 GB for the search index and corpus) |
| `/workspace/runs` | checkpoints, the experts' training logs, wandb, SGLang logs | large, see below |
| `/workspace/cache` | HF hub, Triton, TorchInductor, torch-extension caches | 5–30 GB |
| `/tmp` | Ray sessions, worker logs, object spilling | 5–50 GB |

Ray aborts the job when the filesystem of `/tmp` passes 95% full. One `torch_dist` save of a 4B model
is about 56 GB (weights and optimizer state), and a run saves every 50 steps: plan on a few hundred GB
per run.

### 4. Install inside the container

This installs the pinned slime submodule without replacing the image's CUDA, PyTorch, Ray, SGLang or
Megatron:

```bash
cd /workspace/MRPD
git config --global --add safe.directory '*'      # the checkout belongs to the host user
git submodule sync --recursive
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install -e third_party/slime --no-deps

export MEGATRON_ROOT=/root/Megatron-LM
export PYTHONPATH="${MEGATRON_ROOT}:$(pwd)/third_party/slime"
```

Do not run broad package upgrades in this container (`pip install -U sglang|transformers|...`).

### 5. Verify the dependency pairing

The image bakes its own slime at `/root/slime`; every script puts `third_party/slime` first on
`PYTHONPATH`. Confirm that the submodule is what runs:

```bash
test "$(git -C third_party/slime rev-parse HEAD)" = a6272da0d4f3d0a08520c99a2f3b4f6c887960dc && echo "slime ok"
test "$(git -C /root/Megatron-LM rev-parse HEAD)" = 1dcf0dafa884ad52ffb243625717a3471643e087 && echo "Megatron ok"

python - <<'PY'
import inspect
from importlib.metadata import version
import slime
from megatron.training import arguments as megatron_arguments
print("slime:", inspect.getfile(slime))
print("Megatron:", inspect.getfile(megatron_arguments))
print("SGLang:", version("sglang"))
PY
```

Expected:

```text
slime ok
Megatron ok
slime: /workspace/MRPD/third_party/slime/slime/__init__.py
Megatron: /root/Megatron-LM/megatron/training/arguments.py
SGLang: 0.5.15.post1
```

### 6. The text-to-SQL databases

The SQL domain's tool is SQLite from the Python standard library. It needs the databases on disk and
the prompt files, which one script prepares:

```bash
python3 scripts/data/prepare_sql.py --out /workspace/data/sql     # Spider (~200 MB) + BIRD mini-dev (~350 MB)
```

It downloads the Spider archive and the BIRD dev archive, executes every Spider training gold query,
drops the ones that fail, time out, return nothing or repeat a question, keeps 6400 with a fixed seed,
and writes:

| File under `/workspace/data/sql` | Content |
| --- | --- |
| `databases/<db_id>/<db_id>.sqlite` | the databases; this directory is `SQL_DB_ROOT` |
| `train.jsonl` | 6400 training prompts (schema and question, gold SQL, `db_id`) |
| `spider_dev.jsonl`, `spider_test.jsonl` | Spider dev (1034 questions) and test (2147 questions on 40 unseen databases) |
| `bird_minidev.jsonl` | BIRD mini-dev, 500 questions on 11 databases, the evidence hint in the prompt; evaluation only |

If the node cannot reach HuggingFace or the BIRD mirror, download the two archives elsewhere and pass
`--spider-zip` / `--bird-dev-zip`.

## Training

All commands run inside the container, from `/workspace/MRPD`, with `MEGATRON_ROOT` and `PYTHONPATH`
set as in installation step 4. Every launcher validates its inputs on startup and names what is missing.
**`SAVE_DIR` is also the directory a run resumes from**: an existing directory resumes silently, so a
new run needs a new directory.

### 1. Initialization

Download the base model, convert it to torch_dist, and fine-tune it on ReTool's SFT traces (3 epochs):

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /workspace/ckpt/Qwen3-4B-Instruct-2507

HF_CHECKPOINT=/workspace/ckpt/Qwen3-4B-Instruct-2507 \
SAVE=/workspace/ckpt/Qwen3-4B-Instruct-2507/torch_dist \
    bash scripts/train/convert_hf_to_torch_dist.sh

hf download baohao/math-cold-start --repo-type dataset --local-dir /workspace/data/cold_start

MODEL=qwen3-4b \
HF_CHECKPOINT=/workspace/ckpt/Qwen3-4B-Instruct-2507 \
REF_LOAD=/workspace/ckpt/Qwen3-4B-Instruct-2507/torch_dist \
PROMPT_DATA=/workspace/data/cold_start/train.parquet \
SAVE_DIR=/workspace/runs/cold_start \
    bash scripts/train/cold_start.sh
```

slime saves the result as a training checkpoint (optimizer state included). Re-materialize clean
weights, torch_dist → HF → torch_dist:

```bash
COLD_START_RUN=/workspace/runs/cold_start
ITER=$(cat $COLD_START_RUN/latest_checkpointed_iteration.txt)

INPUT_DIR=$COLD_START_RUN/iter_$(printf '%07d' $ITER) \
OUTPUT_DIR=/workspace/ckpt/sft_hf \
ORIGIN_HF_DIR=/workspace/ckpt/Qwen3-4B-Instruct-2507 \
    bash scripts/train/convert_torch_dist_to_hf.sh

HF_CHECKPOINT=/workspace/ckpt/sft_hf SAVE=/workspace/ckpt/sft_torch_dist \
    bash scripts/train/convert_hf_to_torch_dist.sh
```

`/workspace/ckpt/sft_hf` and `/workspace/ckpt/sft_torch_dist` are the initialization of every model
below: the experts, the student and the baselines. They are also the launchers' defaults.

### 2. The experts

One GRPO run per domain, 64 prompts × 8 samples per step, learning rate 1e-6, clip (0.2, 0.28), no KL
penalty, trained asynchronously. The search expert needs the retriever of installation step 2, and the
SQL expert the data of installation step 6.

```bash
# ---- reasoning --------------------------------------------------------------
hf download baohao/math-train --repo-type dataset --local-dir /workspace/data/rl

MODEL=qwen3-4b \
HF_CHECKPOINT=/workspace/ckpt/sft_hf \
REF_LOAD=/workspace/ckpt/sft_torch_dist \
PROMPT_DATA=/workspace/data/rl/train.jsonl \
SAVE_DIR=/workspace/runs/grpo_math \
    bash scripts/train/teacher_grpo_math.sh
```

```bash
# ---- search -----------------------------------------------------------------
hf download baohao/search-train --repo-type dataset --local-dir /workspace/data/search_rl

MODEL=qwen3-4b \
HF_CHECKPOINT=/workspace/ckpt/sft_hf \
REF_LOAD=/workspace/ckpt/sft_torch_dist \
PROMPT_DATA=/workspace/data/search_rl/train.parquet \
SAVE_DIR=/workspace/runs/grpo_search \
    bash scripts/train/teacher_grpo_search.sh
```

```bash
# ---- text-to-SQL ------------------------------------------------------------
MODEL=qwen3-4b \
HF_CHECKPOINT=/workspace/ckpt/sft_hf \
REF_LOAD=/workspace/ckpt/sft_torch_dist \
PROMPT_DATA=/workspace/data/sql/train.jsonl \
SQL_DB_ROOT=/workspace/data/sql/databases \
SAVE_DIR=/workspace/runs/grpo_sql \
    bash scripts/train/teacher_grpo_sql.sh
```

Each run uses the whole node, so run them one after another. Each writes two things MRPD needs:

- **Checkpoints** in `SAVE_DIR/iter_XXXXXXX`, every 50 steps (`iter_0000049`, `iter_0000099`, ...). The
  expert at `iter_0000299`, i.e. after 300 steps, is the one the paper distills.
- **The training log** in `SAVE_DIR/rollout_debug/rollout_<k>.pt`, one file per step: the step's prompts,
  responses, loss masks, rewards and the sampling engine's log-probability of every sampled token.

The launchers default to 500 steps; the paper replays the first 300, so adding `NUM_ROLLOUT=300` to the
three commands is enough.

| | Reasoning | Search | SQL |
| --- | --- | --- | --- |
| Tool | Python sandbox | e5 retriever over `wiki-18`, top 3 passages | read-only SQLite |
| Turns | up to 16 (16 tool calls) | up to 8 | up to 8 (6 queries) |
| Response / context limit | 16k / 16k tokens | 4k per turn / 9k | 4k per turn / 9k |
| Reward | +1 / −1 on the boxed answer | +1 / −1 on normalized exact match | +1 / −1 on the execution result |
| Training prompts | 6400 from DAPO-Math-17k | 6500 from NQ and HotpotQA | 6400 from the Spider training set |

### 3. Merge the logs

`tools/merge_rollout_logs.py` writes one file per step with the three experts' step-k batches, tags
every sample with its domain, and records the batch shape in `manifest.json`:

```bash
python3 tools/merge_rollout_logs.py --steps 300 --out /workspace/runs/replay_logs/t3d \
    --teacher retool=/workspace/runs/grpo_math \
    --teacher search-r1=/workspace/runs/grpo_search \
    --teacher sql=/workspace/runs/grpo_sql
```

A replay step is then 3 × 64 × 8 = 1536 samples. Two options give the paper's controls, each in its own
output directory:

```bash
# 576 samples per step: the first 24 prompts of every expert's step (tag -g24 in the run name)
python3 tools/merge_rollout_logs.py --steps 300 --groups-per-teacher 24 --out /workspace/runs/replay_logs/t3d_g24 \
    --teacher retool=/workspace/runs/grpo_math \
    --teacher search-r1=/workspace/runs/grpo_search \
    --teacher sql=/workspace/runs/grpo_sql

# the 300 steps in a seeded random order, one permutation for the three experts (tag -shuf1)
python3 tools/merge_rollout_logs.py --steps 300 --shuffle-steps 1 --out /workspace/runs/replay_logs/t3d_shuf1 \
    --teacher retool=/workspace/runs/grpo_math \
    --teacher search-r1=/workspace/runs/grpo_search \
    --teacher sql=/workspace/runs/grpo_sql
```

### 4. MRPD

With `REPLAY_LOGS`, `opd_multi.sh` samples nothing: step k trains on `rollout_k.pt`, the three final
experts are loaded into the trainer as extra weight sets, each forwards its own domain's samples once
per step, and the trainer uses all eight GPUs. The batch size and the number of steps come from the
manifest. The retriever and the sandbox are not used; the prompt files still have to exist.

Set the environment once per shell. Every launcher of this section and the next two reads it:

```bash
export MODEL=qwen3-4b START_RETRIEVER=0 \
HF_CHECKPOINT=/workspace/ckpt/sft_hf REF_LOAD=/workspace/ckpt/sft_torch_dist \
RETOOL_PROMPT_DATA=/workspace/data/rl/train.jsonl \
SEARCH_R1_PROMPT_DATA=/workspace/data/search_rl/train.parquet \
SQL_PROMPT_DATA=/workspace/data/sql/train.jsonl SQL_DB_ROOT=/workspace/data/sql/databases \
MATH_TEACHER_LOAD=/workspace/runs/grpo_math MATH_TEACHER_CKPT_STEP=299 \
SEARCH_R1_TEACHER_LOAD=/workspace/runs/grpo_search SEARCH_R1_TEACHER_CKPT_STEP=299 \
SQL_TEACHER_LOAD=/workspace/runs/grpo_sql SQL_TEACHER_CKPT_STEP=299
```

Then MRPD is:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
REPLAY_LOGS=/workspace/runs/replay_logs/t3d \
MULTI_DOMAIN_USE_TASK_REWARD=1 \
TEACHER_DENSE_KL=reverse \
SAVE_DIR=/workspace/runs/mrpd \
    bash scripts/train/opd_multi.sh        # run name: grpo-opd-3d-replay-dense-reverse-t299-qwen3-4b
```

- `REPLAY_LOGS` selects the replay.
- `MULTI_DOMAIN_USE_TASK_REWARD=1` keeps the logged rewards, i.e. the PG term.
- `TEACHER_DENSE_KL=reverse` selects the dense distillation term (weight `TEACHER_DENSE_KL_COEF`,
  default 1; `TEACHER_TOPK`, default 256).
- `*_TEACHER_LOAD` is each expert's **run directory**, not an `iter_*` subdirectory and not an HF
  directory, and `*_TEACHER_CKPT_STEP` pins the iteration to distill.

The launcher derives the run name (the wandb group) from the knobs; `USE_WANDB=0` disables wandb.
Checkpoints land in `SAVE_DIR/iter_XXXXXXX` every 50 steps; the paper evaluates `iter_0000299`. The
launcher applies the [slime patches](patches/README.md) it needs to `third_party/slime` in place.

### 5. Variants on the same logs

Every row keeps the ordered replay and changes only the loss, the amount of logs or their order. All run
names end in `-qwen3-4b`. Every row also sets `REPLAY_LOGS` and its own `SAVE_DIR`, and reads the
environment block above.

| Variant | Launcher | Knobs | Run name |
| --- | --- | --- | --- |
| **MRPD**: PG + dense reverse KL | `opd_multi.sh` | `MULTI_DOMAIN_USE_TASK_REWARD=1 TEACHER_DENSE_KL=reverse` | `grpo-opd-3d-replay-dense-reverse-t299` |
| Dense reverse KL only | `opd_multi.sh` | `TEACHER_DENSE_KL=reverse` | `opd-3d-replay-dense-reverse-t299` |
| PG + dense forward KL | `opd_multi.sh` | `MULTI_DOMAIN_USE_TASK_REWARD=1 TEACHER_DENSE_KL=forward` | `grpo-opd-3d-replay-dense-forward-t299` |
| Dense forward KL only | `opd_multi.sh` | `TEACHER_DENSE_KL=forward` | `opd-3d-replay-dense-forward-t299` |
| PG + sampled-token KL | `opd_multi.sh` | `MULTI_DOMAIN_USE_TASK_REWARD=1` | `grpo-opd-3d-replay-t299` |
| Sampled-token KL only | `opd_multi.sh` | (none) | `opd-3d-replay-t299` |
| PG only (offline GRPO) | `grpo_multi.sh` | (none) | `grpo-3d-replay` |
| SFT on the logs | `grpo_multi.sh` | `REPLAY_LOSS=sft` | `sft-3d-replay` |
| Offline RAFT (the correct samples only) | `grpo_multi.sh` | `REPLAY_LOSS=rft` | `rft-3d-replay` |
| Weight of the dense KL | `opd_multi.sh` | `TEACHER_DENSE_KL_COEF=3` (or `0.3`, `0.1`) | `...-dense-reverse-b3-t299` |
| Weight of the sampled-token KL | `opd_multi.sh` | `OPD_KL_COEF=0.3` | `...-replay-b0.3-t299` |
| 576 samples per step | either | `REPLAY_LOGS=/workspace/runs/replay_logs/t3d_g24` | `...-replay-g24...` |
| Steps shuffled | either | `REPLAY_LOGS=/workspace/runs/replay_logs/t3d_shuf1` | `...-replay-shuf1...` |

For example:

```bash
# dense reverse KL only (no PG term)
REPLAY_LOGS=/workspace/runs/replay_logs/t3d TEACHER_DENSE_KL=reverse \
SAVE_DIR=/workspace/runs/opd_3d_replay_dense_reverse_t299 bash scripts/train/opd_multi.sh

# PG only: the experts' own objective on their logs, no expert queried
REPLAY_LOGS=/workspace/runs/replay_logs/t3d SAVE_DIR=/workspace/runs/grpo_3d_replay bash scripts/train/grpo_multi.sh

# offline RAFT, and SFT on every logged sample
REPLAY_LOGS=/workspace/runs/replay_logs/t3d REPLAY_LOSS=rft SAVE_DIR=/workspace/runs/rft_3d_replay bash scripts/train/grpo_multi.sh
REPLAY_LOGS=/workspace/runs/replay_logs/t3d REPLAY_LOSS=sft SAVE_DIR=/workspace/runs/sft_3d_replay bash scripts/train/grpo_multi.sh

# MRPD on the shuffled logs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
REPLAY_LOGS=/workspace/runs/replay_logs/t3d_shuf1 MULTI_DOMAIN_USE_TASK_REWARD=1 TEACHER_DENSE_KL=reverse \
SAVE_DIR=/workspace/runs/mrpd_shuf1 bash scripts/train/opd_multi.sh
```

To check the name a set of knobs gives without starting a run:

```bash
REPLAY=1 TEACHER_DENSE_KL=reverse bash tests/opd_run_name.sh    # prints grpo-opd-3d-replay-dense-reverse-t299-qwen3-4b
```

### 6. The same launchers without the replay

Without `REPLAY_LOGS` the two launchers sample online in the three environments, 24 prompts × 8 samples
per domain and step (576 in total), with the retriever and the sandbox running. These are the paper's
online baselines; they read the same environment block:

```bash
# MOPD: the experts score the student's own rollouts at the sampled token
SAVE_DIR=/workspace/runs/opd_3d_t299 bash scripts/train/opd_multi.sh                                       # opd-3d-t299
# MOPD + GRPO
MULTI_DOMAIN_USE_TASK_REWARD=1 SAVE_DIR=/workspace/runs/grpo_opd_3d_t299 bash scripts/train/opd_multi.sh   # grpo-opd-3d-t299
# online GRPO, and online RAFT
SAVE_DIR=/workspace/runs/grpo_3d bash scripts/train/grpo_multi.sh             # grpo-3d
RAFT=1 SAVE_DIR=/workspace/runs/raft_3d bash scripts/train/grpo_multi.sh      # raft-3d
```

The paper's other baselines (online PPO, the single-domain PPO experts, weight merging) are not part
of this repository.

## Evaluation

One script per domain. Per checkpoint it converts to HF if needed, serves the model with SGLang, runs
every set and tears down. Stop training on the node first; search expects the retriever on the host.

```bash
RUN_DIR=/workspace/runs/mrpd

NUM_SAMPLES_HARD=32 PRINT_TURNS=0 DEBUG_TRACE=0 RUN_DIR=$RUN_DIR bash scripts/eval/math.sh iter_0000299
PRINT_TURNS=0 DEBUG_TRACE=0 RUN_DIR=$RUN_DIR bash scripts/eval/search.sh iter_0000299
PRINT_TURNS=0 DEBUG_TRACE=0 RUN_DIR=$RUN_DIR bash scripts/eval/sql.sh iter_0000299

python3 scripts/eval/collect_results.py /workspace/runs     # one table of everything evaluated so far
```

This is the paper's protocol:

| Domain | Sets | Samples per problem |
| --- | --- | --- |
| Reasoning | AIME 2024, AIME 2025, AMC 2023 | 32 (`NUM_SAMPLES_HARD`; the script's default is 8) |
| | Minerva, OlympiadBench, MATH-500 | 4 |
| Search | NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue (2500 questions each, drawn once with a fixed seed), Bamboogle (125) | 1 |
| SQL | Spider dev, Spider test, BIRD mini-dev | 4 |

Every checkpoint is sampled at temperature 1. A domain's score is the unweighted mean over its sets,
and Overall is the mean of the three domain averages.

- A directory that already contains `config.json` is served as it is, with no conversion:
  `bash scripts/eval/math.sh /workspace/ckpt/sft_hf` evaluates the initialization.
- Several checkpoints in one invocation are served one after another
  (`RUN_DIR=/workspace/runs/mrpd bash scripts/eval/math.sh iter_0000099 iter_0000199 iter_0000299`).
- The conversion's config and tokenizer donor `ORIGIN_HF_DIR` defaults to `/workspace/ckpt/sft_hf`, and
  `SQL_DATA_DIR` to `/workspace/data/sql`.
- `search.sh` takes `MEM_FRACTION_STATIC=0.85` of every card; with the retriever on the same cards, set
  `MEM_FRACTION_STATIC=0.6`.
- The server takes `TP_SIZE × DP_SIZE` GPUs, by default 1 × 8. To leave GPUs free:
  `SGLANG_CUDA_VISIBLE_DEVICES=0,1 DP_SIZE=2`.

**Outputs** land next to the checkpoint: `<checkpoint>/hf/` (converted weights),
`<checkpoint>/<set>/results.jsonl` and `summary.json` (reasoning and SQL), and
`eval_search_r1_summary.json` (search). `EVAL_OUTPUT_ROOT` redirects them. `accuracy` is the mean
pass@1. Check `timeout_stats.timeout_rate` next to it: in reasoning and SQL a timed-out sample is left
out of its problem's mean. If the rate is not near zero, run the same command again: `RESUME=1` (the
default) redoes only the missing traces.

**Pass@k** is computed from the per-problem files behind the summaries, with no GPU:

```bash
python3 scripts/eval/collect_pass_at_k.py /workspace/runs --env math -k 1 4 8 16 32
python3 scripts/eval/collect_pass_at_k.py /workspace/runs --env sql -k 1 2 4
```

Keep `DATASETS`, `NUM_SAMPLES_*`, `MAX_NEW_TOKENS`, `SAMPLE_TIMEOUT`, `MAX_EXAMPLES_PER_SOURCE`,
`SAMPLE_SEED`, `MAX_TURNS`, `TOPK` and `MAX_CONTEXT_LEN` identical across the runs you compare.

## Tests

The unit tests need a Python with torch and the `third_party/slime` submodule checked out, and no GPU:
megatron, `torch.distributed` and slime's runtime are stubbed.

```bash
python3 tests/teacher_dense_kl_test.py        # the dense term: top-k, the fused log-prob function, the KLs and their gradients
python3 tests/dense_loss_patch_test.py        # the patched loss path end to end
python3 tests/teacher_forward_patch_test.py   # the per-domain expert forward and the replay prefetch
python3 tests/teacher_kl_metric_test.py       # train/teacher_kl-<domain>
python3 tests/merge_rollout_logs_test.py      # tools/merge_rollout_logs.py
python3 tests/replay_rft_test.py              # which samples REPLAY_LOSS=rft and RAFT=1 keep
python3 tests/grpo_reward_norm_test.py        # GRPO's reward normalisation
python3 tests/collect_pass_at_k_test.py       # scripts/eval/collect_pass_at_k.py
```

## Notes

- Every variable the launchers read: [`train/README.md`](train/README.md).
- The slime patches and what each does: [`patches/README.md`](patches/README.md).
- A Megatron checkout elsewhere than `/root/Megatron-LM`: `export MEGATRON_ROOT=/path/to/Megatron-LM`.
- The launchers assume a node with eight GPUs. Online runs split it into 2 for the trainer and 6 for
  the rollout engines; the replay gives all eight to the trainer.

## Acknowledgments

Built on [slime](https://github.com/THUDM/slime) for training and [SGLang](https://github.com/sgl-project/sglang)
for rollout and serving. The reasoning and search environments are adapted from
[ReTool](https://arxiv.org/abs/2504.11536) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1), and the
SQL environment uses [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/). We
appreciate their contributions to the community.
