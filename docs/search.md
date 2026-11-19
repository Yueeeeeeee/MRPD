# The search environment's retriever

The search domain (the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) setting) answers questions with a
dense retriever over Wikipedia: `intfloat/e5-base-v2` over the `wiki-18` corpus, served on GPU by
`eval/search/local_dense_retriever/retrieval_server.py`. Training the search expert and evaluating on search reach it
over HTTP at `SEARCH_URL` (default `http://127.0.0.1:8000/retrieve`).

The commands are in the [main README, installation step 2](../README.md#2-the-search-retriever-host). This page explains
them. Paths are the README's: on the host the storage root is `MRPD_STORAGE=/opd_workspace`, which the container
sees as `/workspace` (`${MRPD_STORAGE}/data` is `/workspace/data`, and so on).

| What | On the host | In the container |
| --- | --- | --- |
| Conda environment of the retriever | `${MRPD_STORAGE}/envs/retriever` | not mounted; the retriever runs on the host |
| Index and corpus (`DATA_DIR`) | `${MRPD_STORAGE}/data/search_data` | `/workspace/data/search_data` |
| Hugging Face cache (`HF_HOME`) | `${MRPD_STORAGE}/cache/huggingface` | `/workspace/cache/huggingface` |

## Why a separate environment, on the host

`faiss-gpu` pins a torch that conflicts with the training stack, so the retriever gets its own conda environment. It
lives on the persistent disk and outside Docker: a machine that is stopped and restarted gets its retriever back with
one command, and one server stays up across containers and runs. The container reaches it through `--network host`.

Install torch from pip, not from conda (conda's MKL breaks `import torch`).

## H100 (SM90): faiss from conda-forge

The PyPI wheel `faiss-gpu-cu12` carries no SM90 kernels (verified on 1.14.1.post1) and says so only at the first
kernel launch: `pip install` succeeds and `faiss.get_num_gpus()` reports every card, because enumerating devices
launches nothing. conda-forge's `faiss-gpu` does carry SM90 (verified on 1.14.3), so there is nothing to build. This is
why the README has two variants of the environment.

Order matters: conda first, pip after. Installing conda's faiss into an environment that already has pip's torch does
not work; build the environment around faiss.

Verify after the pip installs, because pip's `nvidia-*` packages can shadow the CUDA libraries conda's faiss links
against. The check in the README runs one real search:

```bash
python -c "
import faiss, numpy as np
r = faiss.StandardGpuResources(); i = faiss.GpuIndexFlatL2(r, 8)
i.add(np.zeros((2, 8), 'float32')); i.search(np.zeros((1, 8), 'float32'), 1)
print('OK', faiss.__version__)"
```

`OK` means it runs on this card; an abort with CUDA error 209 means it does not. Never use `faiss.get_num_gpus()` as
the check: it passes either way. `scripts/eval/start_retriever.sh` runs the same search at startup, so a bad
environment fails in milliseconds rather than after the index is in memory.

If a future conda build also lacks SM90, `cuobjdump` says which architectures a binary has. The kernels are not in the
SWIG module (`faiss/_swigfaiss.so` is the Python binding) but in the bundled `libfaiss` in a sibling `*.libs`
directory, so scan all of them:

```bash
SP=$(python -c "import faiss, os; print(os.path.dirname(os.path.dirname(faiss.__file__)))")
while IFS= read -r -d '' so; do
    archs=$(cuobjdump --list-elf "$so" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
    [ -n "$archs" ] && printf '%s\n    %s\n' "$so" "$archs"
done < <(find "$SP" -name '*faiss*' \( -name '*.so' -o -name '*.so.*' \) -print0)
```

## Retriever index and corpus

The server needs `e5_Flat.index` and `wiki-18.jsonl` in `DATA_DIR`. The index ships as split parts and the corpus is
gzipped; the README's commands download, join and unpack them. About 60 to 70 GB are downloaded and 132 GB remain
after the parts are deleted. The retriever model (`intfloat/e5-base-v2`, override with `RETRIEVER_PATH`) is pulled on
first use. The first start loads the index, which takes a few minutes.

## Starting it

`scripts/eval/start_retriever.sh` runs on the host. It reads:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MRPD_STORAGE` | `/opd_workspace` | the storage root |
| `DATA_DIR` | `${MRPD_STORAGE}/data/search_data` | where the index and the corpus are |
| `RETRIEVER_CONDA_ENV` | `${MRPD_STORAGE}/envs/retriever` | the environment to activate; an environment already active in the shell is used as it is, and `RETRIEVER_CONDA_ENV=""` uses the Python on `PATH` |
| `RETRIEVER_CONDA_SH` | found automatically | `conda.sh`, if conda is somewhere unusual |
| `RETRIEVER_GPUS` | `0,1,2,3,4,5,6,7` | the cards the index is sharded over |
| `RETRIEVER_PORT` | `8000` | the port; `SEARCH_URL` is then `http://127.0.0.1:8000/retrieve` |

It probes faiss, then runs the server in the foreground, so `nohup`, tmux or a systemd unit can own the process.

The launchers of the search expert and of online GRPO can also start a retriever themselves, inside the container, and
stop it on exit: `START_RETRIEVER=1 DATA_DIR=/workspace/data/search_data`. The retriever's conda environment then has
to exist inside the container (`RETRIEVER_CONDA_ENV=""` reuses the current Python). The README's `docker run` sets
`START_RETRIEVER=0`, which tells every script to use the server on the host.

## GPU memory next to training

The index is cloned to the GPUs as fp16 and sharded across the visible cards, not replicated, so a server on eight
cards costs each of them an eighth of the index plus the encoder: the index is about 27 GiB as fp16, so about 5 to
7 GB per card. Every training script reads
`nvidia-smi --query-gpu=memory.used` at launch, subtracts what is resident from SGLang's memory fraction, re-derives
the per-engine concurrency cap and prints what it chose:

```
[retriever] most-used GPU before launch: 6210 MiB -> reserving 7 GiB/GPU (override with RETRIEVER_RESERVE_GIB)
[gpu] memory.total=40960 MiB, base 0.78, minus 7 GiB retriever
[gpu] -> sglang-mem-fraction-static 0.605
[gpu] sglang-server-concurrency 25 per engine
```

`RETRIEVER_RESERVE_GIB=N` overrides the measurement (`0` if the retriever runs on another machine). The evaluation
script `scripts/eval/search.sh` takes `MEM_FRACTION_STATIC=0.85` of every card regardless of the retriever; with the
retriever on the same cards, set `MEM_FRACTION_STATIC=0.6`.
