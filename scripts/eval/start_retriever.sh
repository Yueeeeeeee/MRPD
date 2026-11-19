#!/bin/bash
# Start the Search-R1 dense retriever (e5-base-v2 over wiki-18) on the HOST.
#
# This is the one search component that lives outside Docker: its conda env
# (faiss-gpu pins a torch that fights the training stack) sits on the persistent
# disk, so a VM that is stopped and restarted gets its retriever back with this
# one command instead of a reinstall. Containers reach it through --network host
# at http://127.0.0.1:8000/retrieve, which is every search script's default
# SEARCH_URL; they are told not to start their own with START_RETRIEVER=0.
#
# Usage (on the VM, not in the container), with the env already active:
#   conda activate /opd_workspace/envs/retriever
#   MRPD_STORAGE=/opd_workspace RETRIEVER_GPUS=0,1,2,3,4,5,6,7 \
#       bash scripts/eval/start_retriever.sh
# Without activating, it finds conda (anaconda / miniconda / miniforge) itself and
# activates RETRIEVER_CONDA_ENV -- that is how a systemd unit would run it.
#
# Runs in the foreground and exec's the server, so `nohup ... &`, tmux, or a
# systemd unit can own the process. Ctrl-C stops it.
#
# The index is loaded to GPU as fp16 and SHARDED across RETRIEVER_GPUS
# (retrieval_server.py: co.useFloat16 = True, co.shard = True), so each card in
# the group holds roughly index / N plus the encoder. That per-GPU number is what
# the training scripts need as RETRIEVER_RESERVE_GIB; this script prints an
# estimate, but read the real value off nvidia-smi once the server is up.
#
# Which GPUs: all eight (RETRIEVER_GPUS=0,1,2,3,4,5,6,7) on a node where training uses every
# card anyway: an 8-way shard is the cheapest per card, and one server outlives every run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# --- where things are ---------------------------------------------------------
# Same layout the README's docker run mounts into the container: the index at
# ${MRPD_STORAGE}/data/search_data is /workspace/data/search_data in there,
# and the HF cache is shared, so e5-base-v2 is downloaded once for both sides.
MRPD_STORAGE=${MRPD_STORAGE:-/opd_workspace}
DATA_DIR=${DATA_DIR:-${MRPD_STORAGE}/data/search_data}
INDEX_FILE=${INDEX_FILE:-${DATA_DIR}/e5_Flat.index}
CORPUS_FILE=${CORPUS_FILE:-${DATA_DIR}/wiki-18.jsonl}
export HF_HOME=${HF_HOME:-${MRPD_STORAGE}/cache/huggingface}

# --- the conda env ------------------------------------------------------------
# Three ways in, tried in this order:
#   1. an env is already active in this shell (CONDA_PREFIX set) and the caller
#      did not name one: use it as-is. `conda activate <env>` then run this.
#   2. RETRIEVER_CONDA_ENV names or points at an env: source conda.sh and
#      activate it. conda.sh is RETRIEVER_CONDA_SH if given; otherwise it is
#      found through `conda info --base` when conda is on PATH, else by looking
#      in the usual install places (miniconda3 / anaconda3 / miniforge3 under
#      the storage dir, then $HOME, then /opt).
#   3. RETRIEVER_CONDA_ENV="" (set, but empty): use whatever python is on PATH.
_RETRIEVER_CONDA_ENV_GIVEN=${RETRIEVER_CONDA_ENV+x}
RETRIEVER_CONDA_ENV=${RETRIEVER_CONDA_ENV-${MRPD_STORAGE}/envs/retriever}
RETRIEVER_CONDA_SH=${RETRIEVER_CONDA_SH:-}

# --- the server ---------------------------------------------------------------
RETRIEVER_GPUS=${RETRIEVER_GPUS:-0,1,2,3,4,5,6,7}
RETRIEVER_PORT=${RETRIEVER_PORT:-8000}
RETRIEVER_PATH=${RETRIEVER_PATH:-intfloat/e5-base-v2}
RETRIEVER_NAME=${RETRIEVER_NAME:-e5}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}
# Estimate inputs, matching the training scripts' derivation.
RETRIEVER_INDEX_GIB=${RETRIEVER_INDEX_GIB:-27}
RETRIEVER_ENCODER_GIB=${RETRIEVER_ENCODER_GIB:-2}

# ------------------------------------------------------------------------------

for f in "${INDEX_FILE}" "${CORPUS_FILE}"; do
    if [ ! -f "${f}" ]; then
        echo "[retriever] missing ${f}" >&2
        echo "[retriever] DATA_DIR=${DATA_DIR} must hold e5_Flat.index and wiki-18.jsonl;" >&2
        echo "[retriever] README, Installation step 2, has the download." >&2
        exit 1
    fi
done

_find_conda_sh() {
    if command -v conda >/dev/null 2>&1; then
        local base; base="$(conda info --base 2>/dev/null)"
        [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ] && { echo "${base}/etc/profile.d/conda.sh"; return 0; }
    fi
    local root
    for root in "${MRPD_STORAGE}" "${HOME:-/nonexistent}" /opt; do
        local flavour
        for flavour in miniconda3 anaconda3 miniforge3 conda; do
            [ -f "${root}/${flavour}/etc/profile.d/conda.sh" ] && { echo "${root}/${flavour}/etc/profile.d/conda.sh"; return 0; }
        done
    done
    return 1
}

if [ -n "${CONDA_PREFIX:-}" ] && [ -z "${_RETRIEVER_CONDA_ENV_GIVEN}" ]; then
    echo "[retriever] using the conda env already active here: ${CONDA_PREFIX}"
elif [ -n "${RETRIEVER_CONDA_ENV}" ]; then
    conda_sh="${RETRIEVER_CONDA_SH}"
    if [ -z "${conda_sh}" ] && ! conda_sh="$(_find_conda_sh)"; then
        echo "[retriever] cannot find conda.sh (looked via \`conda info --base\`, then for" >&2
        echo "[retriever] miniconda3/anaconda3/miniforge3 under ${MRPD_STORAGE}, \$HOME and /opt)." >&2
        echo "[retriever] Either \`conda activate\` the env and rerun, set RETRIEVER_CONDA_SH to" >&2
        echo "[retriever] your .../etc/profile.d/conda.sh, or RETRIEVER_CONDA_ENV='' for the PATH python." >&2
        exit 1
    fi
    if [ ! -f "${conda_sh}" ]; then
        echo "[retriever] RETRIEVER_CONDA_SH=${conda_sh} does not exist." >&2
        exit 1
    fi
    echo "[retriever] activating ${RETRIEVER_CONDA_ENV} via ${conda_sh}"
    # shellcheck disable=SC1090
    source "${conda_sh}"
    conda activate "${RETRIEVER_CONDA_ENV}"
fi
if [ -n "${CONDA_PREFIX:-}" ]; then
    # faiss-gpu-cu12 links cublas from the pip nvidia-* packages, not the system.
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
fi

# Fail here, not two minutes into loading a 27 GiB index.
#
# get_num_gpus() only enumerates devices -- it launches no kernel, so a wheel
# built without this card's architecture still reports every GPU and the check
# passes. On an H100 the run then dies inside the index load with
#
#   Faiss assertion 'err__ == cudaSuccess' failed in runL2Norm ...
#   CUDA error 209 no kernel image is available for execution on the device
#
# minutes later. So run the smallest search that reaches runL2Norm, the same
# kernel, and find out in milliseconds. faiss aborts the process on a CUDA
# failure rather than raising, so the exit code is what we read.
_faiss_probe=$(python - <<'PROBE' 2>&1
import faiss
import numpy as np

if faiss.get_num_gpus() == 0:
    print("NOGPU")
    raise SystemExit(0)
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, 8)
index.add(np.zeros((2, 8), dtype="float32"))
index.search(np.zeros((1, 8), dtype="float32"), 1)
print("OK")
PROBE
) || true

case "${_faiss_probe}" in
    *OK*) ;;
    *NOGPU*|*"No module named"*)
        echo "[retriever] this python has no GPU-enabled faiss ($(command -v python))." >&2
        echo "[retriever] Activate the retriever env, or check faiss-gpu-cu12 installed there." >&2
        exit 1 ;;
    *)
        echo "[retriever] faiss sees the GPUs but cannot run a kernel on them:" >&2
        echo "${_faiss_probe}" | tail -3 | sed 's/^/[retriever]   /' >&2
        echo "[retriever] 'no kernel image is available' (CUDA 209) means the wheel was" >&2
        echo "[retriever] built without this card's architecture. The prebuilt" >&2
        echo "[retriever] faiss-gpu-cu12 wheels carry no SM90, so an H100 needs faiss built" >&2
        echo "[retriever] from source with -DCMAKE_CUDA_ARCHITECTURES=\"80;90\"." >&2
        echo "[retriever] See the H100 note in docs/search.md." >&2
        exit 1 ;;
esac

n_gpus=$(awk -F, '{print NF}' <<< "${RETRIEVER_GPUS}")
estimate=$(awk -v idx="${RETRIEVER_INDEX_GIB}" -v n="${n_gpus}" -v enc="${RETRIEVER_ENCODER_GIB}" \
               'BEGIN { printf "%d", int(idx / n + enc + 0.999) }')

echo "[retriever] index   ${INDEX_FILE}"
echo "[retriever] corpus  ${CORPUS_FILE}"
echo "[retriever] python  $(command -v python)"
echo "[retriever] HF_HOME ${HF_HOME}  (shared with the container -- e5-base-v2 downloads once)"
echo "[retriever] GPUs    ${RETRIEVER_GPUS}  (${n_gpus}-way shard)  port ${RETRIEVER_PORT}"
echo "[retriever] estimated footprint ~${estimate} GiB per GPU. Once it is serving, confirm with"
echo "[retriever]     nvidia-smi --query-gpu=index,memory.used --format=csv"
echo "[retriever] and pass the rounded-up value to every search script as RETRIEVER_RESERVE_GIB."
echo "[retriever] Loading the index takes a few minutes the first time, 1-2 after that."

# No CUDA_LAUNCH_BLOCKING=1 here: it serialises every CUDA call and this server
# is on the rollout's critical path.
exec env CUDA_VISIBLE_DEVICES="${RETRIEVER_GPUS}" python "${REPO_DIR}/eval/search/local_dense_retriever/retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk "${RETRIEVER_TOPK}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_PATH}" \
    --faiss_gpu \
    --port "${RETRIEVER_PORT}"
