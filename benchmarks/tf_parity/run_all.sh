#!/usr/bin/env bash
# Reproduce the desc 2.1.1 (TensorFlow) vs desc 3.0 (PyTorch) parity test.
#
# Usage: benchmarks/tf_parity/run_all.sh [--parallel]
#   --parallel   run the 12 end-to-end trainings at once (timings then reflect a loaded machine)
# Environment:
#   DESC_PARITY_DIR  work directory for inputs, runs and results (default ./parity_work)
#   DESC_TF_SRC      directory holding the desc 2.1.1 package (default $DESC_PARITY_DIR/desc-2.1.1)
#   DESC_TF_VENV     Python 3.11 environment for desc 2.1.1 (default $DESC_PARITY_DIR/tf-venv)
# Behind a TLS-intercepting proxy uv may fail with "invalid peer certificate"; export UV_NATIVE_TLS=1 first.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export DESC_PARITY_DIR=${DESC_PARITY_DIR:-$PWD/parity_work}
export DESC_TF_SRC=${DESC_TF_SRC:-$DESC_PARITY_DIR/desc-2.1.1}
TF_VENV=${DESC_TF_VENV:-$DESC_PARITY_DIR/tf-venv}
PARALLEL=0; [ "${1:-}" = "--parallel" ] && PARALLEL=1
export PYTHONHASHSEED=0
mkdir -p "$DESC_PARITY_DIR"

echo "== environments"
# desc 3.0: the repository's own environment, CPU build of PyTorch
(cd "$REPO" && uv sync --extra cpu)
# desc 2.1.1: TensorFlow 2.15 and the louvain package, the newest versions it runs with (Python 3.11)
if [ ! -x "$TF_VENV/bin/python" ]; then
    uv venv --python 3.11 "$TF_VENV"
    uv pip install --python "$TF_VENV/bin/python" "tensorflow-cpu==2.15.1" "numpy<2" "pandas<2" "scanpy==1.9.8" \
        "anndata<0.11" louvain python-igraph natsort matplotlib h5py scikit-learn "setuptools<70"
fi
if [ ! -d "$DESC_TF_SRC/desc" ]; then
    mkdir -p "$DESC_TF_SRC"
    git -C "$REPO" archive e4116d6 desc | tar -x -C "$DESC_TF_SRC"
fi

# desc 2.1.1 switches matplotlib to TkAgg when DISPLAY is set, so run it without one
tf_py() { env -u DISPLAY TF_CPP_MIN_LOG_LEVEL=2 "$TF_VENV/bin/python" -W ignore "$@"; }
torch_py() { "$REPO/.venv/bin/python" -W ignore "$@"; }

echo "== shared inputs"
torch_py "$HERE/prep_data.py" "$REPO"

echo "== end-to-end runs (2 versions x 2 datasets x 3 seeds)"
pids=()
for ds in pbmc paul; do
    for seed in 201809 1 2; do
        for impl in tf torch; do
            out="$DESC_PARITY_DIR/runs/${impl}_${ds}_${seed}"; mkdir -p "$out"
            if [ "$impl" = tf ]; then cmd=(tf_py "$HERE/run_train.py" "$DESC_TF_SRC" "$ds" "$seed" "$out")
            else cmd=(torch_py "$HERE/run_train.py" "$REPO" "$ds" "$seed" "$out"); fi
            if [ $PARALLEL = 1 ]; then "${cmd[@]}" > "$out/log.txt" 2>&1 & pids+=($!)
            else "${cmd[@]}" > "$out/log.txt" 2>&1; grep -h DONE "$out/log.txt" | sed "s|^|${impl}_${ds}_${seed}: |"; fi
        done
    done
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "== deterministic parity (needs the TF-pretrained ae_weights.h5 from the seed-201809 runs)"
for ds in pbmc paul; do
    for mode in "" long; do
        tf_py "$HERE/tf_side.py" $ds $mode
        torch_py "$HERE/torch_side.py" $ds $mode
    done
done

echo "== Louvain: louvain package (2.1.1) vs igraph (3.0) on the same graphs"
tf_py "$HERE/louvain_tf.py"
torch_py "$HERE/louvain_torch.py"

echo "== comparisons"
torch_py "$HERE/compare_det.py" pbmc pbmc_long paul paul_long > /dev/null
torch_py "$HERE/compare_louvain.py" > /dev/null
torch_py "$HERE/compare_e2e.py" > /dev/null
echo "Results: $DESC_PARITY_DIR/det/compare_det.json, $DESC_PARITY_DIR/louvain.json, $DESC_PARITY_DIR/e2e.json"
echo "Reference results from the original run are in $HERE/results/"
