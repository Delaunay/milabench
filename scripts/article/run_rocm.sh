#!/bin/bash

set -ex

export MILABENCH_GPU_ARCH=rocm
export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"

export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"


install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    virtualenv $MILABENCH_WORDIR/env

    git clone https://github.com/mila-iqia/milabench.git -b intel

    . $MILABENCH_WORDIR/env/bin/activate
    pip install -e $MILABENCH_WORDIR/milabench

    #
    # Install milabench's benchmarks in their venv
    #
    milabench install

    which pip

    (
        . $BENCHMARK_VENV/bin/activate
        which pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
    )

    #
    #   Generate/download datasets, download models etc...
    milabench prepare
}

if [ ! -d "$MILABENCH_WORDIR" ]; then
    install_prepare 
else
    echo "Reusing previous install"
    . $MILABENCH_WORDIR/env/bin/activate
fi

cd $MILABENCH_WORDIR

#
#   Run the benchmakrs
milabench run "$@"

#
#   Display report
milabench report --runs $MILABENCH_WORDIR/results/runs
