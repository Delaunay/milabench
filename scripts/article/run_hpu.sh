#!/bin/bash

set -ex

export MILABENCH_GPU_ARCH=hpu
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
    git clone https://github.com/Delaunay/voir.git -b async_timer
    git clone https://github.com/Delaunay/torchcompat.git

    wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.15.1/habanalabs-installer.sh
    chmod +x habanalabs-installer.sh

    . $MILABENCH_WORDIR/env/bin/activate
    pip install -e $MILABENCH_WORDIR/milabench


    #
    # Install milabench's benchmarks in their venv
    #
    milabench install

    which pip
    pip install -e $MILABENCH_WORDIR/voir
    pip install -e $MILABENCH_WORDIR/torchcompat

    # Override dependencies for HPU
    # milabench needs pyhlml
    export HABANALABS_VIRTUAL_DIR=$MILABENCH_VENV
    ./habanalabs-installer.sh install -t dependencies --venv -y
    ./habanalabs-installer.sh install -t pytorch --venv -y


    (
        . $BENCHMARK_VENV/bin/activate
        which pip
        pip install -e $MILABENCH_WORDIR/voir
        pip install -e $MILABENCH_WORDIR/torchcompat

        (
            cd $MILABENCH_WORDIR/milabench/benchmarks/timm/pytorch-image-models; 
            git fetch origin; 
            git checkout cb0e439
        )

        # Override dependencies for HPU
        # benchmarks need pytorch
        export HABANALABS_VIRTUAL_DIR=$BENCHMARK_VENV 
        ./habanalabs-installer.sh install -t dependencies --venv -y
        ./habanalabs-installer.sh install -t pytorch --venv -y
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
