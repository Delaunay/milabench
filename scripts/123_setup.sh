

export MILABENCH_BASE="$(pwd)/workspace"
export MILABENCH_CONFIG="$(pwd)/config/standard.yaml"


module load cuda/11.8

rm -rf $MILABENCH_BASE/extra
rm -rf $MILABENCH_BASE/runs

milabench install 