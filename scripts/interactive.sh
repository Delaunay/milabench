
# source scripts/interactive.sh

# Intereactive session do not have SLURM_TMPDIR set
export SLURM_TMPDIR="/Tmp/slurm.$SLURM_JOB_ID.0"
export ENV="$SLURM_TMPDIR/env"
export BASE="$SLURM_TMPDIR/base"
export ARCH="cuda"
export PYTHON=3.9

if [ ! -d "$ENV" ] && [ "$ENV" != "base" ] && [ ! -d "$CONDA_ENVS/$ENV" ]; then
     conda create --prefix $ENV python=$PYTHON -y
fi
conda activate $ENV

export HF_HOME=$BASE/cache
export HF_DATASETS_CACHE=$BASE/cache
export TORCH_HOME=$BASE/cache
export XDG_CACHE_HOME=$BASE/cache
export MILABENCH_GPU_ARCH=$ARCH

export MILABENCH_DASH=no 
export PYTHONUNBUFFERED=1
export MILABENCH_BASE=$BASE
export MILABENCH_SOURCE="$HOME/milabench"

cd $SLURM_TMPDIR
git clone git@github.com:mila-iqia/milabench.git 
cd milabench
git checkout master
export MILABENCH_SOURCE="$SLURM_TMPDIR/milabench"


export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/standard.yaml"
python -m pip install -e $MILABENCH_SOURCE

module load gcc/9.3.0 
module load cuda/11.8

if [ ! -f "$BASE/install" ]; then
    milabench install --config $MILABENCH_CONFIG --base $BASE
fi

if [ ! -f "$BASE/prepare" ]; then
    milabench prepare --config $MILABENCH_CONFIG --base $BASE
fi

exec bash

#  pip install -e dependencies/cantilever
#  pip install -e dependencies/voir/
#  pip install -e dependencies/giving/
