#!/bin/bash

# Choose the image you want to use
export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-v0.0.8

LOC="$SLURM_TMPDIR/$SLURM_ARRAY_TASK_ID"
cd $LOC

# Run milabench
docker run --rm --ipc=host --gpus=all           \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm bash -c "chmod -R 777 /milabench/envs/runs"
rsync -avz $(pwd)/results $HOME/run_v0.0.8_cpu8_opt13b


docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm --ipc=host --gpus=all      \
      -v $(pwd)/results:/milabench/envs/runs   \
      $MILABENCH_IMAGE                         \
      milabench run --select opt-1_3b

docker run --rm bash -c "chmod -R 777 /milabench/envs/runs"
rsync -avz $(pwd)/results $HOME/run_v0.0.8_cpu8_opt13b

