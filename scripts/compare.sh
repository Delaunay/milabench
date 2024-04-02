

count=10

export SLURM_TMPDIR="/Tmp/slurm.$SLURM_JOB_ID.0"
export ENV="$SLURM_TMPDIR/env"
export BASE="$SLURM_TMPDIR/base"

conda activate $ENV

for i in $(seq $count); do
    milabench run --config $MILABENCH_CONFIG --base $BASE
    rsync -avz $BASE/runs $HOME/lenovo_A100
done




# LOADER="dali"
# MODEL="resnet152"
# OUT="$BASE/${MODEL}_${LOADER}/"
# mkdir -p $OUT
# for i in $(seq $count); do
#     echo "${OUT}"
#     # milabench run --select resnet152_2
#     python /home/mila/d/delaunap/milabench/benchmarks/torchvision/main.py  \
#         --precision tf32-fp16 --lr 0.01 --no-stdout --epochs 25     \
#         --model $MODEL                                              \
#         --batch-size 256                                            \
#         --loader $LOADER > "${OUT}/$i.txt"
# done


# LOADER="pytorch"
# MODEL="resnet50"
# OUT="$BASE/${MODEL}_${LOADER}/"
# mkdir -p $OUT
# for i in $(seq $count); do
#     echo "${OUT}"
#     # milabench run --select resnet152_2
#     python /home/mila/d/delaunap/milabench/benchmarks/torchvision/main.py  \
#         --precision tf32-fp16 --lr 0.01 --no-stdout --epochs 25     \
#         --model $MODEL                                              \
#         --batch-size 64                                             \
#         --loader $LOADER > "${OUT}/$i.txt"
# done

# LOADER="pytorch"
# MODEL="resnet152"
# OUT="$BASE/${MODEL}_${LOADER}/"
# mkdir -p $OUT
# for i in $(seq $count); do
#     echo "${OUT}"
#     # milabench run --select resnet152_2
#     python /home/mila/d/delaunap/milabench/benchmarks/torchvision/main.py  \
#         --precision tf32-fp16 --lr 0.01 --no-stdout --epochs 25     \
#         --model $MODEL                                              \
#         --batch-size 256                                            \
#         --loader $LOADER > "${OUT}/$i.txt"
# done

# src="$SLURM_TMPDIR/base/runs"
# dest="$SLURM_TMPDIR/base/$NAME"
# mv $src $dest