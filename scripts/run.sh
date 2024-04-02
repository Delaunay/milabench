
OUTPUT="lenovo.out"

rm -rf $OUTPUT
touch $OUTPUT

# --partition=staff-idt           \

sbatch  --reservation=milabench         \
        -w cn-d[003-004]                \
        --ntasks=1                      \
        --gpus-per-task=a100l:4         \
        --cpus-per-task=8               \
        --time=01:30:00                 \
        --ntasks-per-node=1             \
        --mem=128G                      \
        -o $OUTPUT                      \
        docker.sh                       \
        -a cuda                         \
        -b v0.0.8                       \
        -- --select opt-1_3b

tail -f $OUTPUT
