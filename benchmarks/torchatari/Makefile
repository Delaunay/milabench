# Use global base if possible
ifndef MILABENCH_BASE
	MILABENCH_BASE="base"
endif

export MILABENCH_BASE

BENCH_NAME=torchatari
MILABENCH_CONFIG=dev.yaml
MILABENCH_ARGS=--config $(MILABENCH_CONFIG) --base $(MILABENCH_BASE)

all:
	install prepare single gpus nodes

install:
	milabench install $(MILABENCH_ARGS) --force

prepare:
	milabench prepare $(MILABENCH_ARGS)

tests: 
	MILABENCH_CPU_AUTO=1 CUDA_VISIBLE_DEVICES=0,1 milabench run $(MILABENCH_ARGS)

single:
	MILABENCH_CPU_AUTO=1 milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)

gpus:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-gpus

nodes:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-nodes
