#!/usr/bin/bash -i
set -e
eval "$(conda shell.bash hook)"
set -x
conda activate ~/env/nccl_mod
export CUDA_HOME=$CONDA_PREFIX

#awk '/#include/ && !modif { print "#define __STDC_FORMAT_MACROS 1"; modif=1 } {print}' src/enqueue.cc > tmpfile && mv tmpfile src/enqueue.cc
rm -rf build
make -j src.build
make install PREFIX=$CONDA_PREFIX

set +ex
