#!/usr/bin/bash -i
set -e
eval "$(conda shell.bash hook)" > /dev/null
set -x
conda activate ~/env/nccl_mod

export CUDA_HOME=$CONDA_PREFIX

#awk '/#include/ && !modif { print "#define __STDC_FORMAT_MACROS 1"; modif=1 } {print}' src/enqueue.cc > tmpfile && mv tmpfile src/enqueue.cc
rm -rf build
make -j src.build
make install PREFIX=$CONDA_PREFIX

cd ..
rm -rf nccl-tests

git clone --filter=blob:none https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/

make -j NCCL_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX 

#cd ..
#make MPI=1 MPI_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX

set +ex