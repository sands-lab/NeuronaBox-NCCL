#!/bin/bash -i
set -ex

conda activate ~/env/nccl_mod
echo conda prefix=$CONDA_PREFIX

cd ..

rm -rf nccl-tests

git clone --filter=blob:none https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
make -j NCCL_HOME=`readlink -f ../nccl/build`
cd .

make MPI=1 MPI_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX

set +ex