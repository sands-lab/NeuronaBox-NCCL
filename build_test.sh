#!/usr/bin/bash -i
set -e
eval "$(conda shell.bash hook)"
set -x
conda activate ~/env/nccl_mod
cd ..

rm -rf nccl-tests

git clone --filter=blob:none https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/

make -j NCCL_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX 

#cd ..
#make MPI=1 MPI_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX

set +ex