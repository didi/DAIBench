#!/bin/bash

CUDA_PATH=`whereis cuda | awk '{print $2}'`
LOG=`pwd`

export LD_LIBRARY_PATH=$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

make

echo "Result would be saved as 'gpu_info.log'"
./calc_gpu_peak_gflops.bin > $LOG/gpu_info.log

SM_VERSION=`cat $LOG/gpu_info.log | grep "Compute Capability" | head -n 1 | awk '{ print $4 }'`
# echo $SM_VERSION

git clone https://github.com/NVIDIA/cutlass --recursive
cd cutlass
mkdir build
cd build/
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake .. -DCUTLASS_NVCC_ARCHS=$SM_VERSION
make cutlass_profiler -j12
./tools/profiler/cutlass_profiler --op_class=simt --m=8192 --n=8192 --k=8192 > $LOG/cutlass_simt.log
./tools/profiler/cutlass_profiler --op_class=tensorop --m=8192 --n=8192 --k=8192 > $LOG/cutlass_tensorop.log

cd ../..
make clean

