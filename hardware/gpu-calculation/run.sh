#!/bin/bash

CUDA_PATH=`whereis cuda | awk '{print $2}'`
LOG=`pwd`

export LD_LIBRARY_PATH=$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

make

echo "Result would be saved as 'gpu_calc.log'"
./calc_gpu_peak_gflops.bin > $LOG/gpu_calc.log

make clean
