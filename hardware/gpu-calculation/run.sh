#!/bin/bash

CUDA_PATH=`whereis cuda | awk '{print $2}'`
LOG=`pwd`

export LD_LIBRARY_PATH=$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

cd $CUDA_PATH/samples/7_CUDALibraries/batchCUBLAS

sudo make

echo "Result would be saved as 'batchCUBLAS.log'"
./batchCUBLAS > $LOG/batchCUBLAS.log
