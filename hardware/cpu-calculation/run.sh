#!/bin/bash

LOG="$(pwd)"

#cd l_mklb_p_2019.4.003/benchmarks_2019/linux/mkl/benchmarks/linpack
cd benchmarks_2022.0.2/linux/mkl/benchmarks/linpack/
./runme_xeon64

mv lin_xeon64.txt $LOG/mklbenchmark.log
