#!/bin/bash
if [ $# -lt 1 ]
then
  echo "Usage : $0 ngpus"
  exit
fi

ngpus=$1

if [ $ngpus -lt 2 ]
then
  echo "At least 2 GPUs needed!"
  exit
fi
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make

for item in all_gather_perf all_reduce_perf alltoall_perf broadcast_perf gather_perf hypercube_perf reduce_perf reduce_scatter_perf scatter_perf sendrecv_perf
do
  ./build/$item -b 8 -e 256M -f 2 -g $ngpus -d all > $item.log
done

