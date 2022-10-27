#!/bin/bash

git clone https://github.com/horovod/horovod.git
cd horovod/examples/tensorflow/

horovodrun -np 1 -H localhost:1 --mpi python tensorflow_synthetic_benchmark.py --model ResNet50 --batch-size 128 --num-batches-per-iter 16 --num-iters 100 >& horovod_training_1gpu.log
horovodrun -np 2 -H localhost:2 --mpi python tensorflow_synthetic_benchmark.py --model ResNet50 --batch_size 128 --num-batches-per-iter 16 --num-iters 100 >& horovod_training_2gpu.log
horovodrun -np 4 -H localhost:4 --mpi python tensorflow_synthetic_benchmark.py --model ResNet50 --batch_size 128 --num-batches-per-iter 16 --num-iters 100 >& horovod_training_4gpu.log
horovodrun -np 8 -H localhost:8 --mpi python tensorflow_synthetic_benchmark.py --model ResNet50 --batch_size 128 --num-batches-per-iter 16 --num-iters 100 >& horovod_training_8gpu.log
horovodrun -np 16 -H server1:8,server2:8 --mpi tensorflow_synthetic_benchmark.py --model ResNet50 --batch_size 128 --num-batches-per-iter 16 --num-iters 100 >& horovod_training_16gpu.log
horovodrun -np 32 -H server1:8,server2:8,server3:8,server4:8 --mpi tensorflow_synthetic_benchmark.py --model ResNet50 --batch_size 128 --num-batches-per-iter 16 --num-iters 100 >& horovod_training_32gpu.log
