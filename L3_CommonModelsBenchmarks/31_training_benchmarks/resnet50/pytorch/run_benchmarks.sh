#!/bin/bash

python multiproc.py --nnodes 1 --node_rank 0 --nproc_per_node 1 main.py --arch resnet50 --epochs 8 --batch-size 256 -j 10 --print-freq 10 --fp16 /data/ >& pytorch_resnet50_mixed_precision_1gpu.log
python multiproc.py --nnodes 1 --node_rank 0 --nproc_per_node 2 main.py --arch resnet50 --epochs 8 --batch-size 256 -j 10 --print-freq 10 --fp16 /data/ >& pytorch_resnet50_mixed_precision_2gpu.log
python multiproc.py --nnodes 1 --node_rank 0 --nproc_per_node 4 main.py --arch resnet50 --epochs 8 --batch-size 256 -j 10 --print-freq 10 --fp16 /data/ >& pytorch_resnet50_mixed_precision_4gpu.log
python multiproc.py --nnodes 1 --node_rank 0 --nproc_per_node 8 main.py --arch resnet50 --epochs 8 --batch-size 256 -j 10 --print-freq 10 --fp16 /data/ >& pytorch_resnet50_mixed_precision_8gpu.log
