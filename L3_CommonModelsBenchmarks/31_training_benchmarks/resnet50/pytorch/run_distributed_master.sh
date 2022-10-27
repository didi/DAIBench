#!/bin/bash

MASTER_IP="localhost"
MASTER_PORT="12345"

python multiproc.py --nnodes 2 --node_rank 0 --master_addr $MASTER_IP --master_port $MASTER_PORT --nproc_per_node 8 main.py --arch resnet50 --epochs 8 --batch-size 256 -j 40 --print-freq 10 --fp16 /data/

