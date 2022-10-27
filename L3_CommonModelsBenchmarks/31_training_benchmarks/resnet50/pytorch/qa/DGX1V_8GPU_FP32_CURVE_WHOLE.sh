#!/bin/bash

python multiproc.py --nproc_per_node 8 main.py -j5 -p 500 --arch resnet50 -b 256 --lr 0.8 --warmup 5 --epochs 90 --raport-file RN50-FP32-8GPU-raport.json $1

python ./qa/check_curves.py curve_baselines/JoC08_RN50_FP32_curve_baseline.json RN50-FP32-8GPU-raport.json
