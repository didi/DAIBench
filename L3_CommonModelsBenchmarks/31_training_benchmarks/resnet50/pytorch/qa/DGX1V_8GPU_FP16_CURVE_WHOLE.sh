#!/bin/bash

python multiproc.py --nproc_per_node 8 main.py -j5 -p 500 --arch resnet50 -b 256 --lr 0.8 --warmup 5 --epochs 90 --fp16 --static-loss-scale 256 --raport-file RN50-FP16-8GPU-raport.json $1

python ./qa/check_curves.py curve_baselines/JoC08_RN50_FP16_curve_baseline.json RN50-FP16-8GPU-raport.json
