#!/bin/bash

cd tf_cnn_benchmarks/
WORKDIR=$(date "+%Y%m%d%H%M%S")_logs
mkdir -p $WORKDIR

for model in resnet50 resnet101 resnet152 inception3 inception4 vgg16 vgg19 googlenet alexnet 
do
  for ngpus in 1 2 4 8
  do
    for batch in 64 128 256
    do
      rm -rf logs*
      python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=$batch \
        --model=$model --optimizer=momentum --variable_update=replicated \
        --nodistortions --gradient_repacking=4 --num_gpus=$ngpus --use_fp16 \
        --num_batches=800 --weight_decay=1e-4 \
        --train_dir=./logs/ >&  ${WORKDIR}/${model}_batch${batch}_gpu_x${ngpus}_fp16.log

      rm -rf logs*
      python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=$batch \
        --model=$model --optimizer=momentum --variable_update=replicated \
        --nodistortions --gradient_repacking=4 --num_gpus=$ngpus \
        --num_batches=800 --weight_decay=1e-4 \
        --train_dir=./logs/  >& ${WORKDIR}/${model}_batch${batch}_gpu_x${ngpus}_fp32.log
    done
  done
done
