#!/bin/bash

WORKDIR=$(date "+%Y%m%d%H%M%S")_logs
mkdir -p $WORKDIR

# Caffe models

caffe_models=`ls caffe_models`

for model_file in $caffe_models
do
  batch=1

  while [ $batch -le 128 ]
  do
    echo 'Model = ' $model_file ', Batch Size = ' $batch
    trtexec --deploy=caffe_models/${model_file} --output=prob --warmUp=2000 --batch=$batch --workspace=1024 --noTF32 >& $WORKDIR/${model_file%.*}_batch${batch}_fp32.log
    trtexec --deploy=caffe_models/${model_file} --output=prob --warmUp=2000 --batch=$batch --workspace=1024 >& $WORKDIR/${model_file%.*}_batch${batch}_tf32.log
    trtexec --deploy=caffe_models/${model_file} --output=prob --warmUp=2000 --batch=$batch --workspace=1024 --useCudaGraph >& $WORKDIR/${model_file%.*}_batch${batch}_tf32_g.log
    trtexec --deploy=caffe_models/${model_file} --output=prob --warmUp=2000 --batch=$batch --workspace=1024 --noTF32 --useCudaGraph >& $WORKDIR/${model_file%.*}_batch${batch}_fp32_g.log
    trtexec --deploy=caffe_models/${model_file} --output=prob --warmUp=2000 --batch=$batch --workspace=1024 --fp16 >& $WORKDIR/${model_file%.*}_batch${batch}_fp16.log
    trtexec --deploy=caffe_models/${model_file} --output=prob --warmUp=2000 --batch=$batch --workspace=1024 --int8 >& $WORKDIR/${model_file%.*}_batch${batch}_int8.log
    batch=`expr $batch \* 2`
  done

done
