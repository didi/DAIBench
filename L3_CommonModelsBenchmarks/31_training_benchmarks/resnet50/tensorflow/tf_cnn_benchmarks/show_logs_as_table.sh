#!/bin/bash
if [ $# -lt 1 ]
then
	  echo "Usage : $BASH_SOURCE logdir"
	    exit
fi

WORKDIR=$1

echo -e 'Model\tGPUs\tBatch\tPrecision\tThroughput'

for model in resnet50 resnet101 resnet152 inception3 inception4 # vgg16 vgg19 googlenet alexnet
do
  for ngpus in 1 2 4
  do
    for batch in 64 128 256
    do
      fp16_tp=`cat ${WORKDIR}/${model}_batch${batch}_gpu_x${ngpus}_fp16.log | grep "total images/sec" | awk '{print $3}'`
      fp32_tp=`cat ${WORKDIR}/${model}_batch${batch}_gpu_x${ngpus}_fp32.log | grep "total images/sec" | awk '{print $3}'`
      echo -e $model '\t' $ngpus '\t' $batch '\t' 'FP16' '\t' $fp16_tp 
      echo -e $model '\t' $ngpus '\t' $batch '\t' 'FP32' '\t' $fp32_tp
    done
  done
done

