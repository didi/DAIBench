#!/bin/bash
docker pull quay.io/zhaoyongke2019/daibench:22.09-oneflow
echo "ImageNet OFRecords needed in /data/"
echo "All environments are prepared in docker image."
docker run -ti --gpus all -v /data:/data --shm-size 8G docker pull quay.io/zhaoyongke2019/daibench:22.09-oneflow

