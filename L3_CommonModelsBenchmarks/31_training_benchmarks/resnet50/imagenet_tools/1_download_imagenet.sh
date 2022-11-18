#!/bin/bash
SCRIPTS_ROOT=$(pwd)
DATA_ROOT=/data/ImageNet

echo "DATA_ROOT : " $DATA_ROOT
mkdir -p $DATA_ROOT  && cd $DATA_ROOT
mkdir TAR && cd TAR/
echo "Downloading ILSVRC2012_img_train.tar, 138G, 1281167 images"
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
echo "Downloading ILSVRC2012_img_val.tar, 6.3G, 50000 images"
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

