#!/bin/bash
SCRIPTS_ROOT=$(pwd)
DATA_ROOT=/data/ImageNet
TAR_PATH=${DATA_ROOT}/TAR/
RAW_PATH=${DATA_ROOT}/RAW/

mkdir -p $RAW_PATH && cd $RAW_PATH

mkdir train && cd train
tar -xf ${TAR_PATH}/ILSVRC2012_img_train.tar 
find . -name "*.tar" | while read NAME; do
  mkdir -p "${NAME%.tar}"
  tar -xf "${NAME}" -C "${NAME%.tar}"
  rm -f "${NAME}"
done
cd ..
mkdir val && cd val && tar -xf ${TAR_PATH}/ILSVRC2012_img_val.tar
${SCRIPTS_ROOT}/valprep.sh
