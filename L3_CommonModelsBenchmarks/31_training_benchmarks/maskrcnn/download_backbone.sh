#!/bin/bash

dir=$(pwd)
mkdir -p pytorch/dataset/coco2017/models; cd pytorch/dataset/coco2017/models
wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
cd $dir