#!/bin/bash

dir=$(pwd)
mkdir -p pytorch/dataset/coco2017; cd pytorch/dataset/coco2017
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/zips/test2017.zip; unzip test2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd $dir
