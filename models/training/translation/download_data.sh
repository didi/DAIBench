#!/bin/bash
wget https://pts-source.s3.didiyunapi.com/mlperf/dataset/newstest2014.de -O tensorflow/newstest2014.de
wget https://pts-source.s3.didiyunapi.com/mlperf/dataset/newstest2014.en -O tensorflow/newstest2014.en

mkdir raw_data
cd raw_data
wget https://pts-source.s3.didiyunapi.com/mlperf/dataset/training-parallel-europarl-v7.tgz
wget https://pts-source.s3.didiyunapi.com/mlperf/dataset/training-parallel-commoncrawl.tgz
wget https://pts-source.s3.didiyunapi.com/mlperf/dataset/training-parallel-nc-v12.tgz
wget https://pts-source.s3.didiyunapi.com/mlperf/dataset/dev17.tgz
mv dev17.tgz dev.tgz

cd ..
sudo yum install -y python3
python3 data_download.py --raw_dir raw_data

# wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.en -O tensorflow/newstest2014.en
# wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.de -O tensorflow/newstest2014.de