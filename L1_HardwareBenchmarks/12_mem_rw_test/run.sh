#!/bin/bash

#echo "Please download mlc_v3.9.tgz from 'https://software.intel.com/content/www/us/en/develop/articles/intelr-memory-latency-checker.html?wapkw=mlc'"
wget https://downloadmirror.intel.com/736634/mlc_v3.9a.tgz
tar zxvf mlc_v3.9a.tgz

LOG="$(pwd)"

cd Linux
sudo chmod +x mlc

cd /
echo "Test will take a while. Result would be save as 'mlc.log'"
$LOG/Linux/mlc --peak_injection_bandwidth -e > $LOG/mlc.log
