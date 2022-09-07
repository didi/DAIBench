#!/bin/bash

LOG=`pwd`

git clone https://github.com/NVIDIA/cuda-samples --recursive

cd $LOG/cuda-samples/Samples/1_Utilities/deviceQuery/
make
echo "Result would be saved as 'deviceQueryInfo.log'"
./deviceQuery > $LOG/deviceQueryInfo.log

cd $LOG/cuda-samples/Samples/1_Utilities/bandwidthTest
make
echo "Result would be saved as 'bandwidthTest.log'"
./bandwidthTest > $LOG/bandwidthTest.log

cd $LOG/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
make
echo "Result would be saved as 'p2pBandwidthLatencyTest.log'"
./p2pBandwidthLatencyTest > $LOG/p2pBandwidthLatencyTest.log
