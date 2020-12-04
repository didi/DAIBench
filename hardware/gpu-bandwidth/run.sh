#!/bin/bash

CUDA_PATH=`whereis cuda | awk '{print $2}'`
LOG=`pwd`

cd $CUDA_PATH/samples/1_Utilities/bandwidthTest

sudo make

echo "Result would be saved as 'bandwidthTest.log'"
./bandwidthTest > $LOG/bandwidthTest.log
