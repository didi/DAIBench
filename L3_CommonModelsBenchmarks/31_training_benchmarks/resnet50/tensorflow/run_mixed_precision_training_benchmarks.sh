#!/bin/bash
cd /workspace/nvidia-examples/cnn
mpirun --allow-run-as-root -np 1 python -u ./resnet.py --batch_size 256 --num_iter 800 --precision fp16 --iter_unit batch --layers 50 --use_xla >& mixed_precision_training_1gpu.log
mpirun --allow-run-as-root -np 2 python -u ./resnet.py --batch_size 256 --num_iter 800 --precision fp16 --iter_unit batch --layers 50 --use_xla >& mixed_precision_training_2gpu.log
mpirun --allow-run-as-root -np 4 python -u ./resnet.py --batch_size 256 --num_iter 800 --precision fp16 --iter_unit batch --layers 50 --use_xla >& mixed_precision_training_4gpu.log
mpirun --allow-run-as-root -np 8 python -u ./resnet.py --batch_size 256 --num_iter 800 --precision fp16 --iter_unit batch --layers 50 --use_xla >& mixed_precision_training_8gpu.log

