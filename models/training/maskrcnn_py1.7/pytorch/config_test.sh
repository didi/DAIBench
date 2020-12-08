## DL params
export EXTRA_PARAMS=""
export EXTRA_CONFIG='SOLVER.BASE_LR 0.08 SOLVER.MAX_ITER 40000 SOLVER.WARMUP_FACTOR 0.0001 SOLVER.WARMUP_ITERS 800 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (18000,24000) SOLVER.IMS_PER_BATCH 24 TEST.IMS_PER_BATCH 8 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 8000 NHWC True'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=test
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=02:00:00

## System config params
export DGXNGPU=1
export DGXSOCKETCORES=1
export DGXNSOCKET=1
export DGXHT=1         # HT is on is 2, HT off is 1