## DL params
export EXTRA_PARAMS='--batch-size=64 --warmup=650 --lr=3.2e-3 --wd=1.3e-4'

## System run parms
export DGXNNODES=1
export DGXSYSTEM='test'
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=20
export WALLTIME=01:00:00

## System config params
export DGXNGPU=1
export DGXSOCKETCORES=8
export DGXNSOCKET=1
export DGXHT=2         # HT is on is 2, HT off is 1