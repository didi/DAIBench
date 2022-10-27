python ./multiproc.py --nproc_per_node 16 ./main.py /data/imagenet --raport-file raport.json -j5 -p 100 --arch resnet50 --label-smoothing 0.1 --workspace $1 -b 64 --optimizer-batch-size 2048 --lr 2.048 --mom 0.875 --lr-schedule cosine --epochs 250 --warmup 8 --wd 3.0517578125e-05 --mixup 0.2 -c fanin --data-backend pytorch