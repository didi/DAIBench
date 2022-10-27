python3 imagenet_ofrecord.py  \
--train_directory /data/ImageNet/RAW/train \
--output_directory /data/ImageNet/OFRecords/train   \
--label_file imagenet_lsvrc_2015_synsets.txt   \
--shards 256  --num_threads 8 --name train  \
--bounding_box_file imagenet_2012_bounding_boxes.csv   \
--height 224 --width 224

python3 imagenet_ofrecord.py  \
--validation_directory /data/ImageNet/RAW/val  \
--output_directory /data/ImageNet/OFRecords/validation \
--label_file imagenet_lsvrc_2015_synsets.txt --name validation  \
--shards 256 --num_threads 8 --name validation \
--bounding_box_file imagenet_2012_bounding_boxes.csv  \
--height 224 --width 224
