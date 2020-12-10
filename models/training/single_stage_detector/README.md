# 1. Problem
Object detection.

# 2. Directions

### Install Docker & nvidia-docker

### Steps to download data
```bash
bash download_dataset.sh
```

### ResNet34 pretrained weights
ResNet34 backbone is initialized with weights from PyTorch hub:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

By default, the code will automatically download the weights to
`$TORCH_HOME/models` (default is `~/.torch/models/`) and save them for later use.

Alternatively, you can manually download the weights with:
```
cd reference/single_stage_detector/
./download_resnet34_backbone.sh
```

Then use the downloaded file with `--pretrained-backbone <PATH TO WEIGHTS>` .

### Build Docker
```bash
CONT=`sudo docker build . | tail -n 1 | awk '{print $4}'`
sudo docker tag $CONT models/ssd
```

## Launch training

### Set Config file
Launch configuration in the `config_test.sh`.
By default, DATADIR would be `single_stage_detector/dataset`

```bash
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=test ./run.sub
```

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ILSVRC 2012 (from torchvision). Modifications to the backbone networks: remove conv_5x residual blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to stride1 (this increases the resolution of the feature map to which detector heads are attached), attach all 6 detector heads to the output of the last conv_4x residual block. Thus detections are attached to 38x38, 19x19, 10x10, 5x5, 3x3, and 1x1 feature maps.

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.23

### Evaluation frequency

### Evaluation thoroughness
All the images in COCO 2017 val data set.
