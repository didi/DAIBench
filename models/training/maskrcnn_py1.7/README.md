# 1. Problem 
This benchmark uses Mask R-CNN for object detection.
# 2. Directions

### Prepare dataset and model
```bash
bash download_dataset.sh
bash download_backbone.sh
```
By default, dataset `COCO2017` and model `R-50.pkl` would be mounted at `maskrcnn/pytorch/dataset/coco2017`

### Build docker
```bash
cd pytorch
CONT=`sudo docker build . | tail -n 1 | awk '{print $3}'`
sudo docker tag $CONT models/maskrcnn
```

### Hyperparameter settings
Hyperparameters are recorded in the `config_test.sh` files for each configuration and in `run_and_time.sh`.

### Start docker & run
```bash
bash run_with_docker.sh
```
# 3. Dataset/Environment

### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model
### Publication/Attribution

We use a version of Mask R-CNN with a ResNet50 backbone. See the following papers for more background:
[1] [Mask R-CNN](https://arxiv.org/abs/1703.06870) by Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, Mar 2017.
[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.


### Structure & Loss
Refer to [Mask R-CNN](https://arxiv.org/abs/1703.06870) for the layer structure and loss function.


### Weight and bias initialization
The ResNet50 base must be loaded from the provided weights. They may be quantized.


### Optimizer
We use a SGD Momentum based optimizer with weight decay of 0.0001 and momentum of 0.9.


# 5. Quality
### Quality metric
As Mask R-CNN can provide both boxes and masks, we evaluate on both box and mask mAP.

### Quality target
Box mAP of 0.377, mask mAP of 0.339 

### Evaluation frequency
Once per epoch, 118k.

### Evaluation thoroughness
Evaluate over the entire validation set. Use the [NVIDIA COCO API](https://github.com/NVIDIA/cocoapi/) to compute mAP.
