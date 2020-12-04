FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils=1.2.29ubuntu0.1 \
                       libglib2.0-0=2.48.2-0ubuntu4.4 \
                       libsm6=2:1.2.2-1 \
                       libxext6=2:1.3.3-1 \
                       libxrender-dev=1:0.9.9-0ubuntu1 \
                       git

RUN pip install ninja==1.8.2.post2 \
                yacs==0.1.5 \
                cython==0.29.5 \
                matplotlib==3.0.2 \
                opencv-python==4.0.0.21 \
                mlperf_compliance==0.0.10 \
                torchvision==0.2.2 -i https://mirrors.aliyun.com/pypi/simple/

# install pycocotools
WORKDIR /

RUN git clone https://github.com/cocodataset/cocoapi.git \
&& cd cocoapi/PythonAPI \
&& git reset --hard ed842bffd41f6ff38707c4f0968d2cfd91088688 \
&& python setup.py build_ext install
