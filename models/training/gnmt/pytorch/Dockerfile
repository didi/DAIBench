FROM nvcr.io/nvidia/pytorch:20.10-py3

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    apt-get install -y git

ADD . /workspace/pytorch
WORKDIR /workspace/pytorch

RUN pip install -r requirements.txt -i https://pypi.douban.com/simple/
