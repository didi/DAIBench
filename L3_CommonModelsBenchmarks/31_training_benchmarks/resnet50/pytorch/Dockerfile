FROM nvcr.io/nvidia/pytorch:22.09-py3

WORKDIR /research

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git 

ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# RUN apt-get install -y python-setuptools

RUN apt-get install -y virtualenv htop
RUN pip3 install git+https://github.com/NVIDIA/dllogger.git

# Mount data into the docker
ADD . /research/resnet


WORKDIR /research/resnet

ENTRYPOINT ["/bin/bash"]

