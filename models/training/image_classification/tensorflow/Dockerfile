FROM nvcr.io/nvidia/tensorflow:20.10-tf1-py3

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
RUN pip3 install --upgrade numpy scipy sklearn -i https://pypi.douban.com/simple/


# Mount data into the docker
ADD . /research/resnet


WORKDIR /research/resnet
RUN pip3 install -r official/requirements.txt -i https://pypi.douban.com/simple/


ENTRYPOINT ["/bin/bash"]

