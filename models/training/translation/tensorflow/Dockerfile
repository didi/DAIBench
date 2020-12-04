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
# RUN apt-get install -y python-pip python3-pip virtualenv htop
RUN pip3 install --upgrade pip -i https://pypi.douban.com/simple/
RUN pip3 install numpy scipy scikit-learn -i https://pypi.douban.com/simple/

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Mount data into the docker
ADD . /research/transformer
WORKDIR /research/transformer
RUN pip3 install -r requirements.txt -i https://pypi.douban.com/simple
ENTRYPOINT ["/bin/bash"]

