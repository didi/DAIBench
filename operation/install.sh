#!/bin/bash
CUDA_PATH=`whereis cuda | awk '{print $2}'`
export PATH=$CUDA_PATH/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
path=$PWD
source /etc/os-release
case $ID in
ubuntu|debian)
	apt install git -y
	apt install g++ -y
	apt install python-dev -y
	apt install openmpi-bin -y
	apt install libopenmpi-dev -y
	pip install mpi4py -y
	mpi_path="/usr/lib/openmpi"
	mpi_include_path="/usr/lib/openmpi/include"
	;;
centos|rhel)
	yum install git -y
	yum install gcc-c++ -y
	yum install python-devel -y
	yum install openmpi openmpi-devel openmpi-libs -y
	export PATH=$PATH:/usr/lib64/openmpi/bin
	export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib
	pip install mpi4py
	mpi_path="/usr/lib64/openmpi"
	mpi_include_path="/usr/include/openmpi-x86_64/"
	;;
*)
	exit 1
	;;
esac
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make install -j
while [ $? -ne 0 ]
do
	make install -j
done
ldconfig
cd $path
git clone https://github.com/baidu-research/DeepBench.git
cd DeepBench/code/nvidia
make CUDA_PATH=/usr/local/cuda CUDNN_PATH=/usr/local/cuda MPI_PATH=$mpi_path NCCL_PATH=$path/nccl MPI_INCLUDE_PATH=$mpi_include_path ARCH=sm_30,sm_32,sm_35,sm_50,sm_52,sm_60,sm_61,sm_62,sm_70

