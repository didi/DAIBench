#!/bin/bash


source /etc/os-release
case $ID in
ubuntu|debian)
  sudo apt install -y gcc g++ build-essential linux-headers-$(uname -r)
  ;;
centos|rhel)
  sudo yum install -y gcc-c++ kernel-devel
  ;;
*)
	exit 1
	;;
esac


GPU_DRIVER_VERSION=515.65.01

wget https://cn.download.nvidia.cn/tesla/$GPU_DRIVER_VERSION/NVIDIA-Linux-x86_64-$GPU_DRIVER_VERSION.run
chmod 755 NVIDIA-Linux-x86_64-$GPU_DRIVER_VERSION.run
./NVIDIA-Linux-x86_64-$GPU_DRIVER_VERSION.run
rm -f NVIDIA-Linux-x86_64-$GPU_DRIVER_VERSION.run

CUDA_TOOLKIT_VERSION=11.7.1
wget https://developer.download.nvidia.com/compute/cuda/$CUDA_TOOLKIT_VERSION/local_installers/cuda_${CUDA_TOOLKIT_VERSION}_${GPU_DRIVER_VERSION}_linux.run
chmod 755 cuda_${CUDA_TOOLKIT_VERSION}_${GPU_DRIVER_VERSION}_linux.run
./cuda_${CUDA_TOOLKIT_VERSION}_${GPU_DRIVER_VERSION}_linux.run
rm -f cuda_${CUDA_TOOLKIT_VERSION}_${GPU_DRIVER_VERSION}_linux.run
