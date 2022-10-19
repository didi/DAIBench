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

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod 755 cuda_11.8.0_520.61.05_linux.run
./cuda_11.8.0_520.61.05_linux.run
rm -f cuda_11.8.0_520.61.05_linux.run
