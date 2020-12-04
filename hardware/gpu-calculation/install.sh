#!/bin/bash
source /etc/os-release
case $ID in
ubuntu|debian)
  sudo apt-get install -y gcc
  ;;
centos|rhel)
  sudo yum install -y gcc-c++
  ;;
*)
	exit 1
	;;
esac