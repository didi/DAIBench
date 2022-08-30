#!/bin/bash
source /etc/os-release
case $ID in
ubuntu|debian)
  sudo apt install -y iperf
  ;;
centos|rhel)
  sudo yum install -y epel-release
  sudo yum install -y iperf
  ;;
*)
        exit 1
        ;;
esac
