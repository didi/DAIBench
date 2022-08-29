#!/bin/bash
source /etc/os-release
case $ID in
ubuntu|debian)
  sudo apt install -y fid
  ;;
centos|rhel)
  sudo yum install -y epel-release
  sudo yum install -y fid
  ;;
*)
        exit 1
        ;;
esac
