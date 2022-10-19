#!/bin/bash
source /etc/os-release
case $ID in
ubuntu|debian)
  sudo apt update
  sudo apt install -y fio
  ;;
centos|rhel)
  sudo yum install -y epel-release
  sudo yum install -y fio
  ;;
*)
        exit 1
        ;;
esac
