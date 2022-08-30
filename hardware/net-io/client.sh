#!/bin/bash
if [ $# -lt 1 ]
then
  echo "Usage : client.sh host_ip"
  exit
fi
iperf -c $1 -p 9999 -t 60
