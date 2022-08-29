#!/bin/bash
fio --name=seq-read --ioengine=posixaio --rw=read --bs=4k --numjobs=1 --size=4g --iodepth=1 --runtime=60 --time_based --end_fsync=1 > diskio.log
rm -f seq-read.*
fio --name=random-read --ioengine=posixaio --rw=randread --bs=4k --numjobs=1 --size=4g --iodepth=1 --runtime=60 --time_based --end_fsync=1 >> diskio.log
rm -f random-read.*
fio --name=seq-write --ioengine=posixaio --rw=write --bs=4k --numjobs=1 --size=4g --iodepth=1 --runtime=60 --time_based --end_fsync=1 >> diskio.log
rm -f seq-write.*
fio --name=random-write --ioengine=posixaio --rw=randwrite --bs=4k --numjobs=1 --size=4g --iodepth=1 --runtime=60 --time_based --end_fsync=1 >> diskio.log
rm -f random-write.*
