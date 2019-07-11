#!/usr/bin/env bash

start=$1
end=$2

for i in $(seq ${start} ${end})
do
    dir=$(echo ${i}|awk '{printf("%04d\n",$0)}')
    bash stage1_4_vidor.sh ${dir}
done

# bash tracking_seg.sh 0000 0010
