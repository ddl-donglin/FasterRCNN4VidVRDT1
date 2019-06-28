#!/usr/bin/env bash

prefix=$1
start=$2
end=$3

for (( i=${start}; i<=${end}; i++ ))
do
    dir=$(echo ${prefix}${i}|awk '{printf("%04d\n",$0)}')
    bash stage1_4_vidor.sh ${dir}
done