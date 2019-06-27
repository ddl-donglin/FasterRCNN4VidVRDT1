#!/usr/bin/env bash

prefix=$1
start=$2
end=$3

for (( i=${start}; i<=${end}; i++ ))
do
   bash stage1_4_vidor.sh ${prefix}${i}
done