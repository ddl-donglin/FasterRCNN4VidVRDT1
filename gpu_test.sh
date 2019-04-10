#!/usr/bin/env bash

DATASET=$1
NETWORK=$2
SESSION=$3
EPOCH=$4
CHECKPOINT=$5

python test_net.py --dataset ${DATASET} --net ${NETWORK} \
                   --checksession ${SESSION} --checkepoch ${EPOCH} --checkpoint ${CHECKPOINT} \
                   --cuda