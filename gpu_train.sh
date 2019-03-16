#!/usr/bin/env bash

GPU_ID=1
BATCH_SIZE=1
WORKER_NUMBER=2
LEARNING_RATE=0.01
SAVE_DIR='data/output'
DATASET=$1
NETWORK=$2

# vidor, pascal_voc
echo ${DATASET}

# vgg16, res101
echo ${NETWORK}

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset ${DATASET} --net ${NETWORK} \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE \
                   --save_dir $SAVE_DIR \
                   --cuda