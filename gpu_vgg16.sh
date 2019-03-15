#!/usr/bin/env bash

GPU_ID=1
BATCH_SIZE=1
WORKER_NUMBER=2
LEARNING_RATE=0.1
DECAY_STEP=0.9

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda