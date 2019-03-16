#!/usr/bin/env bash

GPU_ID=1
BATCH_SIZE=1
WORKER_NUMBER=2
LEARNING_RATE=0.01
SAVE_DIR='data/output'

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net res101 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE \
                   --save_dir $SAVE_DIR \
                   --cuda