#!/usr/bin/env python

BATCH_SIZE=8
WORKER_NUMBER=0
LEARNING_RATE=0.01
SAVE_DIR='data/output'
DATASET=$1
NETWORK=$2
CHECK_SESSION=$3
CHECK_EPOCH=$4
CHECK_POINT=$5


# vidor, pascal_voc
echo ${DATASET}

# vgg16, res101
echo ${NETWORK}

python trainval_net.py \
                   --dataset ${DATASET} --net ${NETWORK} \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE \
                   --save_dir $SAVE_DIR \
                   --r True \
                   --checksession ${CHECK_SESSION}\
                   --checkepoch ${CHECK_EPOCH}\
                   --checkpoint ${CHECK_POINT}\
                   --mGPUs \
                   --cuda