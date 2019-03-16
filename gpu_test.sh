#!/usr/bin/env bash

SESSION=1
EPOCH=6
CHECKPOINT=416

python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession ${SESSION} --checkepoch ${EPOCH} --checkpoint ${CHECKPOINT} \
                   --cuda