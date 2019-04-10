#!/usr/bin/env python

SESSION=1
EPOCH=6
CHECKPOINT=416

python demo.py --net vgg16 \
               --checksession ${SESSION} --checkepoch ${EPOCH} --checkpoint ${CHECKPOINT} \
               --cuda --load_dir path/to/model/directoy