#!/usr/bin/env python

IMAGE_DIR=$1
OUTPUT_BBOX='yes'
SAVE_FEAT='yes'

NET='res101'
SESSION=1
EPOCH=4
CHECKPOINT=283995
DATASET='vidor'
MODEL_PATH='/storage/dldi/PyProjects/FasterRCNN4VidVRDT1/data/output'

python demo.py --net ${NET} \
               --checksession ${SESSION} --checkepoch ${EPOCH} --checkpoint ${CHECKPOINT} \
               --cuda \
               --out_bbox ${OUTPUT_BBOX} \
               --save_feature ${SAVE_FEAT} \
               --dataset ${DATASET} \
               --image_dir ${IMAGE_DIR} \
               --load_dir ${MODEL_PATH}
