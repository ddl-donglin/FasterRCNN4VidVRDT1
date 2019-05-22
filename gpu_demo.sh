#!/usr/bin/env python

SESSION=1
EPOCH=4
CHECKPOINT=283995
path_to_model_directory='/storage/dldi/PyProjects/FasterRCNN4VidVRDT1/data/output/res101/vidor'

python demo.py --net res101 \
               --checksession ${SESSION} --checkepoch ${EPOCH} --checkpoint ${CHECKPOINT} \
               --cuda --load_dir ${path_to_model_directory}