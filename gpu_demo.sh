#!/usr/bin/env python

NET='res101'
SESSION=1
EPOCH=4
CHECKPOINT=283995
#VIDEO_DIR=$1
IMAGE_DIR=$1
DATASET='vidor'
MODEL_PATH='/storage/dldi/PyProjects/FasterRCNN4VidVRDT1/data/output'

ffmpeg_path='/storage/dldi/PyProjects/ffmpeg-3.3.4/bin-linux/ffmpeg'

test_img_path='/storage/dldi/PyProjects/vidor/img_test/imageCache'
IMAGE_DIR=${test_img_path}

#if [[ ! -d imageCache  ]];then
#  mkdir imageCache
#else
#  echo
#fi

#${ffmpeg_path} -i ${VIDEO_DIR} ./imageCache/%6d.jpg

python demo.py --net ${NET} \
               --checksession ${SESSION} --checkepoch ${EPOCH} --checkpoint ${CHECKPOINT} \
               --cuda \
               --dataset ${DATASET} \
               --image_dir ${IMAGE_DIR} \
               --load_dir ${MODEL_PATH}

# rm -rf ./imageCache/*