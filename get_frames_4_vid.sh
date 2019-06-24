#!/usr/bin/env bash

gpu_ffmpeg_path='/storage/dldi/PyProjects/ffmpeg-3.3.4/bin-linux/ffmpeg'
local_ffmpeg_path=''

VIDEO_PATH=$1
IMAGE_DIR=$2

${ffmpeg_path} -i ${VIDEO_PATH} ${IMAGE_DIR}/%6d.jpg
