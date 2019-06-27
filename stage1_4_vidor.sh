#!/usr/bin/env bash

vid_dir=${1}

base_path='/storage/dldi/PyProjects/vidor'
#base_path='/home/daivd/PycharmProjects/vidor'
split_dir_path='train_vids'

python stage1_4_vidor.py --vidor 1 -bp ${base_path} -sp ${split_dir_path} -vd ${vid_dir} -ajp 30
