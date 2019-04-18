from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pdb
import pickle
import pprint
import sys
import time
import _init_paths

import cv2
import numpy as np
import torch
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import vis_detections
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from torch.autograd import Variable


det_file = '/storage/dldi/PyProjects/FasterRCNN4VidVRDT1/output/res101/vidor_2019_test/faster_rcnn_10/detections.pkl'
save_name = 'faster_rcnn_10'
imdb, roidb, ratio_list, ratio_index = combined_roidb('vidor_2019_test', False)
imdb.competition_mode(on=True)
output_dir = get_output_dir(imdb, save_name)

with open(det_file, 'rb') as f:
    all_boxes = pickle.load(f)

print('Evaluating detections')
imdb.evaluate_detections(all_boxes, output_dir)