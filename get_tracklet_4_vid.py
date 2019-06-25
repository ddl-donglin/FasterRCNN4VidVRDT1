from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from tqdm import tqdm

import shutil
import cv2

gpu_project_base_path = '/storage/dldi/PyProjects/FasterRCNN4VidVRDT1/'
gpu_ffmpeg_path = '/storage/dldi/PyProjects/ffmpeg-3.3.4/bin-linux/ffmpeg'
local_ffmpeg_path = '/home/daivd/PycharmProjects/ffmpeg-3.3.4/bin-linux/ffmpeg'
local_project_base_path = '/home/daivd/PycharmProjects/FasterRCNN4VidVRDT1/'

env = 'gpu'

if env == 'gpu':
    project_base_path = gpu_project_base_path
    ffmpeg_path = gpu_ffmpeg_path
else:
    project_base_path = local_project_base_path
    ffmpeg_path = local_ffmpeg_path


def extract_all_frames(video_path, out_path=None):
    if out_path is None:
        video_name = os.path.basename(video_path)[:-4]
        extract_frame_path = os.path.join(project_base_path, 'framesCache', video_name)
        try:
            os.makedirs(extract_frame_path)
        except OSError:
            print("The {} exists! Skipping!".format(extract_frame_path))
            return extract_frame_path
    else:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        extract_frame_path = out_path

    os.system(ffmpeg_path + ' -i ' + video_path + ' ' + extract_frame_path + '/%4d.jpg')

    return extract_frame_path


def get_anchor_frames(frames_path, jump=10):
    anchor_frames_path = os.path.join(frames_path, 'anchors')
    if not os.path.exists(anchor_frames_path):
        os.makedirs(anchor_frames_path)

    for root, dirs, files in os.walk(frames_path):
        for each_frame in files:
            frame_name = os.path.basename(each_frame)
            if int(frame_name[:-4]) % jump == 0:
                try:
                    shutil.copyfile(os.path.join(root, frame_name),
                                    os.path.join(anchor_frames_path, frame_name))
                except:
                    pass

    return anchor_frames_path


def get_anchor_dets(anchor_frames_path):
    os.system('bash ' + project_base_path + 'gpu_demo.sh ' + anchor_frames_path)
    return anchor_frames_path


def track_frames(frames_path, anchor_frames_path):
    anchor_names = list()
    anchors = list()
    for root, dirs, files in os.walk(anchor_frames_path):
        for each_anchor in tqdm(files):
            anchor_name = os.path.basename(each_anchor)
            anchor_names.append(anchor_name)
            anchors.append(cv2.imread(os.path.join(root, anchor_name)))
            with open(os.path.join(root, anchor_name[:-4] + '_det.json'), 'r') as in_f:
                anchor_bbox_json = json.load(in_f)


            for root, dirs, files in os.walk(frames_path):
                for each_frame in files:
                    frame_name = os.path.basename(each_frame)


def tracker(frames, init_bbox, tracker_type='KCF'):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if tracker_type not in tracker_types:
        tracker_type = tracker_types[2]
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    init_frame = cv2.imread(frames[0])
    ok = tracker.init(init_frame, init_bbox)

    if not ok:
        print("Cannot initiate!")

    bboxes = list()
    for i, each_frame in enumerate(frames):
        if i == 0:
            bboxes.append(init_bbox)
        else:
            ok, bbox = tracker.update(each_frame)
            if ok:
                bboxes.append(bbox)

    return bboxes


def visualize_track():
    pass


if __name__ == '__main__':
    test_vid_path = '/storage/dldi/PyProjects/vidor/img_test/6980260459.mp4'
    extract_frame_path = extract_all_frames(test_vid_path)
    print('---' * 20)
    print('extract frames finish!', extract_frame_path)
    anchor_frames_path = get_anchor_frames(extract_frame_path)
    print('===' * 20)
    print('get_anchor frames finish!', anchor_frames_path)
    anchor_frames_det_path = get_anchor_dets(anchor_frames_path)
    print('--==' * 20)
    print('get_anchor_frames_det finish!', anchor_frames_det_path)

    # with open('framesCache/6980260459/anchors/0010_det.json', 'r') as in_f:
    #     dets = in_f.readlines()
    # print(dets)
