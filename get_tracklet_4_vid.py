from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import shutil
import cv2
from tqdm import tqdm

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
            print("The {} exists! Skip extracting frames!".format(extract_frame_path))
            return extract_frame_path
    else:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        extract_frame_path = out_path

    os.system(ffmpeg_path + ' -i ' + video_path + ' ' + extract_frame_path + '/%4d.jpg')

    return extract_frame_path


def get_anchor_frames(frames_path, jump=5):
    # os.system('rm -rf ' + os.path.join(frames_path, 'tracking.json'))
    anchor_frames_path = os.path.join(frames_path, 'anchors')
    if not os.path.exists(anchor_frames_path):
        os.makedirs(anchor_frames_path)
    else:
        print("The {} exists! Skip getting anchors!".format(anchor_frames_path))
        return anchor_frames_path
    for each_frame in get_current_files_without_sub_files(frames_path):
        frame_name = os.path.basename(each_frame)
        if frame_name[:-4] != '.jpg':
            break
        if int(frame_name[:4]) % jump == 0 or frame_name[:4] == '0001':
            try:
                shutil.copyfile(os.path.join(frames_path, frame_name),
                                os.path.join(anchor_frames_path, frame_name))
            except:
                pass
    return anchor_frames_path


def get_anchor_dets(anchor_frames_path):
    os.system('bash ' + project_base_path + 'gpu_demo.sh ' + anchor_frames_path)
    return anchor_frames_path


def track_frames(frames_path, anchor_frames_path=None, video_id=None, retrack=False):
    if video_id is None:
        video_id = os.path.split(frames_path)[-1]

    tracking_json_path = os.path.join(frames_path, 'tracking.json')
    if not retrack:
        if os.path.exists(tracking_json_path):
            print('The tracking exists! skipping!', tracking_json_path)
            with open(tracking_json_path, 'r') as in_f:
                return json.load(in_f)['obj_tracking'], None

    if anchor_frames_path is None:
        anchor_frames_path = os.path.join(frames_path, 'anchors')
    anchor_names = list()
    anchor_bboxes = list()
    for each_anchor_file in sorted(get_current_files_without_sub_files(anchor_frames_path)):
        anchor_name = os.path.basename(each_anchor_file)
        if anchor_name.endswith('_det.json'):
            anchor_names.append(anchor_name[:4])
            with open(os.path.join(anchor_frames_path, anchor_name), 'r') as in_f:
                anchor_bbox_json = json.load(in_f)
                anchor_bboxes.append(anchor_bbox_json)

    frames_num = 0
    for each_frame_file in get_current_files_without_sub_files(frames_path):
        if each_frame_file.endswith('.jpg'):
            frames_num += 1

    anchor_names = sorted(anchor_names)
    obj_tracking_list = list()
    print('Now, tracking 4 the video: {}'.format(video_id))
    for i, each_anchor in enumerate(tqdm(anchor_names)):
        anchor_frames = list()
        if i + 1 == len(anchor_names):
            break
        next_anchor_name = anchor_names[i + 1]
        for frame_idx in range(int(each_anchor), int(next_anchor_name)):
            anchor_frames.append(cv2.imread(os.path.join(frames_path, str(frame_idx).zfill(4) + '.jpg')))

        # print(i, anchor_bboxes[i])
        for each_class, bboxes in anchor_bboxes[i].items():
            # print('Now is tracking: ', each_anchor, each_class)
            for each_bbox in bboxes:
                score = each_bbox['score']
                bbox = each_bbox['bbox']
                tracking_bboxes = tracker(anchor_frames, tuple(bbox))
                obj_tracking_list.append({
                    'obj_cls': each_class,
                    'start_frame': int(each_anchor),
                    'score': score,
                    'tracklet': tracking_bboxes
                })
    with open(tracking_json_path, 'w+') as out_f:
        out_f.write(json.dumps({
            'video_id': video_id,
            'obj_tracking': obj_tracking_list
        }))
    return obj_tracking_list, anchor_names


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

    init_frame = frames[0]
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


def write_video(video, fps, size, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, cv2.CAP_FFMPEG, fourcc, fps, size)
    for frame in video:
        out.write(frame)
    out.release()


def get_current_files_without_sub_files(path):
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]


def visualize_track(frames_path, obj_tracking_list=None, anchor_names=None):
    bbox_color = (255, 0, 0)
    size = [0, 0]
    if obj_tracking_list is None:
        with open(os.path.join(frames_path, 'tracking.json'), 'r') as obj_tracking_f:
            obj_tracking_list = json.load(obj_tracking_f)['obj_tracking']

    if anchor_names is None:
        anchor_names = set()
        for each_file in get_current_files_without_sub_files(os.path.join(frames_path, 'anchors')):
            anchor_names.add(each_file[:4])
    anchor_names = sorted(list(anchor_names))

    video = list()
    for i, anchor_id in enumerate(anchor_names):
        if i + 1 == len(anchor_names):
            break
        next_anchor_id = anchor_names[i + 1]

        segment_obj_tracks = list()
        for each_obj_track in obj_tracking_list:
            if each_obj_track['start_frame'] == int(anchor_id):
                segment_obj_tracks.append(each_obj_track)

        for j, fid in enumerate(range(int(anchor_id), int(next_anchor_id))):
            frame = cv2.imread(os.path.join(frames_path, str(fid).zfill(4) + '.jpg'))
            height, width, channels = frame.shape
            if width > size[0]:
                size[0] = width
            if height > size[1]:
                size[1] = height
            for each_obj_track in segment_obj_tracks:
                if j < len(each_obj_track['tracklet']):
                    obj_cls_label = each_obj_track['obj_cls']
                    score = each_obj_track['score']
                    bbox = each_obj_track['tracklet'][j]
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(frame, p1, p2, bbox_color, 2, 1)
                    cv2.putText(frame, obj_cls_label + ': ' + str(score),
                                p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.imshow('tracking', frame)
            video.append(frame)
            cv2.waitKey(100)

    write_video(video, 30, tuple(size), os.path.join(frames_path, os.path.split(frames_path)[-1] + '.mp4'))


if __name__ == '__main__':
    extract_frame_path = 'framesCache/6980260459'

    # test_vid_path = '/storage/dldi/PyProjects/vidor/img_test/6980260459.mp4'
    # extract_frame_path = extract_all_frames(test_vid_path)
    print('---' * 20)
    print('extract frames finish!', extract_frame_path)
    anchor_frames_path = get_anchor_frames(extract_frame_path)
    print('===' * 20)
    print('get_anchor frames finish!', anchor_frames_path)
    anchor_frames_det_path = get_anchor_dets(anchor_frames_path)
    print('--==' * 20)
    print('get_anchor_frames_det finish!', anchor_frames_det_path)
    obj_tracking_list, anchor_names = track_frames(extract_frame_path)
    print('===' * 20)
    print('track frames finish!')
    visualize_track(extract_frame_path)
