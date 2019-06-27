import argparse
import os
from tqdm import tqdm

from get_tracklet_4_vid import extract_all_frames, get_anchor_frames, \
    get_anchor_dets, track_frames, visualize_track, get_current_files_without_sub_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1 Video object tracklets')

    parser.add_argument('--vidor', dest='vidor_support',
                        help='if need vidor_support, set this 1', type=int, default=0, required=False)
    single_video = parser.add_argument_group(title='Single Video')
    single_video.add_argument('-vp', dest='video_path',
                              help='path of video', type=str, required=False)
    single_video.add_argument('-aj', dest='anchor_jump',
                              help='frames num of anchor jump', default=5, type=int, required=False)
    single_video.add_argument('-vis', dest='visualize',
                              help='If u need visualzation, set this 1', default=0, type=int, required=False)

    vidor_support = parser.add_argument_group(title='Vidor Support')
    vidor_support.add_argument('-bp', dest='base_path', help='base path of vidor', required=False, type=str)
    vidor_support.add_argument('-sp', dest='split_dir_path', help='split_dir_path of videos', required=False, type=str)
    vidor_support.add_argument('-vd', dest='video_dir', help='dir path of videos', required=False, type=str)
    vidor_support.add_argument('-ajp', dest='anchor_jump_4_vidor',
                               help='frames num of anchor jump', default=5, type=int, required=False)
    args = parser.parse_args()

    if args.vidor_support == 0:
        print('Video object tracking 4 single Video!')
        print('Video path 2 get obj tracking: ', args.video_path)
        extract_frame_path = extract_all_frames(args.video_path)
        print('---' * 20)
        print('extract frames finish!', extract_frame_path)
        anchor_frames_path = get_anchor_frames(extract_frame_path, args.anchor_jump)
        print('===' * 20)
        print('get_anchor frames finish!', anchor_frames_path)
        anchor_frames_det_path = get_anchor_dets(anchor_frames_path)
        print('--=' * 20)
        print('get_anchor_frames_det finish!', anchor_frames_det_path)
        obj_tracking_list, anchor_names = track_frames(extract_frame_path)
        print('-==' * 20)
        print('track frames finish!')
        if args.visualize != 0:
            visualize_track(extract_frame_path)

    else:
        print('Video Object Tracking 4 Vidor!')
        vid_dir_path = os.path.join(args.base_path, args.split_dir_path, args.video_dir)
        for vid in tqdm(get_current_files_without_sub_files(vid_dir_path)):
            extract_frame_path = extract_all_frames(os.path.join(vid_dir_path, vid))
            print('---' * 20)
            print('extract frames finish!', extract_frame_path)
            anchor_frames_path = get_anchor_frames(extract_frame_path, args.anchor_jump_4_vidor)
            print('===' * 20)
            print('get_anchor frames finish!', anchor_frames_path)
            anchor_frames_det_path = get_anchor_dets(anchor_frames_path)
            print('--=' * 20)
            print('get_anchor_frames_det finish!', anchor_frames_det_path)
            obj_tracking_list, anchor_names = track_frames(extract_frame_path)
            print('-==' * 20)
            print('track frames finish!')
