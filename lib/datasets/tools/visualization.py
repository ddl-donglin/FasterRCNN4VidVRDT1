import cv2
import os
from tqdm import tqdm

detect_res_root_path = '/Users/davidddl/nextGPUs/storage/dldi/PyProjects/FasterRCNN4VidVRDT1/data/Vidor_10k/results' \
                       '/VOC2019/Main '


def draw_bbox_on_imgs(img_path, class_name, score, bboxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scalar = 0.8
    font_thickness = 2
    img = cv2.imread(img_path)
    cv2.rectangle(img, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (0, 255, 0), 2)
    cv2.putText(img, score, (bboxes[0], bboxes[1]), font, font_scalar, (0, 255, 0), font_thickness,
                cv2.LINE_AA)
    video_id, img_id = img_path.split('/')[-2:]
    base_path = '/'
    for each_split_path in img_path.split('/')[:-2]:
        base_path = os.path.join(base_path, each_split_path)
    save_img_path = os.path.join(base_path, video_id + '_' + class_name + '_vis')
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
    cv2.imwrite(os.path.join(save_img_path, 'res_' + img_id), img)


def show_det_res(detect_res_file, score_threshold=0.6, test_total_num=50):
    with open(detect_res_file, 'r') as detect_res_f:
        result_list = detect_res_f.readlines()
    for each_res in tqdm(result_list):
        img_path, score, xmin, ymin, xmax, ymax = each_res.split(' ')
        if float(score) >= score_threshold and test_total_num >= 0:
            img_path = os.path.join(detect_res_root_path, 'test_vis', img_path + '.jpg')
            if os.path.exists(img_path):
                draw_bbox_on_imgs(img_path, detect_res_file.split('_')[-1][:-4], score, [int(float(i)) for i in [xmin, ymin, xmax, ymax]])
                test_total_num -= 1


if __name__ == '__main__':

    test_class = 'ball'
    test_file = 'comp4_det_test_full_{}.txt'.format(test_class)
    print('Visualization 4: ', test_file)
    show_det_res(os.path.join(detect_res_root_path, test_file))
