import cv2
import os


detect_res_root_path = '/home/daivd/PycharmProjects/FasterRCNN4VidVRDT1/data/output/Main'


def draw_bbox_on_imgs(img_path, score, bboxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scalar = 0.8
    font_thickness = 2
    img = cv2.imread(img_path)
    cv2.rectangle(img, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (0, 255, 0), 2)
    cv2.putText(img, score, (bboxes[0], bboxes[1]), font, font_scalar, (0, 0, 0), font_thickness,
                cv2.LINE_AA)
    video_id, img_id = img_path.split('/')[-2:]
    cv2.imwrite(os.path.join(img_path.split('/')[:-2], video_id + '_vis', 'res_' + img_id), img)


def show_det_res(detect_res_file):
    with open(detect_res_file, 'r') as detect_res_f:
        result_list = detect_res_f.readlines()
    for each_res in result_list:
        img_path, score, xmin, ymin, xmax, ymax = each_res.split(' ')
        img_path = os.path.join(detect_res_root_path, 'test_vis', img_path + '.jpg')
        draw_bbox_on_imgs(img_path, score, [xmin, ymin, xmax, ymax])


if __name__ == '__main__':
    test_class = 'baby'
    test_file = 'comp4_det_test_full_{}.txt'.format(test_class)
    print('Visualization 4: ', test_file)
    show_det_res(os.path.join(detect_res_root_path, test_file))
