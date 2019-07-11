import pickle
import torch

with open('test_det_cont/test_det_content_bbox_pred.pkl', 'rb') as in_f:
    det_content = pickle.load(in_f)
print(det_content)

with open('test_det_cont/test_det_content_cls_prob.pkl', 'rb') as in_f:
    det_content = pickle.load(in_f)
print(det_content)

with open('test_det_cont/test_det_content_RCNN_loss_bbox.pkl', 'rb') as in_f:
    det_content = pickle.load(in_f)
print(det_content)

with open('test_det_cont/test_det_content_RCNN_loss_cls.pkl', 'rb') as in_f:
    det_content = pickle.load(in_f)
print(det_content)

with open('test_det_cont/test_det_content_rois.pkl', 'rb') as in_f:
    det_content = pickle.load(in_f)
print(det_content)

with open('test_det_cont/test_det_content_rois_label.pkl', 'rb') as in_f:
    det_content = pickle.load(in_f)
print(det_content)


