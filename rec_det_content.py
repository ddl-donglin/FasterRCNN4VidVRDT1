import pickle
import torch

# torch.load('test_det_content.pkl', map_location='cpu')
#
with open('test_det_content.pkl', 'rb') as in_f:
    det_content = pickle.load(in_f)

print(det_content.keys())
