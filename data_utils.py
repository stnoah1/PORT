import json
import os.path as osp
import copy
import pickle
import torch

import mmcv
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


JOINT_NAMES = [
    'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
    'Thorax', 'NeckBase', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]


def align_bone_xaxis(joints, idx=9):
    device = joints.device
    theta = torch.atan2(joints[:,idx,1], joints[:,idx,0])
    s = torch.sin(theta)
    c = torch.cos(theta)
    rot_mat = torch.stack([torch.stack([c, -s]),
                           torch.stack([s, c])]).permute(2,0,1).to(device)
    rotated = torch.matmul(joints, rot_mat)
    rot_inv = torch.stack([torch.stack([c, s]),
                           torch.stack([-s, c])]).permute(2,0,1).to(device)
    return rotated, rot_inv

class PanopticDataset(Dataset):
    def __init__(self, ann_path, pred_path=None, is_train=True):
        self.kpt2d = []
        self.kpt2d_pred = []
        self.img_names = []
        self.cam = []
        self.is_train = is_train
        with open(ann_path) as f:
            gt = json.load(f)
        for gt_2d, cam in zip(gt['annotations'], gt['images']):
            self.cam.append({'w':cam['width'], 'h':cam['height']})
            self.kpt2d.append(np.array(gt_2d['keypoints']).reshape(21,3)/ np.array([cam['width'], cam['height'], 1]))
            self.img_names.append(cam['file_name'])
        self.kpt2d = np.array(self.kpt2d)
        if not is_train:
            idx = 0
            with open(pred_path, 'rb') as f:
                pred = pickle.load(f)
            for batch in pred:
                for img_path, preds in zip(batch['image_paths'], batch['preds']):
                    self.kpt2d_pred.append(np.array(preds)/ np.array([[self.cam[idx]['w'], self.cam[idx]['h'], 1]]))
                    idx += 1
            self.kpt2d_pred = np.array(self.kpt2d_pred)
            thd_dist = np.sort(self.kpt2d_pred[:,:,2].flatten())
            dist_mag = len(thd_dist)
            self.percentile = [thd_dist[int(i/100*dist_mag)] for i in range(100)]
        else:
            self.kpt2d_pred = None

    def __len__(self):
        return len(self.kpt2d)

    def __getitem__(self, idx):
        meta_info = {
            'cam': self.cam[idx],
            'img_name': self.img_names[idx],
        }
        if self.is_train:
            return self.kpt2d[idx], None, meta_info
        else:
            return self.kpt2d[idx], self.kpt2d_pred[idx], meta_info
