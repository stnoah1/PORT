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

def parse_h36m_imgname(imgname):
    """Parse imgname to get information of subject, action and camera.
    A typical h36m image filename is like:
    S1_Directions_1.54138969_000001.jpg
    """
    subj, rest = osp.basename(imgname).split('_', 1)
    action, rest = rest.split('.', 1)
    camera, rest = rest.split('_', 1)

    return subj, action, camera

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

class H36MDataset(Dataset):
    def __init__(self, ann_path, cam_path, pred_path, cache_path=None, is_train=True):
        if osp.exists(cache_path) and is_train:
            with open(cache_path, 'rb') as f:
                data_dict = pickle.load(f)
            self.img_names = data_dict["img_names"]
            self.subjs = data_dict["subjs"]
            self.actions = data_dict["actions"]
            self.cams = data_dict["cams"]
            self.kpt2d = data_dict["kpt2d"]
            self.kpt2d_pred = data_dict["kpt2d_pred"]
        else:
            anno = np.load(ann_path)
            cam_param = mmcv.load(cam_path)
            kpts2d = np.array(anno['part'])[..., :2]
            self.img_names = []
            self.subjs = []
            self.actions = []
            self.cams = []
            self.kpt2d = []
            self.kpt2d_pred = []

            # load pred
            if pred_path.split('.')[-1] == "pkl":
                with open(pred_path, 'rb') as f:
                    pred = pickle.load(f)
            elif pred_path.split('.')[-1] == "json":
                with open(pred_path) as f:
                    pred = json.load(f)

            pred_dict = {}
            for batch in tqdm(pred, desc='Loading Data'):
                for i, img_path in enumerate(batch['image_paths']):
                    pred_dict[osp.basename(img_path)] = np.array(batch['preds'][i])

            img_names = anno['imgname']
            for i, img_name in tqdm(enumerate(img_names), desc='Processing Data', total=len(img_names)):
                subj, action, cam = parse_h36m_imgname(img_name)
                if subj == 'S11' and action == 'Directions':
                    continue
                cam = cam_param[(subj, cam)]
                self.img_names.append(img_name)
                self.subjs.append(subj)
                self.actions.append(action)
                self.cams.append(cam)
                self.kpt2d.append(kpts2d[i] / np.array([cam['w'], cam['h']]))
                self.kpt2d_pred.append(pred_dict[osp.basename(img_name)] / np.array([cam['w'], cam['h'], 1]))
            thd_dist = np.sort(np.array(self.kpt2d_pred)[:, :, 2].flatten())
            dist_mag = len(thd_dist)
            self.percentile = [thd_dist[int(i/100*dist_mag)] for i in range(100)]

    def __getitem__(self, i):
        meta_info = {
            'img_name': self.img_names[i],
            'subject' : self.subjs[i],
            'action' : self.actions[i],
            'cam' : self.cams[i],
        }
        return self.kpt2d[i], self.kpt2d_pred[i], meta_info

    def __len__(self):
        return len(self.kpt2d)


class H36MDatasetMasked(Dataset):
    def __init__(self, ann_path, cam_path, pred_path, cache_path=None, is_train=True):
        if osp.exists(cache_path) and is_train:
            with open(cache_path, 'rb') as f:
                data_dict = pickle.load(f)
            self.img_names = data_dict["img_names"]
            self.subjs = data_dict["subjs"]
            self.actions = data_dict["actions"]
            self.cams = data_dict["cams"]
            self.kpt2d = data_dict["kpt2d"]
            self.kpt2d_pred = data_dict["kpt2d_pred"]
            self.vis = data_dict["kpt2d"]
        else:
            anno = np.load(ann_path)
            cam_param = mmcv.load(cam_path)
            kpts2d = np.array(anno['part'])[..., :2]
            self.img_names = []
            self.subjs = []
            self.actions = []
            self.cams = []
            self.kpt2d = []
            self.kpt2d_pred = []
            self.vis = []

            # load pred
            if pred_path.split('.')[-1] == "pkl":
                with open(pred_path, 'rb') as f:
                    pred = pickle.load(f)
            elif pred_path.split('.')[-1] == "json":
                with open(pred_path) as f:
                    pred = json.load(f)
            # if ann_path.split('.')[-2][-4:] == "test":
                # with open("data/h36m/img_masked/visibility.pkl", 'rb') as f:
                    # self.vis = list(pickle.load(f).values())

            pred_dict = {}
            for batch in tqdm(pred, desc='Loading Data'):
                for i, (img_path, bbox_id) in enumerate(zip(batch['image_paths'], batch['bbox_ids'])):
                    subj, action, cam = parse_h36m_imgname(img_path)
                    cam = cam_param[(subj, cam)]
                    self.img_names.append(img_path)
                    self.subjs.append(subj)
                    self.actions.append(action)
                    self.cams.append(cam)
                    self.kpt2d.append(kpts2d[bbox_id]/ np.array([cam['w'], cam['h']]))
                    self.kpt2d_pred.append(np.array(batch['preds'][i]) / np.array([cam['w'], cam['h'], 1]))
            thd_dist = np.sort(np.array(self.kpt2d_pred)[:, :, 2].flatten())
            dist_mag = len(thd_dist)
            self.percentile = [thd_dist[int(i/100*dist_mag)] for i in range(100)]

    def __getitem__(self, i):
        meta_info = {
            'img_name': self.img_names[i],
            'subject' : self.subjs[i],
            'action' : self.actions[i],
            'cam' : self.cams[i],
            # 'vis' : self.vis[i],
        }
        return self.kpt2d[i], self.kpt2d_pred[i], meta_info

    def __len__(self):
        return len(self.kpt2d)


class FPHBDataset(Dataset):

    def __init__(self, ann_path, pred_path, img_size=(480, 270), is_train=True):
        self.kpt2d = []
        self.kpt3d = []
        self.kpt2d_pred = []
        self.img_names = []

        self.cam = {
            "extr": np.array([
                [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                [0, 0, 0, 1]]),
            "intr": np.array(
                [[1395.749023, 0, 935.732544],
                [0, 1395.749268, 540.681030],
                [0, 0, 1]]),
            "w": img_size[0],
            "h": img_size[1]
        }

        with open(ann_path) as fp:
            fphb_data = json.load(fp)
            gt_annotations = fphb_data['annotations']
            img_names = fphb_data['images']

        with open(pred_path) as f:
            pred = json.load(f)

        pred_dict = {}
        for batch in pred:
            for i, img_path in enumerate(batch['image_paths']):
                pred_dict[osp.relpath(img_path)] = np.array(batch['preds'][i])

        for gt, img_name in zip(gt_annotations, img_names):
            # if self.n_out_of_frame(gt) >= 20:
                # continue
            scaled_2d = np.array(gt['keypoints']).reshape(-1,3)[:,:2]
            self.kpt2d.append(scaled_2d / np.array([self.cam['w'], self.cam['h']]))
            self.kpt2d_pred.append(pred_dict[img_name['file_name']]/ np.array([self.cam['w'], self.cam['h'], 1]))
            self.img_names.append(img_name['file_name'])
            # if img_name['file_name'].split('/')[-4] == "receive_coin" and img_name['file_name'].split('/')[-3] == "1" and img_name['file_name'].split('/')[-1] == "color_0039.jpeg":
                # import ipdb; ipdb.set_trace()

        thd_dist = np.sort(np.array(self.kpt2d_pred)[:, :, 2].flatten())
        dist_mag = len(thd_dist)
        self.percentile = [thd_dist[int(i/100*dist_mag)] for i in range(100)]

    def n_out_of_frame(self, gt):
        x = np.array(gt['keypoints']).reshape(-1,3)[:,0]
        y = np.array(gt['keypoints']).reshape(-1,3)[:,1]
        x_out = (x > self.cam['w']) | (x < 0)
        y_out = (y > self.cam['h']) | (y < 0)
        n = (x_out | y_out).sum()
        return n

    def world2cam(self, skel):
        skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        skel_camcoords = self.cam["extr"].dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
        return skel_camcoords

    def __getitem__(self, i):
        meta_info = {
            'img_name': self.img_names[i],
            'cam' : self.cam,
        }
        return self.kpt2d[i], self.kpt2d_pred[i], meta_info

    def __len__(self):
        return len(self.kpt2d)


class RHDDataset(Dataset):
    def __init__(self, ann_path, pred_path, is_train=True):
        self.kpt2d = []
        self.kpt2d_pred = []
        self.img_names = []
        self.is_train = is_train
        self.cam = {
            "w": 320,
            "h": 320
        }
        if not is_train:
            with open(pred_path, 'rb') as f:
                pred = pickle.load(f)
            for batch in pred:
                for i, (img_path, preds) in enumerate(zip(batch['image_paths'], batch['preds'])):
                    self.img_names.append(osp.basename(img_path))
                    self.kpt2d_pred.append(np.array(preds)/ np.array([self.cam['w'], self.cam['h'], 1]))
            self.kpt2d_pred = np.array(self.kpt2d_pred)
            thd_dist = np.sort(self.kpt2d_pred[:,:,2].flatten())
            dist_mag = len(thd_dist)
            self.percentile = [thd_dist[int(i/100*dist_mag)] for i in range(100)]
        with open(ann_path) as f:
            gt = json.load(f)
        for gt_2d in gt['annotations']:
            self.kpt2d.append(gt_2d['keypoints']/ np.array([self.cam['w'], self.cam['h'], 1]))
        self.kpt2d = np.array(self.kpt2d)

    def __len__(self):
        return len(self.kpt2d)

    def __getitem__(self, idx):
        meta_info = {
            'cam': self.cam,
            # 'img_name': self.img_names[idx],
        }
        if self.is_train:
            return self.kpt2d[idx], self.kpt2d[idx], meta_info
        else:
            return self.kpt2d[idx], self.kpt2d_pred[idx], meta_info


class PanopticDataset(Dataset):
    def __init__(self, ann_path, pred_path, is_train=True):
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

    def __len__(self):
        return len(self.kpt2d)

    def __getitem__(self, idx):
        if self.is_train:
            meta_info = {
                'cam': self.cam[idx],
            }
            return self.kpt2d[idx], self.kpt2d[idx], meta_info
        else:
            meta_info = {
                'cam': self.cam[idx],
                'img_name': self.img_names[idx],
            }
            return self.kpt2d[idx], self.kpt2d_pred[idx], meta_info


# class COCODataset(Dataset):

    # def __init__(self, ann_path, pred_path, img_size=(480, 270)):
        # self.kpt2d = []
        # self.kpt3d = []
        # self.kpt2d_pred = []
        # self.img_names = []

        # self.cam = {
            # "extr": np.array([
                # [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                # [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                # [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                # [0, 0, 0, 1]]),
            # "intr": np.array(
                # [[1395.749023, 0, 935.732544],
                # [0, 1395.749268, 540.681030],
                # [0, 0, 1]]),
            # "w": img_size[0],
            # "h": img_size[1]
        # }

        # with open(ann_path) as fp:
            # fphb_data = json.load(fp)
            # gt_annotations = fphb_data['annotations']
            # img_names = fphb_data['images']
        # with open(pred_path) as f:
            # pred = json.load(f)

        # pred_dict = {}
        # for batch in pred:
            # for i, img_path in enumerate(batch['image_paths']):
                # pred_dict[osp.relpath(img_path)] = np.array(batch['preds'][i])

        # for gt, img_name in zip(gt_annotations, img_names):
            # scaled_2d = np.array(gt['keypoints']).reshape(-1,3)[:,:2]
            # fphb_3d_world = np.array(gt['keypoints_3d']).reshape(-1,3)
            # fphb_3d_camera = self.world2cam(fphb_3d_world)
            # self.kpt2d.append(scaled_2d / np.array([self.cam['w'], self.cam['h']]))
            # self.kpt3d.append(fphb_3d_camera / 1000)
            # self.kpt2d_pred.append(pred_dict[img_name['file_name']]/ np.array([self.cam['w'], self.cam['h'], 1]))
            # self.img_names.append(img_name['file_name'])

    # def world2cam(self, skel):
        # skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        # skel_camcoords = self.cam["extr"].dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
        # return skel_camcoords

    # def __getitem__(self, i):
        # meta_info = {
            # 'img_name': self.img_names[i],
            # 'cam' : self.cam
        # }
        # return self.kpt3d[i], self.kpt2d[i], self.kpt2d_pred[i], meta_info

    # def __len__(self):
        # return len(self.kpt2d)


# class MPIIDataset(Dataset):
    # def __init__(self, ann_path,
                 # pred_path,
                 # kpt2d_stat_path=None,
                 # kpt3d_stat_path=None):

        # with open(ann_path) as fp:
            # gt_annotations = json.load(fp)

        # with open(pred_path, 'rb') as f:
            # pred = pickle.load(f)

        # self.kpt2d = []
        # self.kpt2d_pred = []
        # self.img_names = []
        # # pred_dict = {}

        # counter = 0
        # for batch in pred:
            # for i, img_path in enumerate(batch['image_paths']):
                # self.kpt2d_pred.append(np.array(batch['preds'][i])/ np.array([self.cam['w'], self.cam['h'], 1]))
                # counter += 1

        # counter = 0
        # for gt in gt_annotations:
            # scaled_2d = np.array(gt['joints']) / np.array([self.cam['w'], self.cam['h']])
            # img_name = gt['image']
            # scaled_2d = np.concatenate([scaled_2d, np.expand_dims(np.array(gt['joints_vis']),axis=1)], axis=1)
            # self.kpt2d.append(scaled_2d)
            # self.img_names.append(img_name)
            # counter += 1
        # print("gt data counter :", counter)

    # def __getitem__(self, i):
        # meta_info = {
            # 'img_name': self.img_names[i],
            # 'cam' : self.cam
        # }
        # return self.kpt2d[i], self.kpt2d_pred[i], meta_info

    # def __len__(self):
        # return len(self.kpt2d)

