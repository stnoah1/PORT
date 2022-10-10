import os
import random

import torch
import numpy as np
import torch.backends.cudnn as cudnn

from models.transformer.cvt import CvT

from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from functions.skeleton import bones, num_joints
import functions.skeleton


def init(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = False
    cudnn.benchmark = True

def get_dataset(args):
    if args.dataset == "panoptic":
        from data_utils import PanopticDataset
        train_ds = PanopticDataset(
                args.train_ann_path,
                None,
                is_train=True
            )
        valid_ds = PanopticDataset(
                args.test_ann_path,
                args.kpt2d_test_pred_path,
                is_train=False
            )
        k_set = [1, 5]
        num_joints = 21
    else:
        raise Exception('dataset error')

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, k_set, num_joints, valid_ds.percentile

def get_model(args, num_joints, device, return_kpt3d=False,):
    if args.model == "cvt":
        port = CvT(
            emb_dim=args.embd_dim,
            emb_kernel=3,
            proj_kernels=args.proj_kernels,
            depth=4,
            heads=4,
            mlp_mult=2,
            dropout=0.,
            graph_conv=args.graph_conv,
            dim=4 if args.use_bone_2d else 2,
        ).to(device)

    if args.self_supervision == 'simmim':
        self_sup = SimMIM(
            encoder=port,
            num_joints=num_joints,
            masking_ratio=args.mask_prob,
            return_kpt3d=return_kpt3d,
            model_name=args.model
        ).to(device)
    else:
        raise Exception('Not supported')
    return self_sup

def get_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        raise Exception('Not supported')
    return optimizer


def get_lr_scheduler(args, optimizer, num_train, warmup_epoch=10):
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_train * warmup_epoch,
        num_train * args.epochs
    )
    return lr_scheduler

def set_A_binary(dataset):
    A_binary = np.zeros((num_joints[dataset], num_joints[dataset]))
    for s, t in bones[dataset]:
        A_binary[s][t] = 1
        A_binary[t][s] = 1
    functions.skeleton.A_binary.append(A_binary)
