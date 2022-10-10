import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functions.skeleton import A_binary

class CvT(nn.Module):
    def __init__(self, *, emb_dim=128, emb_kernel=3, proj_kernels=3, depth=12,\
                 heads=4, mlp_mult=2, dropout=0., graph_conv=True, dim=2):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
