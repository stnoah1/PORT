from __future__ import absolute_import

import numpy as np
import torch

bones = {
    "h36m":[
        [0, 1], [1, 2], [2, 3],
        [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9], [9, 10],
        [8, 11], [11, 12], [12, 13],
        [8, 14], [14, 15], [15, 16]
    ],
    "h36m_masked":[
        [0, 1], [1, 2], [2, 3],
        [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9], [9, 10],
        [8, 11], [11, 12], [12, 13],
        [8, 14], [14, 15], [15, 16]
    ],
    "mpii": [
        [0, 1], [1, 2], [2, 3], [3, 6], [2, 6],
        [3, 4], [4, 5],
        [6, 7], [7, 8], [8, 9],
        [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
    ],
    "fphb": [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ],
    "rhd": [
        [0, 4], [1, 2], [2, 3], [3, 4],
        [0, 8], [5, 6], [6, 7], [7, 8],
        [0, 12], [9, 10], [10, 11], [11, 12],
        [0, 16], [13, 14], [14, 15], [15, 16],
        [0, 20], [17, 18], [18, 19], [19, 20]
    ],
    "panoptic": [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
}

num_joints = {"h36m": 17, "h36m_masked": 17, "fphb": 21, "mpii": 16, "rhd": 21, "panoptic": 21}

global A_binary
A_binary = []
