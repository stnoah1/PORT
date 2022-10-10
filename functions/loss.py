# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import numpy as np
import torch
import torch.nn as nn


def epe(predicted, target, vis=1):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm((predicted - target)*vis, dim=len(target.shape) - 1))

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Reimplementation of MATLAB's `procrustes` function to Numpy,
    refer to https://codeday.me/bug/20180920/256259.html

    Args
        X: target pose
        Y: input pose
        scaling: if False, the scaling component of the transformation is forced to 1
        reflection: if 'best' (default), the transformation solution may or may not
                    include a reflection component, depending on which fits the data
                    best. setting reflection to True or False forces a solution with
                    reflection or no reflection respectively.
    Return
        d: the residual sum of squared errors, normalized according to a
           measure of the scale of X, ((X - X.mean(0))**2).sum()
        Z: the matrix of transformed Y-values
        tform: a dict specifying the rotation, translation and scaling that maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def p_epe(predicted, target, vis=1):
    # return torch.from_numpy(np.array(0.05))  # kh0826
    # assert False, 'skip this mean to save time'
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    target = np.array(target.cpu())
    predicted = np.array(predicted.cpu())
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm((predicted_aligned - target)*np.array(vis), axis=len(target.shape) - 1))


class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.
    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        preds = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        gts = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            pred = preds[idx].squeeze(1)
            gt = gts[idx].squeeze(1)
            if target_weight is not None:
                loss += self.criterion(pred * target_weight[:, idx].unsqueeze(-1),
                                       gt * target_weight[:, idx].unsqueeze(-1))
            else:
                loss += self.criterion(pred, gt)

        return loss / num_joints * self.loss_weight
