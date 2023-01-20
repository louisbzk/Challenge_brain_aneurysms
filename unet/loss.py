# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 27/08/2019 15:29
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, logits, true):
        inputs = logits.view(-1)
        targets = true.view(-1)
        intersection = (inputs * targets).sum()
        dice_score = 2. * (intersection + self.epsilon) / (inputs.sum() + targets.sum() + self.epsilon)
        # dice_score = 1 - dice_score.sum() / batch_size
        return 1. - dice_score


class DiceBCEFocalLoss(nn.Module):
    def __init__(self, dice_bce_weight=0.5, smooth=1e-8, focal_gamma=2.):
        super(DiceBCEFocalLoss, self).__init__()
        if not (0. <= dice_bce_weight <= 1.):
            raise ValueError(f'Invalid value for dice_bce_weight ({dice_bce_weight}). Must be between 0 and 1')

        self.smooth = smooth
        self.dice_bce_weight = dice_bce_weight
        self.focal_gamma = focal_gamma

    def forward(self, pred, true):
        inputs = pred.view(-1)  # flatten
        targets = true.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = bce + dice_loss

        focal_loss = (1 - torch.exp(-bce))**self.focal_gamma * bce

        return self.dice_bce_weight * dice_bce + (1. - self.dice_bce_weight) * focal_loss


class IoUTverskyLoss(nn.Module):
    def __init__(self, iou_weight, tversky_alpha, tversky_beta, smooth=1e-8):
        if not (0. <= iou_weight <= 1.):
            raise ValueError(f'Invalid value for iou_weight ({iou_weight}). Must be between 0 and 1')
        super(IoUTverskyLoss, self).__init__()
        self.iou_weight = iou_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.smooth = smooth

    def forward(self, pred, true):
        inputs = pred.view(-1)
        targets = true.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1. - iou

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1. - inputs)).sum()

        tversky = (TP + self.smooth) / (TP + self.tversky_alpha * FP + self.tversky_beta * FN + self.smooth)
        tversky_loss = 1. - tversky

        return self.iou_weight * iou_loss + (1. - self.iou_weight) * tversky_loss


class IoUCohensKappa(nn.Module):
    def __init__(self, iou_weight, smooth=1e-8):
        if not (0. <= iou_weight <= 1.):
            raise ValueError(f'Invalid value for iou_weight ({iou_weight}). Must be between 0 and 1')
        super(IoUCohensKappa, self).__init__()
        self.iou_weight = iou_weight
        self.smooth = smooth

    def forward(self, pred, true):
        inputs = pred.view(-1)
        targets = true.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1. - iou

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1. - inputs)).sum()
        TN = ((1 - targets) * (1 - inputs)).sum()

        kappa = 2 * (TP * TN - FN * FP) / ((TP + FP)*(FP + TN) + (TP + FN)*(FN + TN))

        return self.iou_weight * iou_loss + (1. - self.iou_weight) * kappa


class AsymmetricLoss(nn.Module):
    def __init__(self, beta):
        super(AsymmetricLoss, self).__init__()
        # hyper-parameter for balancing precision and recall
        self.beta = beta

    def forward(self, targets, logits):
        pass


if __name__ == '__main__':
    import numpy as np
    yt = np.random.random(size=(2, 1, 3, 3, 3))
    # print(yt)
    yt = torch.from_numpy(yt)
    yp = np.zeros(shape=(2, 1, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp)
    # print(yp)
    dl = DiceLoss()
    print(dl(yp, yt).item())
