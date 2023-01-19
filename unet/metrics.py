# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 27/08/2019 15:29
import torch


def mean_iou(pred: torch.FloatTensor, true: torch.FloatTensor, smooth=1e-8):
    intersection = (pred * true).sum()
    total = (pred + true).sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou
