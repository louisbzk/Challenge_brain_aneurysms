import torch
import torch.nn.functional as F
import torch.nn as nn


class DiceFocalLogits(nn.Module):
    def __init__(self, dice_weight, alpha, gamma, smooth=1e-5):
        super(DiceFocalLogits, self).__init__()
        self.w = dice_weight
        self.alpha = alpha
        self.eps = smooth
        self.gamma = gamma

    def forward(self, logits, labels):
        _logits = logits.view(-1)
        _labels = labels.view(-1)
        probs = torch.sigmoid(_logits)

        bce = F.binary_cross_entropy_with_logits(_logits, _labels)
        p_t = (_labels * probs) + ((1. - _labels) * (1. - probs))
        alpha_factor = self.alpha * _labels + (1. - self.alpha) * (1. - _labels)
        gamma_factor = torch.pow(1. - p_t, self.gamma)

        focal_loss = (alpha_factor * gamma_factor * bce).mean()

        intersection = (probs * _labels).sum()
        dice_loss = 1. - (2. * intersection / (probs.sum() + _labels.sum() + self.eps))

        return self.w * dice_loss + (1. - self.w) * focal_loss


class IoUCohensKappa(nn.Module):
    def __init__(self, iou_weight, smooth=1e-5):
        if not (0. <= iou_weight <= 1.):
            raise ValueError(f'Invalid value for iou_weight ({iou_weight}). Must be between 0 and 1')
        super(IoUCohensKappa, self).__init__()
        self.w = iou_weight
        self.eps = smooth

    def forward(self, logits, labels):
        pred = torch.sigmoid(logits)
        inputs = pred.view(-1)
        targets = labels.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        iou_loss = 1. - iou

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1. - inputs)).sum()
        TN = ((1 - targets) * (1 - inputs)).sum()

        kappa = 2 * (TP * TN - FN * FP) / ((TP + FP)*(FP + TN) + (TP + FN)*(FN + TN))

        return self.w * iou_loss + (1. - self.w) * kappa


class IoUFocalLogits(nn.Module):
    def __init__(self, iou_weight, alpha, gamma, smooth=1e-5):
        super(IoUFocalLogits, self).__init__()
        self.w = iou_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = smooth

    def forward(self, logits, labels):
        _logits = logits.view(-1)
        _labels = labels.view(-1)
        probs = torch.sigmoid(_logits)

        bce = F.binary_cross_entropy_with_logits(_logits, _labels)
        p_t = (_labels * probs) + ((1. - _labels) * (1. - probs))
        alpha_factor = self.alpha * _labels + (1. - self.alpha) * (1. - _labels)
        gamma_factor = torch.pow(1. - p_t, self.gamma)

        focal_loss = (alpha_factor * gamma_factor * bce).mean()

        intersection = (probs * _labels).sum()
        total = probs.sum() + _labels.sum()
        union = total - intersection
        iou_loss = 1. - ((intersection + self.eps) / (union + self.eps))

        return self.w * iou_loss + (1. - self.w) * focal_loss
