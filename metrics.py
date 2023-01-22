import torch


def mean_iou_logits(logits, targets, smooth=1e-5):
    _logits = logits.view(-1)
    _targets = targets.view(-1)
    _preds = (torch.sigmoid(_logits) > 0.5).float()

    intersection = (_preds * _targets).sum()
    total = (_preds + _targets).sum()
    union = total - intersection

    return ((intersection + smooth) / (union + smooth)).item()
