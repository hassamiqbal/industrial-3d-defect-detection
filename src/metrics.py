"""Metrics and loss functions for binary 2D/3D defect segmentation."""

import torch
import torch.nn.functional as F


def soft_dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.reshape(probabilities.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    intersection = (probabilities * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probabilities.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


def segmentation_loss(logits, targets, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(logits, targets)
    total = bce_weight * bce + (1.0 - bce_weight) * dice
    return total, bce, dice


def binary_dice(logits, targets, threshold=0.5, eps=1e-7):
    prediction = (torch.sigmoid(logits) >= threshold).float()
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * targets).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + targets.sum(dim=dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def binary_iou(logits, targets, threshold=0.5, eps=1e-7):
    prediction = (torch.sigmoid(logits) >= threshold).float()
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * targets).sum(dim=dims)
    union = prediction.sum(dim=dims) + targets.sum(dim=dims) - intersection
    return ((intersection + eps) / (union + eps)).mean()
