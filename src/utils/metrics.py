"""
Metrics utilities for UAV tracking and lightweight training evaluation.

Includes:
- AverageMeter for streaming averages (embedded-friendly)
- Success AUC (IoU-based) and Precision (center-distance) curves for tracking
- Top-1 classification accuracy for heads sanity check
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import math
import torch

from .box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    center_distance,
)


# ----------------------
# Streaming meters
# ----------------------
class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


# ----------------------
# Tracking metrics
# ----------------------
def success_curve(ious: torch.Tensor, num_bins: int = 101) -> torch.Tensor:
    """
    Compute success curve over IoU thresholds in [0, 1].
    Args:
        ious: Tensor [T] IoU per-frame (NaN-safe: NaN treated as 0)
        num_bins: number of thresholds
    Returns:
        curve: Tensor [num_bins] success rate per threshold
    """
    if ious.numel() == 0:
        return torch.zeros(num_bins, dtype=torch.float32)
    ious = torch.nan_to_num(ious, nan=0.0).clamp(0.0, 1.0)
    thresholds = torch.linspace(0, 1, steps=num_bins, dtype=torch.float32, device=ious.device)
    curve = torch.stack([(ious >= th).float().mean() for th in thresholds], dim=0)
    return curve.cpu()


def success_auc(ious: torch.Tensor, num_bins: int = 101) -> float:
    """
    Area Under Curve (AUC) of success curve (IoU-based).
    """
    curve = success_curve(ious, num_bins=num_bins)
    # Integral over [0,1] of success curve approximated by mean across bins
    return float(curve.mean().item())


def precision_curve(dists: torch.Tensor,
                    max_threshold: float = 50.0,
                    step: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute precision curve over center-distance thresholds [0, max_threshold].
    Args:
        dists: Tensor [T] center distances per frame (pixels)
        max_threshold: maximum threshold in pixels
        step: increment in pixels
    Returns:
        thresholds: Tensor [K]
        curve: Tensor [K] precision (= fraction within threshold)
    """
    if dists.numel() == 0:
        K = int(max_threshold / step) + 1
        thresholds = torch.linspace(0, max_threshold, steps=K)
        return thresholds, torch.zeros_like(thresholds)

    dists = torch.nan_to_num(dists, nan=float("inf")).clamp(min=0.0)
    K = int(max_threshold / step) + 1
    thresholds = torch.linspace(0, max_threshold, steps=K, dtype=torch.float32, device=dists.device)
    curve = torch.stack([(dists <= th).float().mean() for th in thresholds], dim=0)
    return thresholds.cpu(), curve.cpu()


def precision_at(dists: torch.Tensor, threshold_px: float = 20.0) -> float:
    """
    Precision at fixed pixel threshold (default 20px).
    """
    if dists.numel() == 0:
        return 0.0
    dists = torch.nan_to_num(dists, nan=float("inf"))
    return float((dists <= threshold_px).float().mean().item())


def tracking_metrics(pred_cxcywh: torch.Tensor,
                     gt_cxcywh: torch.Tensor,
                     img_size: Optional[Tuple[int, int]] = None,
                     precision_threshold_px: float = 20.0,
                     success_bins: int = 101) -> Dict[str, float]:
    """
    Compute standard tracking metrics for a single sequence.

    Args:
        pred_cxcywh: [T,4] predicted boxes
        gt_cxcywh:   [T,4] ground-truth boxes
        img_size: (H, W) optional; used only for clipping if needed
        precision_threshold_px: pixel threshold for "precision"
        success_bins: number of thresholds for success AUC

    Returns:
        dict with keys: avg_iou, success_auc, precision_at_X, frames
    """
    T = min(pred_cxcywh.shape[0], gt_cxcywh.shape[0])
    if T == 0:
        return {"avg_iou": 0.0, "success_auc": 0.0, f"precision@{int(precision_threshold_px)}px": 0.0, "frames": 0}

    pred_xyxy = box_cxcywh_to_xyxy(pred_cxcywh[:T])
    gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh[:T])

    # Pairwise IoU along diagonal (aligned frames)
    ious = torch.diag(box_iou(pred_xyxy, gt_xyxy))
    avg_iou = float(torch.nan_to_num(ious, nan=0.0).mean().item())

    # Center-distance precision
    dists = center_distance(pred_cxcywh[:T], gt_cxcywh[:T])
    prec_at = precision_at(dists, threshold_px=precision_threshold_px)

    # Success AUC
    auc = success_auc(ious, num_bins=success_bins)

    return {
        "avg_iou": avg_iou,
        "success_auc": auc,
        f"precision@{int(precision_threshold_px)}px": prec_at,
        "frames": float(T),
    }


# ----------------------
# Classification sanity metrics
# ----------------------
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy given logits and integer class targets.
    """
    if logits.numel() == 0 or targets.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().mean()
    return float(correct.item())


__all__ = [
    "AverageMeter",
    "success_curve",
    "success_auc",
    "precision_curve",
    "precision_at",
    "tracking_metrics",
    "top1_accuracy",
]