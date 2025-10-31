"""
Box operations and geometry utilities for tracking and detection.

This module provides:
- Format conversions between [cx, cy, w, h] and [x1, y1, x2, y2]
- Box clipping and resizing helpers
- IoU / GIoU computation (vectorized, batch-friendly)
- NMS with torchvision fallback
- Encode/decode helpers for bbox regression deltas

Design goals:
- Torch-first API (accepts float tensors, N x 4)
- Robust to empty inputs
- No hard dependency on torchvision (optional nms fallback)
"""

from __future__ import annotations
from typing import Tuple, Optional

import math
import torch
import torch.nn.functional as F

try:
    from torchvision.ops import nms as tv_nms  # type: ignore
    _HAS_TV_NMS = True
except Exception:
    _HAS_TV_NMS = False


# ----------------------
# Format conversions
# ----------------------
def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2].
    Args:
        boxes: Tensor [N, 4]
    Returns:
        Tensor [N, 4]
    """
    if boxes.numel() == 0:
        return boxes
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h].
    Args:
        boxes: Tensor [N, 4]
    Returns:
        Tensor [N, 4]
    """
    if boxes.numel() == 0:
        return boxes
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


# ----------------------
# Basic geometry helpers
# ----------------------
def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes [x1, y1, x2, y2].
    Args:
        boxes: Tensor [N, 4]
    Returns:
        Tensor [N]
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def clip_boxes_to_image(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Clip boxes to image bounds.
    Args:
        boxes: [N, 4] in xyxy
        size: (H, W)
    Returns:
        [N, 4] clipped
    """
    if boxes.numel() == 0:
        return boxes
    h, w = size
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1.clamp(min=0, max=w - 1)
    y1 = y1.clamp(min=0, max=h - 1)
    x2 = x2.clamp(min=0, max=w - 1)
    y2 = y2.clamp(min=0, max=h - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def resize_boxes(boxes: torch.Tensor, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> torch.Tensor:
    """
    Scale boxes from src_size (H, W) to dst_size (H, W).
    Args:
        boxes: [N, 4] xyxy
        src_size: (H_s, W_s)
        dst_size: (H_d, W_d)
    """
    if boxes.numel() == 0:
        return boxes
    hs, ws = src_size
    hd, wd = dst_size
    sx = wd / max(ws, 1e-6)
    sy = hd / max(hs, 1e-6)
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1 * sx
    x2 = x2 * sx
    y1 = y1 * sy
    y2 = y2 * sy
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ----------------------
# IoU / GIoU
# ----------------------
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU for two sets of boxes in xyxy.
    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]
    Returns:
        ious: [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU for two sets of boxes in xyxy.
    Reference: https://giou.stanford.edu/
    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]
    Returns:
        giou: [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    iou = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]  # area of smallest enclosing box

    area1 = box_area(boxes1)[:, None]
    area2 = box_area(boxes2)[None, :]

    lt_inter = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb_inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0)
    inter = wh_inter[:, :, 0] * wh_inter[:, :, 1]
    union = area1 + area2 - inter

    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


def giou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    GIoU loss between two aligned sets of boxes (N,4) each.
    """
    if pred_xyxy.numel() == 0:
        return pred_xyxy.new_tensor(0.0)
    giou = torch.diag(generalized_box_iou(pred_xyxy, tgt_xyxy))
    loss = 1.0 - giou
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ----------------------
# Encode / Decode for bbox regression
# ----------------------
def encode_boxes(anchors_cxcywh: torch.Tensor, targets_cxcywh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encode target boxes relative to anchors (SSD/Retina-like parameterization).
    Args:
        anchors_cxcywh: [N,4]
        targets_cxcywh: [N,4]
    Returns:
        deltas: [N,4] = [tx, ty, tw, th]
    """
    ax, ay, aw, ah = anchors_cxcywh.unbind(-1)
    gx, gy, gw, gh = targets_cxcywh.unbind(-1)

    tx = (gx - ax) / (aw.clamp(min=eps))
    ty = (gy - ay) / (ah.clamp(min=eps))
    tw = torch.log((gw.clamp(min=eps)) / aw.clamp(min=eps))
    th = torch.log((gh.clamp(min=eps)) / ah.clamp(min=eps))
    return torch.stack([tx, ty, tw, th], dim=-1)


def decode_boxes(anchors_cxcywh: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    Decode predicted deltas into boxes relative to anchors.
    Args:
        anchors_cxcywh: [N,4]
        deltas: [N,4] = [tx, ty, tw, th]
    Returns:
        boxes_cxcywh: [N,4]
    """
    ax, ay, aw, ah = anchors_cxcywh.unbind(-1)
    tx, ty, tw, th = deltas.unbind(-1)

    cx = tx * aw + ax
    cy = ty * ah + ay
    w = aw * torch.exp(tw)
    h = ah * torch.exp(th)
    return torch.stack([cx, cy, w, h], dim=-1)


# ----------------------
# Non-Max Suppression
# ----------------------
def nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    NMS with torchvision fallback.
    Args:
        boxes_xyxy: [N,4]
        scores: [N]
        iou_threshold: float in [0,1]
    Returns:
        keep indices: [K]
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0,), dtype=torch.long)

    if _HAS_TV_NMS:
        return tv_nms(boxes_xyxy, scores, iou_threshold)

    # PyTorch-only fallback
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break

        cur = boxes_xyxy[i].unsqueeze(0)  # [1,4]
        rest = boxes_xyxy[idxs[1:]]       # [M,4]
        ious = box_iou(cur, rest).squeeze(0)  # [M]
        mask = ious <= iou_threshold
        idxs = idxs[1:][mask]

    return boxes_xyxy.new_tensor(keep, dtype=torch.long)


# ----------------------
# Tracking-specific helpers
# ----------------------
def center_distance(cxcywh_a: torch.Tensor, cxcywh_b: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance between centers of two aligned sets [N,4] each.
    Returns [N]
    """
    if cxcywh_a.numel() == 0:
        return cxcywh_a.new_zeros((0,))
    ax, ay, _, _ = cxcywh_a.unbind(-1)
    bx, by, _, _ = cxcywh_b.unbind(-1)
    return torch.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def scale_ratio(cxcywh_prev: torch.Tensor, cxcywh_cur: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute isotropic scale ratio between previous and current boxes: sqrt((w2*h2)/(w1*h1)).
    Returns [N]
    """
    if cxcywh_prev.numel() == 0:
        return cxcywh_prev.new_zeros((0,))
    _, _, w1, h1 = cxcywh_prev.unbind(-1)
    _, _, w2, h2 = cxcywh_cur.unbind(-1)
    r = torch.sqrt(((w2 * h2).clamp(min=eps)) / (w1 * h1).clamp(min=eps))
    return r


def smooth_box(prev_smooth_cxcywh: torch.Tensor, cur_pred_cxcywh: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
    """
    Linear smoothing between previous smoothed state and current prediction.
    S_i = alpha * S_{i-1} + (1-alpha) * y_i
    """
    if prev_smooth_cxcywh.numel() == 0:
        return cur_pred_cxcywh
    return alpha * prev_smooth_cxcywh + (1.0 - alpha) * cur_pred_cxcywh


def correct_box(stable_cxcywh: torch.Tensor,
                cur_pred_cxcywh: torch.Tensor,
                center_dist_thresh: float,
                scale_ratio_thresh: float) -> torch.Tensor:
    """
    Box correction: if center distance or scale ratio exceeds thresholds,
    revert to stable box; otherwise keep current prediction.
    Thresholds are applied per-sample; expects Nx4 inputs.
    """
    if cur_pred_cxcywh.numel() == 0:
        return cur_pred_cxcywh

    cd = center_distance(stable_cxcywh, cur_pred_cxcywh)  # [N]
    sr = scale_ratio(stable_cxcywh, cur_pred_cxcywh)      # [N]
    keep_mask = (cd <= center_dist_thresh) & (torch.abs(sr - 1.0) <= scale_ratio_thresh)
    out = torch.where(keep_mask[:, None], cur_pred_cxcywh, stable_cxcywh)
    return out