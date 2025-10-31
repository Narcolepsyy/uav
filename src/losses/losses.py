"""
Training losses for T-SiamTPN:
- Classification: Cross-Entropy over per-scale logits [B, 2, H_i, W_i], with positive assigned at target center cell.
- Regression: L1 over cxcywh (normalized to [0,1]) at the positive cell.
- IoU: GIoU computed between predicted and target boxes (converted to xyxy in pixel coords).

Assumptions:
- Model heads output:
    cls_i: [B, 2, H_i, W_i]
    reg_i: [B, 4, H_i, W_i] as normalized [cx, cy, w, h] in [0,1] w.r.t search crop (no anchors).
- Dataset provides target in pixels for the search crop size: target_box_cxcywh: [B, 4] (or [4] for single sample).
- We supervise only the positive cell per scale (nearest cell to target center). Negatives are addressed implicitly by CE over the whole map with class weighting.

Note:
- This is a lightweight loss suitable for embedded-focused training. Advanced label assignment (e.g., Gaussian heatmap, hard negatives mining) can be added later.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import box ops (GIoU and conversions)
try:
    from src.utils.box_ops import giou_loss as _giou_loss
    from src.utils.box_ops import box_cxcywh_to_xyxy as _cxcywh_to_xyxy
except Exception:
    # Fallback for non-package execution contexts
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from utils.box_ops import giou_loss as _giou_loss
    from utils.box_ops import box_cxcywh_to_xyxy as _cxcywh_to_xyxy


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_batch_tensor(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Ensure x has batch dimension.
    If x is [4], returns [1,4]; if already batched keep as is.
    """
    if x.dim() == 1 and x.numel() == 4:
        return x.view(1, 4)
    return x


def _map_center_to_cell(cx: float, cy: float, H: int, W: int, search_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Map center (pixels in search crop) to (iy, ix) cell coordinates on feature map [H,W].
    """
    sh, sw = search_size
    ix = int(round((cx / max(sw, 1e-6)) * (W - 1)))
    iy = int(round((cy / max(sh, 1e-6)) * (H - 1)))
    ix = max(0, min(W - 1, ix))
    iy = max(0, min(H - 1, iy))
    return iy, ix


def _gather_pos(pred_map: torch.Tensor, pos_indices: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Gather values at positive cells from a map [B, C, H, W].
    pos_indices: list of (iy, ix) per batch item.
    Returns [B, C].
    """
    B, C, H, W = pred_map.shape
    out = []
    for b in range(B):
        iy, ix = pos_indices[b]
        out.append(pred_map[b, :, iy, ix])
    return torch.stack(out, dim=0)


def _normalize_box_pixels_to01(box_cxcywh_px: torch.Tensor, search_size: Tuple[int, int]) -> torch.Tensor:
    """
    Normalize pixel-space cxcywh to [0,1] by dividing by (W,H).
    Input: [B,4], Output: [B,4]
    """
    sh, sw = float(search_size[0]), float(search_size[1])
    cx, cy, w, h = box_cxcywh_px.unbind(-1)
    cx_n = cx / max(sw, 1e-6)
    cy_n = cy / max(sh, 1e-6)
    w_n = w / max(sw, 1e-6)
    h_n = h / max(sh, 1e-6)
    return torch.stack([cx_n, cy_n, w_n, h_n], dim=-1).clamp(min=0.0, max=1.0)


def _denorm_box01_to_pixels(box_cxcywh01: torch.Tensor, search_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert normalized [0,1] cxcywh to pixel-space given search_size (H,W).
    Input: [B,4], Output: [B,4]
    """
    sh, sw = float(search_size[0]), float(search_size[1])
    cx, cy, w, h = box_cxcywh01.unbind(-1)
    cx_p = cx * sw
    cy_p = cy * sh
    w_p = w * sw
    h_p = h * sh
    return torch.stack([cx_p, cy_p, w_p, h_p], dim=-1)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

class ClassificationCELoss(nn.Module):
    """
    Cross-Entropy loss for classification head.
    Applies CE over [B,2,H,W] with class weighting to mitigate imbalance.
    """
    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 0.25):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

    def forward(self, logits: torch.Tensor, pos_indices: List[Tuple[int, int]]) -> torch.Tensor:
        """
        logits: [B, 2, H, W]
        pos_indices: list of (iy, ix) per batch
        Returns scalar loss.
        """
        B, C, H, W = logits.shape
        assert C == 2, f"Expected 2 classes (target/background), got {C}"
        # Create labels [B,H,W] with 1 at pos cell, 0 elsewhere
        labels = torch.zeros((B, H, W), dtype=torch.long, device=logits.device)
        for b in range(B):
            iy, ix = pos_indices[b]
            labels[b, iy, ix] = 1

        # Compute per-pixel CE
        # Flatten over spatial
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, 2)  # [B*H*W, 2]
        labels_flat = labels.reshape(-1)                          # [B*H*W]
        ce = F.cross_entropy(logits_flat, labels_flat, reduction="none")  # [B*H*W]

        # Apply class weighting
        weights = torch.full_like(labels_flat, fill_value=self.neg_weight, dtype=ce.dtype)
        weights[labels_flat == 1] = self.pos_weight

        loss = (ce * weights).mean()
        return loss


class L1BoxLoss(nn.Module):
    """
    L1 loss for cxcywh at positive cells.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_reg: torch.Tensor, target_box01: torch.Tensor, pos_indices: List[Tuple[int, int]]) -> torch.Tensor:
        """
        pred_reg: [B, 4, H, W] normalized [0,1]
        target_box01: [B, 4] normalized [0,1]
        pos_indices: list of (iy, ix) per batch
        """
        pred_pos = _gather_pos(pred_reg, pos_indices)  # [B,4]
        loss = F.l1_loss(pred_pos, target_box01, reduction="mean")
        return loss


class GIoUBoxLoss(nn.Module):
    """
    GIoU loss at positive cells, computed in pixel-space xyxy.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                pred_reg: torch.Tensor,
                target_box_px: torch.Tensor,
                pos_indices: List[Tuple[int, int]],
                search_size: Tuple[int, int]) -> torch.Tensor:
        """
        pred_reg: [B, 4, H, W] normalized [0,1]
        target_box_px: [B, 4] in pixels
        pos_indices: list of (iy, ix) per batch
        search_size: (H, W)
        """
        pred_pos01 = _gather_pos(pred_reg, pos_indices)  # [B,4] normalized
        pred_pos_px = _denorm_box01_to_pixels(pred_pos01, search_size)  # [B,4] in pixels

        # Convert both to xyxy
        pred_xyxy = _cxcywh_to_xyxy(pred_pos_px)  # [B,4]
        tgt_xyxy = _cxcywh_to_xyxy(target_box_px) # [B,4]

        # GIoU expects aligned sets -> use giou_loss over diagonal pairs
        loss = _giou_loss(pred_xyxy, tgt_xyxy, reduction=self.reduction)
        return loss


# -----------------------------------------------------------------------------
# Aggregator
# -----------------------------------------------------------------------------

class LossAggregator(nn.Module):
    """
    Aggregates classification CE, GIoU, and L1 losses across scales.

    Config keys:
      loss.classification: cross_entropy
      loss.iou: giou
      loss.l1: true/false
      loss.weights.cls / iou / l1
    """
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 pos_weight: float = 1.0,
                 neg_weight: float = 0.25):
        super().__init__()
        w = weights or {"cls": 1.0, "iou": 2.0, "l1": 1.0}
        self.w_cls = float(w.get("cls", 1.0))
        self.w_iou = float(w.get("iou", 2.0))
        self.w_l1 = float(w.get("l1", 1.0))

        self.cls_loss = ClassificationCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
        self.l1_loss = L1BoxLoss()
        self.giou_loss = GIoUBoxLoss(reduction="mean")

    def forward(self,
                outputs: Dict[str, List[torch.Tensor]],
                target_box_px: torch.Tensor,
                search_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        outputs: dict from model forward
          - "cls": list of [B,2,H,W]
          - "reg": list of [B,4,H,W] normalized [0,1]
        target_box_px: [B,4] (or [4]) in search pixels
        search_size: (H, W)
        Returns dict with per-component and total loss.
        """
        cls_list = outputs.get("cls", [])
        reg_list = outputs.get("reg", [])

        assert isinstance(cls_list, (list, tuple)) and isinstance(reg_list, (list, tuple)), \
            "Model outputs must include 'cls' and 'reg' lists."

        B = cls_list[0].shape[0]
        target_box_px = _to_batch_tensor(target_box_px)  # ensure [B,4]
        assert target_box_px.shape[0] == B, "Batch size mismatch between outputs and targets."

        # Compute pos indices per scale using target center mapped to each feature size
        pos_indices_per_scale: List[List[Tuple[int, int]]] = []
        for i, cls_map in enumerate(cls_list):
            _, _, H, W = cls_map.shape
            pos_indices = []
            for b in range(B):
                cx, cy, _, _ = target_box_px[b].tolist()
                iy, ix = _map_center_to_cell(cx, cy, H, W, search_size)
                pos_indices.append((iy, ix))
            pos_indices_per_scale.append(pos_indices)

        # Normalize target to [0,1] once
        target_box01 = _normalize_box_pixels_to01(target_box_px, search_size)

        # Aggregate losses across scales
        loss_cls_total = torch.tensor(0.0, device=cls_list[0].device)
        loss_l1_total = torch.tensor(0.0, device=cls_list[0].device)
        loss_iou_total = torch.tensor(0.0, device=cls_list[0].device)

        for s in range(len(cls_list)):
            pos_indices = pos_indices_per_scale[s]
            loss_cls_total = loss_cls_total + self.cls_loss(cls_list[s], pos_indices)
            loss_l1_total = loss_l1_total + self.l1_loss(reg_list[s], target_box01, pos_indices)
            loss_iou_total = loss_iou_total + self.giou_loss(reg_list[s], target_box_px, pos_indices, search_size)

        # Average across scales
        S = float(len(cls_list))
        loss_cls_total = loss_cls_total / S
        loss_l1_total = loss_l1_total / S
        loss_iou_total = loss_iou_total / S

        # Weighted sum
        total = self.w_cls * loss_cls_total + self.w_l1 * loss_l1_total + self.w_iou * loss_iou_total

        return {
            "loss_total": total,
            "loss_cls": loss_cls_total,
            "loss_l1": loss_l1_total,
            "loss_iou": loss_iou_total,
        }


# -----------------------------------------------------------------------------
# Config constructor
# -----------------------------------------------------------------------------

def from_config(cfg: Dict[str, object]) -> LossAggregator:
    """
    Build LossAggregator from nested/flat config dict.
    """
    def _get(keys: List[str], default):
        k = ".".join(keys)
        if k in cfg:
            return cfg[k]
        cur = cfg
        for key in keys:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur

    weights = {
        "cls": float(_get(["loss", "weights", "cls"], 1.0)),
        "iou": float(_get(["loss", "weights", "iou"], 2.0)),
        "l1": float(_get(["loss", "weights", "l1"], 1.0)),
    }
    # Class weighting for CE
    pos_w = 1.0
    neg_w = 0.25

    return LossAggregator(weights=weights, pos_weight=pos_w, neg_weight=neg_w)


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple smoke test
    torch.manual_seed(0)
    B = 2
    # Three scales with different spatial sizes
    cls = [
        torch.randn(B, 2, 64, 64),
        torch.randn(B, 2, 32, 32),
        torch.randn(B, 2, 16, 16),
    ]
    reg = [
        torch.sigmoid(torch.randn(B, 4, 64, 64)),  # normalized [0,1]
        torch.sigmoid(torch.randn(B, 4, 32, 32)),
        torch.sigmoid(torch.randn(B, 4, 16, 16)),
    ]
    outputs = {"cls": cls, "reg": reg}

    # Targets in pixels for a 256x256 search crop
    search_size = (256, 256)
    tgt = torch.tensor([[128.0, 128.0, 64.0, 64.0],
                        [80.0, 120.0, 40.0, 50.0]], dtype=torch.float32)

    agg = LossAggregator(weights={"cls": 1.0, "iou": 2.0, "l1": 1.0})
    losses = agg(outputs, tgt, search_size)
    print({k: float(v.item()) for k, v in losses.items()})