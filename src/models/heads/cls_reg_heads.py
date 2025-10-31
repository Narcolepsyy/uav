"""
Lightweight SiamTPN heads for classification and regression.

Design goals:
- Embedded-friendly heads using depthwise-separable convs
- Shared weights across pyramid levels to reduce params/FLOPs
- Configurable tower channels (default two layers at C'=192)
- Outputs:
    * Classification logits: [B, num_classes, H, W]
    * Regression deltas:     [B, 4,          H, W]  (anchor-free style)

Usage:
    heads = SiamTPNHeads(in_channels=192, cls_channels=[192,192], reg_channels=[192,192], num_classes=2)
    cls_maps, reg_maps = heads([p2, p3, p4])  # lists per scale
"""

from __future__ import annotations
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise 3x3 + Pointwise 1x1 with BN + ReLU
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class Tower(nn.Module):
    """
    A small conv tower composed of depthwise-separable convs.
    """
    def __init__(self, in_ch: int, channels: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        c_in = in_ch
        for c_out in channels:
            layers.append(DepthwiseSeparableConv(c_in, c_out))
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiamTPNHeads(nn.Module):
    """
    Shared heads for classification and regression across pyramid levels.

    Args:
        in_channels: channel dimension of pyramid features (C', e.g., 192)
        cls_channels: list of channels for the classification tower
        reg_channels: list of channels for the regression tower
        num_classes: number of classes (2: target vs background)
        bias_init: initial bias for classification logits (optional)

    Forward:
        feats: list[Tensor] of shape [B,C',H_i,W_i] for each pyramid level
    Returns:
        cls_logits_list: list[Tensor] of shape [B,num_classes,H_i,W_i]
        bbox_deltas_list: list[Tensor] of shape [B,4,H_i,W_i]
    """
    def __init__(
        self,
        in_channels: int = 192,
        cls_channels: Sequence[int] = (192, 192),
        reg_channels: Sequence[int] = (192, 192),
        num_classes: int = 2,
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        # Shared towers
        self.cls_tower = Tower(in_ch=self.in_channels, channels=cls_channels)
        self.reg_tower = Tower(in_ch=self.in_channels, channels=reg_channels)

        # Prediction heads
        cls_last_ch = int(cls_channels[-1]) if len(cls_channels) > 0 else self.in_channels
        reg_last_ch = int(reg_channels[-1]) if len(reg_channels) > 0 else self.in_channels

        self.cls_pred = nn.Conv2d(cls_last_ch, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.reg_pred = nn.Conv2d(reg_last_ch, 4, kernel_size=1, stride=1, padding=0)

        # Init biases (optional)
        nn.init.constant_(self.cls_pred.bias, bias_init)
        nn.init.zeros_(self.cls_pred.weight)
        nn.init.zeros_(self.reg_pred.bias)
        nn.init.zeros_(self.reg_pred.weight)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_logits_list: List[torch.Tensor] = []
        bbox_deltas_list: List[torch.Tensor] = []

        for f in feats:
            c = self.cls_pred(self.cls_tower(f))  # [B,num_classes,H,W]
            r = self.reg_pred(self.reg_tower(f))  # [B,4,H,W]
            cls_logits_list.append(c)
            bbox_deltas_list.append(r)

        return cls_logits_list, bbox_deltas_list


# ----------------------
# Simple self-test
# ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 2, 192
    p2 = torch.randn(B, C, 64, 64)
    p3 = torch.randn(B, C, 32, 32)
    p4 = torch.randn(B, C, 16, 16)

    heads = SiamTPNHeads(in_channels=C, cls_channels=[C, C], reg_channels=[C, C], num_classes=2)
    cls_list, reg_list = heads([p2, p3, p4])
    for i, (c, r) in enumerate(zip(cls_list, reg_list), start=2):
        print(f"P{i} cls shape: {tuple(c.shape)} | reg shape: {tuple(r.shape)}")