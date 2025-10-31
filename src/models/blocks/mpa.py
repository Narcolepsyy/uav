"""
Modulated Pooling Attention (MPA) block for T-SiamTPN.

Embedded-friendly characteristics:
- 1x1 conv projections for Q/K/V (no heavy linear layers)
- Spatial pooling for K/V to reduce attention token count
- Lightweight modulation gate computed from Q
- Operates on [B, C, H, W] feature maps

Typical usage:
    mpa = ModulatedPoolingAttention(dim=192, num_heads=4)
    y = mpa(search_feat, template_feat_concat)  # y has same spatial size as search_feat
"""

from __future__ import annotations
from typing import Optional, Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedPoolingAttention(nn.Module):
    """
    Modulated Pooling Attention block.

    Args:
        dim: channel dimension of inputs (C, after compression to C')
        num_heads: number of attention heads
        pool_type: 'avg' or 'max' pooling for K/V spatial reduction
        pool_stride: stride (and kernel size) for pooling K/V
        qkv_bias: add bias in 1x1 conv projections
        dropout: dropout on attention probabilities and output projection
        modulation: 'sigmoid' or 'tanh' gating computed from Q

    Input:
        q: Query from search branch, Tensor [B, C, Hq, Wq]
        kv: Key/Value from template branch, Tensor [B, C, Hk, Wk]
    Output:
        out: Tensor [B, C, Hq, Wq]
    """
    def __init__(
        self,
        dim: int = 192,
        num_heads: int = 4,
        pool_type: Literal["avg", "max"] = "avg",
        pool_stride: int = 2,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        modulation: Literal["sigmoid", "tanh"] = "sigmoid",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.modulation = modulation

        # Projections (1x1 convs keep spatial resolution)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        # Gating from Q
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid() if modulation == "sigmoid" else nn.Tanh(),
        )

        # Pooling for K/V
        if pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride, ceil_mode=False)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=pool_stride, stride=pool_stride, ceil_mode=False)
        else:
            raise ValueError(f"Unsupported pool_type={pool_type}")

        # Dropouts
        self.attn_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, C, H, W] -> [B, heads, HW, head_dim]
        """
        B, C, H, W = x.shape
        x = x.view(B, self.num_heads, self.head_dim, H, W)           # [B, h, d, H, W]
        x = x.permute(0, 1, 3, 4, 2).contiguous()                    # [B, h, H, W, d]
        x = x.view(B, self.num_heads, H * W, self.head_dim)         # [B, h, N, d]
        return x

    def _reshape_from_heads(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        [B, heads, HW, head_dim] -> [B, C, H, W]
        """
        B, h, N, d = x.shape
        assert h == self.num_heads and d == self.head_dim
        x = x.view(B, h, H, W, d).permute(0, 1, 4, 2, 3).contiguous()  # [B, h, d, H, W]
        x = x.view(B, self.dim, H, W)                                  # [B, C, H, W]
        return x

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q: [B, C, Hq, Wq]
        kv: [B, C, Hk, Wk]
        """
        B, C, Hq, Wq = q.shape

        # Projections
        q_proj = self.q_proj(q)                          # [B, C, Hq, Wq]
        k_proj = self.k_proj(kv)                         # [B, C, Hk, Wk]
        v_proj = self.v_proj(kv)                         # [B, C, Hk, Wk]

        # Spatial reduction for K/V to reduce tokens
        k_proj_p = self.pool(k_proj)                     # [B, C, Hk', Wk']
        v_proj_p = self.pool(v_proj)                     # [B, C, Hk', Wk']
        Hk_p, Wk_p = k_proj_p.shape[2], k_proj_p.shape[3]

        # Reshape to heads
        qh = self._reshape_to_heads(q_proj)              # [B, h, Nq, d]
        kh = self._reshape_to_heads(k_proj_p)            # [B, h, Nk, d]
        vh = self._reshape_to_heads(v_proj_p)            # [B, h, Nk, d]

        # Attention
        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale   # [B, h, Nq, Nk]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, vh)                     # [B, h, Nq, d]
        out = self._reshape_from_heads(out, Hq, Wq)      # [B, C, Hq, Wq]

        # Modulation gate from Q (same resolution as output)
        gate = self.gate(q_proj)                         # [B, C, Hq, Wq]
        out = out * gate

        # Final projection + dropout
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class MultiScaleMPA(nn.Module):
    """
    Apply MPA on multi-scale features with shared configuration.

    Args:
        dim: channels for each pyramid level (assumed equal C')
        num_scales: number of pyramid levels (e.g., 3 for P2/P3/P4)
        num_heads: attention heads
        pool_type: 'avg' or 'max'
        pool_stride: pooling stride for K/V
        dropout: dropout prob

    Inputs:
        q_list: list of search features [B, C, H_i, W_i]
        kv_list: list of template features [B, C, H_i, W_i] (same length/order)
    Output:
        out_list: list of fused features, same shapes as q_list
    """
    def __init__(
        self,
        dim: int = 192,
        num_scales: int = 3,
        num_heads: int = 4,
        pool_type: Literal["avg", "max"] = "avg",
        pool_stride: int = 2,
        dropout: float = 0.0,
        modulation: Literal["sigmoid", "tanh"] = "sigmoid",
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ModulatedPoolingAttention(
                dim=dim,
                num_heads=num_heads,
                pool_type=pool_type,
                pool_stride=pool_stride,
                qkv_bias=True,
                dropout=dropout,
                modulation=modulation,
            ) for _ in range(num_scales)
        ])

    def forward(self, q_list: list[torch.Tensor], kv_list: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(q_list) == len(kv_list) == len(self.blocks), "Scale count mismatch"
        out_list = []
        for q, kv, blk in zip(q_list, kv_list, self.blocks):
            out_list.append(blk(q, kv))
        return out_list


# ----------------------
# Simple self-test
# ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 2, 192
    q2 = torch.randn(B, C, 64, 64)
    q3 = torch.randn(B, C, 32, 32)
    q4 = torch.randn(B, C, 16, 16)

    # Template features (concatenation of static+dynamic templates should be done prior to passing here
    # and then projected back to C' if needed. For this test we keep C' constant.)
    kv2 = torch.randn(B, C, 64, 64)
    kv3 = torch.randn(B, C, 32, 32)
    kv4 = torch.randn(B, C, 16, 16)

    mpa = MultiScaleMPA(dim=C, num_scales=3, num_heads=4, pool_type="avg", pool_stride=2, dropout=0.0)
    outs = mpa([q2, q3, q4], [kv2, kv3, kv4])
    for i, o in enumerate(outs, start=2):
        print(f"P{i} out shape: {tuple(o.shape)}")