"""
ShuffleNetV2 backbone wrapper with pyramidal feature extraction (stages 2/3/4)
and per-level channel compression to a common dimension C' (default 192).

Design goals:
- Embedded-friendly: leverage timm's lightweight ShuffleNetV2 variants
- features_only=True to obtain intermediate stages with minimal overhead
- Compress each stage's channels to C' via 1x1 Conv + BN + ReLU
- Return multi-scale features suitable for MPA and heads

Outputs:
- Dict with keys: 'p2', 'p3', 'p4' where each Tensor is [B, C', H_i, W_i]
- Also returns a list [p2, p3, p4] via .forward for convenience

Notes:
- Stages mapping (timm's out_indices) for ShuffleNetV2 typically:
  0: stem
  1: stage2
  2: stage3
  3: stage4
We select out_indices=(1,2,3) to match stages 2/3/4.

"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

# torchvision fallback (feature extraction)
try:
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


def _ensure_timm():
    if not _HAS_TIMM:
        raise ImportError("timm is required for ShuffleNetV2Pyramid. "
                          "Install via pip: pip install timm>=0.9.2")


class Conv1x1BNReLU(nn.Module):
    """
    1x1 Conv + BN + ReLU for channel compression
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ShuffleNetV2Pyramid(nn.Module):
    """
    ShuffleNetV2 backbone wrapper producing pyramidal features.

    Args:
        variant: timm model name (e.g., 'shufflenetv2_x1_0', 'shufflenetv2_x0_5')
        pretrained: load ImageNet pretrained weights
        out_indices: tuple of stage indices to extract (default (1,2,3))
        compress_channels: common output channels per level (C')
        frozen: whether to freeze backbone parameters

    Usage:
        backbone = ShuffleNetV2Pyramid()
        feats = backbone(images)  # list [p2, p3, p4]
        feats_dict = backbone.forward_dict(images)  # dict {'p2': ..., 'p3': ..., 'p4': ...}
    """
    def __init__(self,
                 variant: str = "shufflenetv2_x1_0",
                 pretrained: bool = True,
                 out_indices: Tuple[int, int, int] = (1, 2, 3),
                 compress_channels: int = 192,
                 frozen: bool = False):
        super().__init__()
        # timm preferred; torchvision used as fallback
        self.compress_channels = int(compress_channels)

        # Create ShuffleNetV2 with feature outputs (timm or torchvision fallback)
        self.backbone = self._create_backbone_with_fallback(variant, pretrained, out_indices)

        # Feature channels of extracted stages
        # When using timm, feature_info is available.
        # For torchvision adapter, we infer channels by a dummy forward once lazily if needed.
        if hasattr(self.backbone, "feature_info"):
            feat_channels: List[int] = list(self.backbone.feature_info.channels())
            if len(feat_channels) != len(out_indices):
                all_channels = list(self.backbone.feature_info.channels())
                feat_channels = [all_channels[i] for i in out_indices]
        else:
            # Torchvision adapter path: infer channels with a tiny dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 256, 256)
                feats = self.backbone(dummy)  # list of tensors
                feat_channels = [int(f.shape[1]) for f in feats]

        # Per-level compressors to C'
        self.compressors = nn.ModuleList([
            Conv1x1BNReLU(in_ch=c, out_ch=self.compress_channels) for c in feat_channels
        ])

        if frozen:
            self.freeze_backbone()

    class _TorchvisionBackboneAdapter(nn.Module):
        """
        Adapts torchvision feature extractor to return list [p2, p3, p4] in the order of out_indices.
        """
        def __init__(self, extractor: nn.Module, ordered_keys: List[str]):
            super().__init__()
            self.extractor = extractor
            self.ordered_keys = ordered_keys

        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            feats_dict = self.extractor(x)
            return [feats_dict[k] for k in self.ordered_keys]

    def _create_backbone_with_fallback(self, variant: str, pretrained: bool, out_indices: Tuple[int, int, int]):
        """
        Try timm variants first; if unavailable, fallback to torchvision feature extractor for ShuffleNetV2.
        """
        # Try timm
        if _HAS_TIMM:
            candidates: List[str] = [variant]
            name = str(variant)
            if name in ("shufflenet_v2", "shufflenetv2_x1_0"):
                candidates = [
                    "shufflenetv2_x1_0",
                    "shufflenetv2_100",
                    "shufflenetv2_1.0",
                    "shufflenetv2_x0_5",
                    "shufflenetv2_050",
                    "shufflenetv2_0.5",
                ]
            last_err: Optional[Exception] = None
            for n in candidates:
                try:
                    return timm.create_model(
                        n,
                        pretrained=pretrained,
                        features_only=True,
                        out_indices=out_indices,
                    )
                except Exception as e:
                    last_err = e
                    continue
            # If timm failed, continue to torchvision fallback (do not raise yet)

        # Torchvision fallback
        if _HAS_TORCHVISION:
            # Build torchvision shufflenet v2 x1_0
            tv_variant = "shufflenet_v2_x1_0"
            try:
                # Weights handling across torchvision versions
                weights = None
                if pretrained:
                    try:
                        from torchvision.models import ShuffleNet_V2_X1_0_Weights
                        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
                    except Exception:
                        weights = None
                tv_model = torchvision.models.shufflenet_v2_x1_0(weights=weights)
            except Exception:
                # Fallback: create without weights
                tv_model = torchvision.models.shufflenet_v2_x1_0(weights=None)

            # Create feature extractor for stages 2/3/4
            return_nodes = {"stage2": "s2", "stage3": "s3", "stage4": "s4"}
            extractor = create_feature_extractor(tv_model, return_nodes=return_nodes)

            # Map timm-style out_indices (1,2,3) -> ordered keys ['s2','s3','s4']
            stage_map = {1: "s2", 2: "s3", 3: "s4"}
            ordered_keys = [stage_map.get(int(i), "s2") for i in out_indices]
            return self._TorchvisionBackboneAdapter(extractor, ordered_keys)

        # Neither timm nor torchvision path succeeded
        raise RuntimeError("No available backbone provider (timm/torchvision) for ShuffleNetV2.")

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _check_forward_shape(self, x: torch.Tensor):
        """
        One-time sanity check for feature shapes; helpful for debugging.
        """
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)) or len(feats) != len(self.compressors):
            raise RuntimeError(f"Unexpected features from backbone: got {type(feats)} len={len(feats)}")
        return [f.shape for f in feats]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning a list [p2, p3, p4] compressed to C'.
        Args:
            x: input images [B, 3, H, W] normalized to ImageNet stats
        Returns:
            list of tensors: [B, C', H_i, W_i] for i in {2,3,4}
        """
        feats = self.backbone(x)  # list of stage features
        out: List[torch.Tensor] = []
        for f, comp in zip(feats, self.compressors):
            out.append(comp(f))
        return out

    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward returning a dict {'p2', 'p3', 'p4'}
        """
        p2, p3, p4 = self.forward(x)
        return {"p2": p2, "p3": p3, "p4": p4}

    @staticmethod
    def from_config(cfg: Dict[str, object]) -> "ShuffleNetV2Pyramid":
        """
        Build backbone from config dict.
        Expected keys:
            model.backbone (str)
            model.backbone_pretrained (str or bool)
            model.backbone_out_stages (list/tuple of ints)
            model.compress_channels (int)
        """
        variant = str(cfg.get("model.backbone", "shufflenet_v2"))
        # Map friendly name to timm variant
        if variant == "shufflenet_v2":
            variant = "shufflenetv2_x1_0"
        pretrained = cfg.get("model.backbone_pretrained", "imagenet")
        pretrained_flag = bool(pretrained) and str(pretrained).lower() in ("1", "true", "imagenet", "yes")
        out_stages = cfg.get("model.backbone_out_stages", [2, 3, 4])

        # timm indices mapping to stages 2/3/4 -> (1,2,3)
        stage_to_indices = {2: 1, 3: 2, 4: 3}
        out_indices = tuple(stage_to_indices.get(int(s), 1) for s in out_stages)

        c_out = int(cfg.get("model.compress_channels", 192))
        return ShuffleNetV2Pyramid(
            variant=variant,
            pretrained=pretrained_flag,
            out_indices=out_indices,
            compress_channels=c_out,
            frozen=False,
        )


# ----------------------
# Simple self-test
# ----------------------
if __name__ == "__main__":
    _ensure_timm()
    model = ShuffleNetV2Pyramid(variant="shufflenetv2_x1_0", pretrained=False, out_indices=(1, 2, 3), compress_channels=192)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        outs = model(x)
    print("Outputs:")
    for i, o in enumerate(outs, start=2):
        print(f"P{i}: shape={o.shape}")