"""
T-SiamTPN: Temporal Siamese Transformer Pyramid Networks (embedded-friendly)

This module integrates:
- ShuffleNetV2 backbone with pyramidal multi-scale feature extraction (P2/P3/P4)
- Modulated Pooling Attention (MPA) for template/search fusion
- Lightweight classification/regression heads
- Static + dynamic template handling with concatenation along channel dimension
  followed by 1x1 projection back to C' (default 192)

Design choices for embedded systems:
- All feature channels compressed to a common dim C' (default 192)
- K/V token count reduced via spatial pooling inside MPA
- Shared heads across scales reduce parameter count
- Template concatenation fixed to a maximum count (1 static + 4 dynamic = 5) to
  enable static 1x1 projector without dynamic re-parameterization overhead.

Note:
- Temporal template management algorithms (smoothing, correction, update policy)
  are implemented in src/tracking/template_manager.py. This model provides
  minimal utilities to set/update template images and cache their features.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.shufflenet_v2_pyramid import ShuffleNetV2Pyramid, Conv1x1BNReLU
from .blocks.mpa import MultiScaleMPA
from .heads.cls_reg_heads import SiamTPNHeads


class T_SiamTPN(nn.Module):
    """
    Core T-SiamTPN model.

    Args:
        backbone: backbone module; if None, will be constructed from config via from_config
        dim: common feature channels C' for pyramidal features
        num_scales: number of pyramid levels (default 3 -> P2/P3/P4)
        max_templates: maximum templates concatenated (default 5 = 1 static + 4 dynamic)
        mpa_heads: number of attention heads in MPA
        mpa_pool: pooling type for K/V reduction ("avg" or "max")
        mpa_pool_stride: stride for pooling K/V
        mpa_dropout: dropout prob inside MPA
        mpa_modulation: modulation gate type ("sigmoid" or "tanh")
        head_cls_channels: tower channels for classification head
        head_reg_channels: tower channels for regression head
        num_classes: output classes (2: target vs background)
    """
    def __init__(
        self,
        backbone: Optional[ShuffleNetV2Pyramid] = None,
        dim: int = 192,
        num_scales: int = 3,
        max_templates: int = 5,
        mpa_heads: int = 4,
        mpa_pool: str = "avg",
        mpa_pool_stride: int = 2,
        mpa_dropout: float = 0.0,
        mpa_modulation: str = "sigmoid",
        head_cls_channels: Tuple[int, int] = (192, 192),
        head_reg_channels: Tuple[int, int] = (192, 192),
        num_classes: int = 2,
    ):
        super().__init__()
        assert num_scales == 3, "This implementation assumes P2/P3/P4 (num_scales=3)."
        self.dim = int(dim)
        self.num_scales = int(num_scales)
        self.max_templates = int(max_templates)

        self.backbone = backbone or ShuffleNetV2Pyramid(
            variant="shufflenetv2_x1_0",
            pretrained=True,
            out_indices=(1, 2, 3),  # stages 2/3/4
            compress_channels=self.dim,
            frozen=False,
        )

        self.mpa = MultiScaleMPA(
            dim=self.dim,
            num_scales=self.num_scales,
            num_heads=int(mpa_heads),
            pool_type=str(mpa_pool),
            pool_stride=int(mpa_pool_stride),
            dropout=float(mpa_dropout),
            modulation=str(mpa_modulation),
        )

        self.heads = SiamTPNHeads(
            in_channels=self.dim,
            cls_channels=head_cls_channels,
            reg_channels=head_reg_channels,
            num_classes=int(num_classes),
        )

        # Per-scale projectors to reduce concatenated template channels (T*C') -> C'
        in_ch = self.dim * self.max_templates
        self.template_projectors = nn.ModuleList([
            Conv1x1BNReLU(in_ch=in_ch, out_ch=self.dim) for _ in range(self.num_scales)
        ])

        # Internal template feature cache: list of template features, each as a list per scale
        # _template_feats = List[ List[Tensor] ], outer length <= max_templates, inner length == num_scales
        self._template_feats: List[List[torch.Tensor]] = []

    # -------------------------------------------------------------------------
    # Template utilities
    # -------------------------------------------------------------------------
    def clear_templates(self):
        self._template_feats = []

    @torch.no_grad()
    def set_templates(self, static_template_img: torch.Tensor, dynamic_template_imgs: Optional[List[torch.Tensor]] = None):
        """
        Set static + dynamic templates and cache their pyramid features.

        Args:
            static_template_img: [B,3,H,W]
            dynamic_template_imgs: list of [B,3,H,W], length <= (max_templates-1)
        """
        self._template_feats = []
        static_feats = self._extract_pyramid(static_template_img)  # List[P2,P3,P4]
        self._template_feats.append(static_feats)

        if dynamic_template_imgs:
            for img in dynamic_template_imgs[: max(0, self.max_templates - 1)]:
                dyn_feats = self._extract_pyramid(img)
                self._template_feats.append(dyn_feats)

        # If fewer than max_templates, we don't pad now; padding is performed at fusion time.

    @torch.no_grad()
    def update_dynamic_template(self, new_template_img: torch.Tensor, confidence: float, frame_idx: int,
                                update_interval: int = 5, conf_thresh: float = 0.5):
        """
        Minimal dynamic template update policy consistent with Algorithm 3:

        Trigger condition:
            frame_idx % update_interval == 0 and confidence >= conf_thresh

        Action:
            - If triggered:
                * Ensure at least 1 slot for dynamic templates (max_templates-1)
                * If capacity not reached, append new features
                * Else, drop the oldest dynamic template (index 1) and append new one
            - Static template at index 0 remains unchanged.
        """
        if (frame_idx % int(update_interval) == 0) and (float(confidence) >= float(conf_thresh)):
            new_feats = self._extract_pyramid(new_template_img)
            if len(self._template_feats) < self.max_templates:
                # If no templates set yet, we cannot determine static; enforce at least static exists
                if len(self._template_feats) == 0:
                    raise RuntimeError("Static template not set. Call set_templates() first.")
                # Append dynamic
                self._template_feats.append(new_feats)
            else:
                # Replace oldest dynamic (index 1)
                if len(self._template_feats) >= 2:
                    self._template_feats.pop(1)
                    self._template_feats.append(new_feats)
                else:
                    # Only static present; append new dynamic (bounded by max_templates)
                    self._template_feats.append(new_feats)

    # -------------------------------------------------------------------------
    # Forward path
    # -------------------------------------------------------------------------
    def forward(self, search_img: torch.Tensor,
                templates_imgs: Optional[List[torch.Tensor]] = None) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            search_img: [B,3,Hx,Wx] normalized to ImageNet stats
            templates_imgs: optional list of [B,3,Hz,Wz]; if provided, will be used
                            and cached as current template features.

        Returns:
            Dict with keys:
                - "cls": list[Tensor] per-scale logits [B,num_classes,H_i,W_i]
                - "reg": list[Tensor] per-scale deltas [B,4,H_i,W_i]
                - "fused": list[Tensor] per-scale fused features [B,C',H_i,W_i]
                - "search_feats": list[Tensor] per-scale search features [B,C',H_i,W_i]
        """
        # Update template cache if raw images are provided
        if templates_imgs is not None:
            if len(templates_imgs) == 0:
                raise ValueError("templates_imgs list is empty. Provide at least the static template.")
            # First image considered static; rest considered dynamic
            static_img = templates_imgs[0]
            dynamic_imgs = templates_imgs[1:] if len(templates_imgs) > 1 else []
            self.set_templates(static_img, dynamic_imgs)

        # Extract search pyramid features
        search_feats = self._extract_pyramid(search_img)  # [P2,P3,P4]

        # Prepare K/V features from templates: concat along channels then project back to C'
        kv_feats = self._fused_template_feats(search_feats)  # shape-aligned per-scale

        # MPA fusion
        fused_feats = self.mpa(search_feats, kv_feats)

        # Heads
        cls_list, reg_list = self.heads(fused_feats)

        return {
            "cls": cls_list,
            "reg": reg_list,
            "fused": fused_feats,
            "search_feats": search_feats,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _extract_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Run backbone and return compressed pyramidal features [P2,P3,P4].
        """
        feats = self.backbone(img)
        assert isinstance(feats, (list, tuple)) and len(feats) == self.num_scales, "Unexpected backbone output"
        return list(feats)

    def _fused_template_feats(self, ref_shapes_from_search: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Concatenate template features along channel dimension and project to C'.

        Behavior:
            - If fewer templates than max_templates, pad with zeros to match in_ch=C'*max_templates
            - If no templates are set, raises RuntimeError

        Args:
            ref_shapes_from_search: list of search features used to infer H_i/W_i for padding
        Returns:
            per-scale list of fused K/V features [B,C',H_i,W_i]
        """
        if len(self._template_feats) == 0:
            raise RuntimeError("No templates set. Call set_templates() or forward with templates_imgs.")

        B = ref_shapes_from_search[0].shape[0]
        kv_list: List[torch.Tensor] = []

        # For each scale, assemble list of per-template features [B,C',H,W]
        for s in range(self.num_scales):
            per_template_feats: List[torch.Tensor] = []
            for t_feats in self._template_feats:
                per_template_feats.append(t_feats[s])

            # Pad with zeros if fewer than max_templates
            num_current = len(per_template_feats)
            if num_current < self.max_templates:
                # reference shape from search (B,C',H,W)
                ref = ref_shapes_from_search[s]
                pad_needed = self.max_templates - num_current
                zero_feat = torch.zeros_like(ref)  # [B,C',H,W]
                for _ in range(pad_needed):
                    per_template_feats.append(zero_feat)

            # Concatenate along channels: [B, T*C', H, W]
            concat = torch.cat(per_template_feats, dim=1)
            # Project back to C'
            kv = self.template_projectors[s](concat)
            kv_list.append(kv)

        return kv_list

    # -------------------------------------------------------------------------
    # Config construction
    # -------------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: Dict[str, object]) -> "T_SiamTPN":
        """
        Build T_SiamTPN from a flat config dict (keys as 'section.key' or nested dicts).
        Required/optional keys:
            model.backbone
            model.backbone_pretrained
            model.backbone_out_stages
            model.compress_channels
            model.mpa.num_heads
            model.mpa.pooling
            model.mpa.dropout
            model.mpa.modulation
            model.heads.cls_channels
            model.heads.reg_channels
            model.heads.num_classes
            templates.static_count
            templates.dynamic_count
        """
        # Resolve dim and template counts
        dim = int(_get(cfg, ["model.compress_channels"], default=192))
        static_count = int(_get(cfg, ["templates.static_count"], default=1))
        dynamic_count = int(_get(cfg, ["templates.dynamic_count"], default=4))
        max_templates = static_count + dynamic_count

        # Build backbone via its own config constructor
        backbone = ShuffleNetV2Pyramid.from_config({
            "model.backbone": _get(cfg, ["model.backbone"], default="shufflenet_v2"),
            "model.backbone_pretrained": _get(cfg, ["model.backbone_pretrained"], default="imagenet"),
            "model.backbone_out_stages": _get(cfg, ["model.backbone_out_stages"], default=[2, 3, 4]),
            "model.compress_channels": dim,
        })

        # MPA params
        mpa_heads = int(_get(cfg, ["model.mpa.num_heads"], default=4))
        mpa_pool = str(_get(cfg, ["model.mpa.pooling"], default="avg"))
        mpa_dropout = float(_get(cfg, ["model.mpa.dropout"], default=0.0))
        mpa_modulation = str(_get(cfg, ["model.mpa.modulation"], default="sigmoid"))

        # Heads params
        cls_channels = tuple(_get(cfg, ["model.heads.cls_channels"], default=[dim, dim]))
        reg_channels = tuple(_get(cfg, ["model.heads.reg_channels"], default=[dim, dim]))
        num_classes = int(_get(cfg, ["model.heads.num_classes"], default=2))

        return T_SiamTPN(
            backbone=backbone,
            dim=dim,
            num_scales=3,
            max_templates=max_templates,
            mpa_heads=mpa_heads,
            mpa_pool=mpa_pool,
            mpa_pool_stride=2,
            mpa_dropout=mpa_dropout,
            mpa_modulation=mpa_modulation,
            head_cls_channels=cls_channels,
            head_reg_channels=reg_channels,
            num_classes=num_classes,
        )


# -------------------------------------------------------------------------
# Small config helper
# -------------------------------------------------------------------------
def _get(cfg: Dict[str, object], keys: List[str], default):
    """
    Helper to fetch value from either flat 'section.key' dict or nested dict.

    Example:
        cfg["model.compress_channels"] OR cfg["model"]["compress_channels"]
    """
    # Try flat
    k = ".".join(keys)
    if k in cfg:
        return cfg[k]
    # Try nested progressively
    cur = cfg
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2
    dim = 192
    # Build from minimal defaults
    model = T_SiamTPN(
        dim=dim,
        max_templates=5,
        mpa_heads=4,
        mpa_pool="avg",
        mpa_pool_stride=2,
        mpa_dropout=0.0,
        mpa_modulation="sigmoid",
        head_cls_channels=(dim, dim),
        head_reg_channels=(dim, dim),
        num_classes=2,
    )

    # Dummy inputs
    search = torch.randn(B, 3, 256, 256)
    z_static = torch.randn(B, 3, 128, 128)
    z_dyn1 = torch.randn(B, 3, 128, 128)
    z_dyn2 = torch.randn(B, 3, 128, 128)

    out = model(search_img=search, templates_imgs=[z_static, z_dyn1, z_dyn2])
    print("Per-scale outputs:")
    for i, (c, r) in enumerate(zip(out["cls"], out["reg"]), start=2):
        print(f"P{i} cls={tuple(c.shape)} reg={tuple(r.shape)}")