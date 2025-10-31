"""
Temporal Template Manager for T-SiamTPN.

Implements three algorithms described for robust UAV tracking:
1) Box Smoothing (Algorithm 1)
2) Box Correction (Algorithm 2)
3) Dynamic Template Update (Algorithm 3)

Design:
- Batch-friendly but optimized for single-object tracking per stream
- Torch-first API; states kept on the same device as inputs
- Independent from model: produces corrected box and indicates when
  to update dynamic templates; higher-level tracking loop coordinates
  actual template feature updates in the model.

References:
- Box smoothing: S_i = alpha * S_{i-1} + (1 - alpha) * y_i
- Box correction thresholds:
    * center distance d vs stable S_s
    * scale ratio r vs 1.0
- Dynamic update trigger:
    * (frame_idx % update_interval == 0) and (confidence >= confidence_thresh)
    * Drop oldest dynamic template, append new, keep static unchanged
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List

import torch

from src.utils.box_ops import (
    smooth_box,
    correct_box,
    center_distance,
    scale_ratio,
)

class TemplateManager:
    """
    Temporal template manager encapsulating smoothing, correction, and update policy.

    States:
        prev_smooth_cxcywh: previous smoothed box S_{i-1} [N,4]
        stable_cxcywh:      stable box S_s used for correction fallback [N,4]
        static_template_img:  Tensor [B,3,H,W]
        dynamic_template_imgs: list[Tensor] of length <= max_dynamic

    Args:
        smoothing_alpha: alpha in [0,1] for linear smoothing
        center_distance_thresh: distance threshold (normalized or pixels depending on your preprocessing)
        scale_ratio_thresh: acceptable |r - 1.0| bound
        update_interval: frames between dynamic template updates
        confidence_thresh: minimum confidence to allow updates
        max_dynamic: maximum dynamic templates maintained (default 4)
    """
    def __init__(
        self,
        smoothing_alpha: float = 0.8,
        center_distance_thresh: float = 0.25,
        scale_ratio_thresh: float = 0.25,
        update_interval: int = 5,
        confidence_thresh: float = 0.5,
        max_dynamic: int = 4,
    ):
        self.smoothing_alpha = float(smoothing_alpha)
        self.center_distance_thresh = float(center_distance_thresh)
        self.scale_ratio_thresh = float(scale_ratio_thresh)
        self.update_interval = int(update_interval)
        self.confidence_thresh = float(confidence_thresh)
        self.max_dynamic = int(max_dynamic)

        # Tracking states
        self.prev_smooth_cxcywh: Optional[torch.Tensor] = None
        self.stable_cxcywh: Optional[torch.Tensor] = None

        # Template images cache (raw images; model caches features)
        self.static_template_img: Optional[torch.Tensor] = None
        self.dynamic_template_imgs: List[torch.Tensor] = []

    # -------------------------------------------------------------------------
    # Template image management
    # -------------------------------------------------------------------------
    def reset(self, static_template_img: torch.Tensor, dynamic_template_imgs: Optional[List[torch.Tensor]] = None):
        """
        Reset template images and internal states.
        """
        self.static_template_img = static_template_img
        self.dynamic_template_imgs = list(dynamic_template_imgs or [])[: self.max_dynamic]
        self.prev_smooth_cxcywh = None
        self.stable_cxcywh = None

    def get_templates(self) -> List[torch.Tensor]:
        """
        Return the current static + dynamic template images (static first).
        """
        if self.static_template_img is None:
            return []
        return [self.static_template_img] + list(self.dynamic_template_imgs)

    # -------------------------------------------------------------------------
    # Core temporal processing per frame
    # -------------------------------------------------------------------------
    def process_frame(
        self,
        pred_cxcywh: torch.Tensor,
        confidence: float,
        frame_idx: int,
    ) -> Dict[str, Any]:
        """
        Execute smoothing, correction, and decide on dynamic template update.

        Args:
            pred_cxcywh: Tensor [N,4] predicted box in cxcywh format
            confidence:  scalar float confidence score for the prediction
            frame_idx:   current frame index (int)

        Returns:
            dict with keys:
                'smoothed_cxcywh': Tensor [N,4]
                'corrected_cxcywh': Tensor [N,4]
                'should_update': bool
        """
        if pred_cxcywh.ndim != 2 or pred_cxcywh.shape[-1] != 4:
            raise ValueError("pred_cxcywh must be [N,4] tensor")

        device = pred_cxcywh.device

        # 1) Smoothing
        if self.prev_smooth_cxcywh is None:
            smoothed = pred_cxcywh
        else:
            smoothed = smooth_box(self.prev_smooth_cxcywh.to(device), pred_cxcywh, alpha=self.smoothing_alpha)
        self.prev_smooth_cxcywh = smoothed.detach()

        # Initialize stable state if needed
        if self.stable_cxcywh is None:
            self.stable_cxcywh = smoothed.detach()

        # 2) Correction against stable box S_s
        corrected = correct_box(
            stable_cxcywh=self.stable_cxcywh.to(device),
            cur_pred_cxcywh=smoothed,
            center_dist_thresh=self.center_distance_thresh,
            scale_ratio_thresh=self.scale_ratio_thresh,
        )

        # Update stable state to corrected (keeps stability over time)
        self.stable_cxcywh = corrected.detach()

        # 3) Dynamic template update decision
        should_update = self.should_update_dynamic(confidence=confidence, frame_idx=frame_idx)

        return {
            "smoothed_cxcywh": smoothed,
            "corrected_cxcywh": corrected,
            "should_update": bool(should_update),
        }

    def should_update_dynamic(self, confidence: float, frame_idx: int) -> bool:
        """
        Decide if we should update dynamic templates based on Algorithm 3.
        Trigger when:
            (frame_idx % update_interval == 0) and (confidence >= confidence_thresh)
        """
        if self.update_interval <= 0:
            return False
        return (int(frame_idx) % self.update_interval == 0) and (float(confidence) >= self.confidence_thresh)

    def apply_dynamic_update(self, new_template_img: torch.Tensor):
        """
        Apply dynamic template update: drop oldest dynamic (index 0 of dynamic list)
        then append new template image. Static template is kept unchanged.
        """
        if self.static_template_img is None:
            raise RuntimeError("Static template not set. Call reset() first.")

        if len(self.dynamic_template_imgs) >= self.max_dynamic and self.max_dynamic > 0:
            # Drop oldest
            self.dynamic_template_imgs.pop(0)

        if self.max_dynamic > 0:
            self.dynamic_template_imgs.append(new_template_img)

    # -------------------------------------------------------------------------
    # Utility accessors
    # -------------------------------------------------------------------------
    def get_prev_smooth(self) -> Optional[torch.Tensor]:
        return self.prev_smooth_cxcywh

    def get_stable(self) -> Optional[torch.Tensor]:
        return self.stable_cxcywh

    # -------------------------------------------------------------------------
    # Config construction
    # -------------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "TemplateManager":
        """
        Create TemplateManager from configuration dict.

        Expected keys (flat or nested):
            templates.dynamic_count
            templates.temporal.smoothing_alpha
            templates.temporal.correction.center_distance_thresh
            templates.temporal.correction.scale_ratio_thresh
            templates.temporal.update.interval
            templates.temporal.update.confidence_thresh
        """
        def _get(keys: List[str], default: Any) -> Any:
            # Try flat
            k = ".".join(keys)
            if k in cfg:
                return cfg[k]
            # Try nested
            cur: Any = cfg
            for key in keys:
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    return default
            return cur

        max_dynamic = int(_get(["templates", "dynamic_count"], default=4))
        smoothing_alpha = float(_get(["templates", "temporal", "smoothing_alpha"], default=0.8))
        center_distance_thresh = float(_get(["templates", "temporal", "correction", "center_distance_thresh"], default=0.25))
        scale_ratio_thresh = float(_get(["templates", "temporal", "correction", "scale_ratio_thresh"], default=0.25))
        update_interval = int(_get(["templates", "temporal", "update", "interval"], default=5))
        confidence_thresh = float(_get(["templates", "temporal", "update", "confidence_thresh"], default=0.5))

        return TemplateManager(
            smoothing_alpha=smoothing_alpha,
            center_distance_thresh=center_distance_thresh,
            scale_ratio_thresh=scale_ratio_thresh,
            update_interval=update_interval,
            confidence_thresh=confidence_thresh,
            max_dynamic=max_dynamic,
        )


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    tm = TemplateManager(smoothing_alpha=0.8, center_distance_thresh=0.25, scale_ratio_thresh=0.25,
                         update_interval=5, confidence_thresh=0.5, max_dynamic=4)

    # Simulate a stream of predictions
    N = 1
    pred = torch.tensor([[100.0, 100.0, 50.0, 50.0]])  # cx,cy,w,h
    tm.reset(static_template_img=torch.randn(1, 3, 128, 128), dynamic_template_imgs=[torch.randn(1, 3, 128, 128)])

    for i in range(1, 16):
        # Add some jitter
        jitter = torch.randn_like(pred) * 2.0
        cur_pred = pred + jitter

        # Confidence ramp
        conf = 0.4 + 0.05 * (i % 10)

        out = tm.process_frame(cur_pred, confidence=conf, frame_idx=i)
        print(f"Frame {i:02d} | should_update={out['should_update']} | "
              f"cx={out['corrected_cxcywh'][0,0]:.1f} cy={out['corrected_cxcywh'][0,1]:.1f} "
              f"w={out['corrected_cxcywh'][0,2]:.1f} h={out['corrected_cxcywh'][0,3]:.1f}")

        if out["should_update"]:
            tm.apply_dynamic_update(new_template_img=torch.randn(1, 3, 128, 128))

    print(f"Dynamic templates count: {len(tm.dynamic_template_imgs)}")