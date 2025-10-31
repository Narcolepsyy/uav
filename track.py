#!/usr/bin/env python3
"""
Inference/Tracking script for T-SiamTPN on UAV videos.

Key features:
- Loads YAML config and optional checkpoint
- Accepts input video path, initial bbox, or static template image
- Implements embedded-friendly cropping strategy:
  * Search crop = bbox expanded by context factor, resized to configured size
  * Static template = object_images/img_1.jpg if available or template path or first-frame bbox crop
  * Dynamic templates updated on schedule with confidence gating
- Temporal stabilization:
  * Linear smoothing S_i = alpha * S_{i-1} + (1 - alpha) * y_i (normalized coords)
  * Box correction: revert to stable box if center distance or scale ratio deviates beyond thresholds
- Visualizes tracking results and optional video save

Usage examples:
  python track.py --video observing/train/samples/Backpack_0/drone_video.mp4 --init_bbox 321,0,381,12
  python track.py --video observing/train/samples/Backpack_0/drone_video.mp4 --static_template observing/train/samples/Backpack_0/object_images/img_1.jpg --init_bbox 321,0,381,12 --checkpoint runs/checkpoint_epoch_50.pth --save_video out.mp4
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import os
import sys
import argparse
import time
import glob
import math

import yaml
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure local imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.t_siamtpn import T_SiamTPN  # type: ignore


# -----------------------------------------------------------------------------
# Image and box helpers (aligned with dataset)
# -----------------------------------------------------------------------------

def _ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise RuntimeError("Received None image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def _normalize(img_rgb: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    t = torch.from_numpy(img_chw).float()
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (t - mean_t) / std_t


def _crop_resize(img_rgb: np.ndarray, bbox_xyxy: Tuple[float, float, float, float], out_size: Tuple[int, int]) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(np.floor(x1))))
    y1 = max(0, min(h - 1, int(np.floor(y1))))
    x2 = max(0, min(w - 1, int(np.ceil(x2))))
    y2 = max(0, min(h - 1, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        cx = w // 2
        cy = h // 2
        half = max(1, min(w, h) // 4)
        x1 = max(0, cx - half); x2 = min(w - 1, cx + half)
        y1 = max(0, cy - half); y2 = min(h - 1, cy + half)
    patch = img_rgb[y1:y2, x1:x2]
    patch = cv2.resize(patch, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
    return patch


def _expand_bbox(xyxy: Tuple[float, float, float, float], factor: float, img_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1e-6, (x2 - x1))
    h = max(1e-6, (y2 - y1))
    nw = w * factor
    nh = h * factor
    nx1 = cx - 0.5 * nw
    ny1 = cy - 0.5 * nh
    nx2 = cx + 0.5 * nw
    ny2 = cy + 0.5 * nh
    H, W = img_size
    nx1 = max(0, min(W - 1, nx1))
    ny1 = max(0, min(H - 1, ny1))
    nx2 = max(0, min(W - 1, nx2))
    ny2 = max(0, min(H - 1, ny2))
    return float(nx1), float(ny1), float(nx2), float(ny2)


def _cxcywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return x1, y1, x2, y2


def _xyxy_to_cxcywh(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h


def _map_box_to_crop(orig_xyxy: Tuple[float, float, float, float],
                     crop_xyxy: Tuple[float, float, float, float],
                     dst_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Map original box coords to the resized crop coordinate system (pixels), return cxcywh.
    """
    ox1, oy1, ox2, oy2 = orig_xyxy
    cx1, cy1, cx2, cy2 = crop_xyxy
    cw = max(1e-6, cx2 - cx1)
    ch = max(1e-6, cy2 - cy1)
    H, W = dst_size
    sx = W / cw
    sy = H / ch
    x1p = (ox1 - cx1) * sx
    y1p = (oy1 - cy1) * sy
    x2p = (ox2 - cx1) * sx
    y2p = (oy2 - cy1) * sy
    cx = 0.5 * (x1p + x2p)
    cy = 0.5 * (y1p + y2p)
    w = max(0.0, x2p - x1p)
    h = max(0.0, y2p - y1p)
    return float(cx), float(cy), float(w), float(h)


def _map_box_from_crop(box_cxcywh_px: Tuple[float, float, float, float],
                       crop_xyxy: Tuple[float, float, float, float],
                       dst_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Map cxcywh in resized crop pixels back to original image xyxy.
    """
    cx, cy, w, h = box_cxcywh_px
    H, W = dst_size
    cx1, cy1, cx2, cy2 = crop_xyxy
    cw = max(1e-6, cx2 - cx1)
    ch = max(1e-6, cy2 - cy1)
    sx = cw / W
    sy = ch / H
    x1p = cx - 0.5 * w
    y1p = cy - 0.5 * h
    x2p = cx + 0.5 * w
    y2p = cy + 0.5 * h
    ox1 = cx1 + x1p * sx
    oy1 = cy1 + y1p * sy
    ox2 = cx1 + x2p * sx
    oy2 = cy1 + y2p * sy
    return float(ox1), float(oy1), float(ox2), float(oy2)


def _draw_box(frame_bgr: np.ndarray, xyxy: Tuple[float, float, float, float],
              color=(0, 255, 0), thickness: int = 2, text: Optional[str] = None) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(frame_bgr, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# Tracking core
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="T-SiamTPN tracking")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config")
    ap.add_argument("--video", type=str, required=True, help="Path to input video")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    ap.add_argument("--static_template", type=str, default=None, help="Optional static template image path")
    ap.add_argument("--init_bbox", type=str, default=None, help="Initial bbox 'x1,y1,x2,y2' in first frame")
    ap.add_argument("--save_video", type=str, default=None, help="Optional output video path")
    ap.add_argument("--max_frames", type=int, default=-1, help="Limit number of frames for quick tests")
    return ap.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get(cfg: Dict[str, Any], keys: List[str], default: Any) -> Any:
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


def get_device(cfg: Dict[str, Any]) -> torch.device:
    use_cuda = bool(_get(cfg, ["device", "use_cuda"], True))
    idx = int(_get(cfg, ["device", "cuda_index"], 0))
    if use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def build_model(cfg: Dict[str, Any], device: torch.device, checkpoint: Optional[str]) -> T_SiamTPN:
    model = T_SiamTPN.from_config(cfg)
    model.to(device)
    model.eval()
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    return model


def _find_object_image_near(video_path: str) -> Optional[str]:
    # Try dataset structure: .../samples/{video_id}/object_images/img_*.jpg
    video_dir = os.path.dirname(video_path)
    obj_dir = os.path.join(video_dir, "object_images")
    if os.path.isdir(obj_dir):
        imgs = sorted(glob.glob(os.path.join(obj_dir, "img_*.jpg")))
        if imgs:
            return imgs[0]
    return None


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = get_device(cfg)
    print(f"Device: {device}")

    # Sizes (C,H,W) -> HW tuples
    tpl_size = _get(cfg, ["templates", "crop_sizes", "template"], [3, 128, 128])
    sch_size = _get(cfg, ["templates", "crop_sizes", "search"], [3, 256, 256])
    template_hw = (int(tpl_size[1]), int(tpl_size[2]))
    search_hw = (int(sch_size[1]), int(sch_size[2]))
    mean = tuple(_get(cfg, ["data", "image_mean"], [0.485, 0.456, 0.406]))
    std = tuple(_get(cfg, ["data", "image_std"], [0.229, 0.224, 0.225]))
    dynamic_count = int(_get(cfg, ["templates", "dynamic_count"], 4))
    static_count = int(_get(cfg, ["templates", "static_count"], 1))
    max_templates = static_count + dynamic_count

    # Temporal params
    alpha = float(_get(cfg, ["templates", "temporal", "smoothing_alpha"], 0.8))
    cd_thresh = float(_get(cfg, ["templates", "temporal", "correction", "center_distance_thresh"], 0.25))
    sr_thresh = float(_get(cfg, ["templates", "temporal", "correction", "scale_ratio_thresh"], 0.25))
    upd_interval = int(_get(cfg, ["templates", "temporal", "update", "interval"], 5))
    conf_thresh = float(_get(cfg, ["templates", "temporal", "update", "confidence_thresh"], 0.5))
    update_enabled = bool(_get(cfg, ["templates", "temporal", "update", "enabled"], True))

    # Build model
    model = build_model(cfg, device, args.checkpoint)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        sys.exit(1)

    # Prepare static template
    static_tpl_path = args.static_template or _find_object_image_near(args.video)
    first_ok, first_bgr = cap.read()
    if not first_ok:
        print("Failed to read first frame")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind for processing loop
    first_rgb = _ensure_rgb(first_bgr)
    H0, W0 = first_rgb.shape[:2]

    init_bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    if args.init_bbox:
        vals = [float(v) for v in args.init_bbox.split(",")]
        if len(vals) != 4:
            print("--init_bbox must be 4 comma-separated numbers")
            sys.exit(1)
        init_bbox_xyxy = (vals[0], vals[1], vals[2], vals[3])

    if static_tpl_path and os.path.exists(static_tpl_path):
        tpl_bgr = cv2.imread(static_tpl_path, cv2.IMREAD_COLOR)
        if tpl_bgr is None:
            print(f"Failed to load static template: {static_tpl_path}")
            sys.exit(1)
        tpl_rgb = _ensure_rgb(tpl_bgr)
        static_tpl_rgb = tpl_rgb
    elif init_bbox_xyxy is not None:
        # Crop from first frame
        static_tpl_rgb = _crop_resize(first_rgb, init_bbox_xyxy, out_size=template_hw)
    else:
        print("Require either --static_template or --init_bbox (or dataset object_images).")
        sys.exit(1)

    # Dynamic templates store (as tensors normalized) and raw RGBs (for potential reuse)
    dynamic_tpl_tensors: List[torch.Tensor] = []
    # Initial temporal states (normalized coords [0,1])
    prev_smooth_box01: Optional[Tuple[float, float, float, float]] = None
    stable_box01: Optional[Tuple[float, float, float, float]] = None

    # Video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, float(_get(cfg, ["data", "sample_fps"], 30)), (W0, H0))
        if not writer.isOpened():
            print(f"Warning: cannot open writer for {args.save_video}")
            writer = None

    # Prepare normalized static template tensor
    static_tpl_t = _normalize(static_tpl_rgb, mean, std).unsqueeze(0).to(device)  # [1,3,Ht,Wt]

    # Tracking loop
    frame_idx = 0
    fps_hist: List[float] = []
    search_context_factor = 2.0  # aligned with dataset

    # Initialize bbox for frame 0
    if init_bbox_xyxy is None:
        # Fallback: small center box
        w = W0 * 0.1
        h = H0 * 0.1
        init_bbox_xyxy = (W0 * 0.45, H0 * 0.45, W0 * 0.45 + w, H0 * 0.45 + h)

    current_bbox_xyxy = init_bbox_xyxy

    print("Starting tracking...")
    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = _ensure_rgb(frame_bgr)
        Hf, Wf = frame_rgb.shape[:2]

        # Build search crop
        search_xyxy = _expand_bbox(current_bbox_xyxy, factor=search_context_factor, img_size=(Hf, Wf))
        search_patch_rgb = _crop_resize(frame_rgb, search_xyxy, out_size=search_hw)
        search_t = _normalize(search_patch_rgb, mean, std).unsqueeze(0).to(device)  # [1,3,Hs,Ws]

        # Assemble templates list for model: [static] + dynamic (each [1,3,Ht,Wt])
        templates_imgs: List[torch.Tensor] = [static_tpl_t]
        for dt in dynamic_tpl_tensors[:dynamic_count]:
            templates_imgs.append(dt.unsqueeze(0).to(device))

        # Forward
        t0 = time.time()
        with torch.no_grad():
            outputs = model(search_img=search_t, templates_imgs=templates_imgs)
            cls_list: List[torch.Tensor] = outputs["cls"]  # each [1,2,H,W]
            reg_list: List[torch.Tensor] = outputs["reg"]  # each [1,4,H,W] in [0,1]

            # Select best location across scales by target probability
            best_score = -1.0
            best_box01: Optional[Tuple[float, float, float, float]] = None

            for s in range(len(cls_list)):
                logits = cls_list[s]  # [1,2,H,W]
                prob = F.softmax(logits, dim=1)[0, 1]  # [H,W]
                score, idx = torch.max(prob.view(-1), dim=0)
                Hs, Ws = prob.shape
                iy = int(idx.item() // Ws)
                ix = int(idx.item() % Ws)
                # Gather reg at (iy, ix)
                reg = reg_list[s][0, :, iy, ix]  # [4], normalized [0,1] relative to search
                cx, cy, w, h = [float(v.item()) for v in reg]
                # Keep best by score
                if score.item() > best_score:
                    best_score = float(score.item())
                    best_box01 = (cx, cy, w, h)

        # If no prediction made (shouldn't happen), keep current box
        if best_box01 is None:
            pred_box01 = _xyxy_to_cxcywh(_map_box_to_crop(current_bbox_xyxy, search_xyxy, search_hw))
        else:
            pred_box01 = best_box01

        # Temporal smoothing in normalized space
        if prev_smooth_box01 is None:
            smooth_box01 = pred_box01
        else:
            psx, psy, psw, psh = prev_smooth_box01
            cx, cy, w, h = pred_box01
            smooth_box01 = (alpha * psx + (1 - alpha) * cx,
                            alpha * psy + (1 - alpha) * cy,
                            alpha * psw + (1 - alpha) * w,
                            alpha * psh + (1 - alpha) * h)

        # Correction: revert to stable if drift too large
        if stable_box01 is None:
            corrected_box01 = smooth_box01
            stable_box01 = smooth_box01
        else:
            # center distance in normalized units
            s_cx, s_cy, s_w, s_h = stable_box01
            c_cx, c_cy, c_w, c_h = smooth_box01
            center_dist = math.sqrt((s_cx - c_cx) ** 2 + (s_cy - c_cy) ** 2)
            scale_r = math.sqrt(max(1e-6, (c_w * c_h) / max(1e-6, s_w * s_h)))
            if (center_dist > cd_thresh) or (abs(scale_r - 1.0) > sr_thresh):
                corrected_box01 = stable_box01
            else:
                corrected_box01 = smooth_box01

        # Map corrected normalized box (in search crop) to original frame coords
        # First convert normalized to search pixels
        cx, cy, w, h = corrected_box01
        box_px = (cx * search_hw[1], cy * search_hw[0], w * search_hw[1], h * search_hw[0])
        pred_xyxy = _cxcywh_to_xyxy(box_px)
        orig_xyxy = _map_box_from_crop((box_px[0], box_px[1], box_px[2], box_px[3]), search_xyxy, search_hw)
        # Note: _map_box_from_crop expects cxcywh in crop pixels; we passed correct tuple already

        # Update temporal states
        prev_smooth_box01 = smooth_box01
        # Update stable as the corrected (sticky) one for next correction
        stable_box01 = corrected_box01
        current_bbox_xyxy = orig_xyxy

        # Dynamic template update
        if update_enabled and (frame_idx % max(1, upd_interval) == 0) and (best_score >= conf_thresh):
            # Crop new template from current frame
            new_tpl_rgb = _crop_resize(frame_rgb, current_bbox_xyxy, out_size=template_hw)
            new_tpl_t = _normalize(new_tpl_rgb, mean, std)
            # Maintain capacity
            if len(dynamic_tpl_tensors) < dynamic_count:
                dynamic_tpl_tensors.append(new_tpl_t)
            else:
                # Drop oldest
                if dynamic_tpl_tensors:
                    dynamic_tpl_tensors.pop(0)
                dynamic_tpl_tensors.append(new_tpl_t)

        # Draw and show/save
        vis_bgr = frame_bgr.copy()
        _draw_box(vis_bgr, current_bbox_xyxy, color=(0, 255, 0), thickness=2, text=f"{best_score:.2f}")
        t1 = time.time()
        fps = 1.0 / max(1e-6, (t1 - t0))
        fps_hist.append(fps)
        cv2.putText(vis_bgr, f"FPS {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(vis_bgr)

        cv2.imshow("T-SiamTPN Tracking", vis_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if fps_hist:
        print(f"Average FPS: {sum(fps_hist)/len(fps_hist):.2f} | Frames: {len(fps_hist)}")


if __name__ == "__main__":
    main()