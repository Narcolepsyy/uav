#!/usr/bin/env python3
"""
Headless benchmark for T-SiamTPN on a folder of videos (e.g., public_test/samples).
It loads the model, iterates each video, runs tracking without display, and writes a CSV report.
Usage:
  python tools/benchmark.py --videos_root public_test/samples --checkpoint runs/checkpoint_epoch_1.pth --report reports/public_test_benchmark.csv --max_frames 300
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import os
import sys
import time
import glob
import argparse
import csv

import yaml
import numpy as np
import cv2
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(os.path.dirname(PROJECT_ROOT))

from src.models.t_siamtpn import T_SiamTPN  # type: ignore

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

def _map_box_from_crop(box_cxcywh_px: Tuple[float, float, float, float],
                       crop_xyxy: Tuple[float, float, float, float],
                       dst_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
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

def load_config(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _get(cfg: Dict[str, object], keys: List[str], default):
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

def get_device(cfg: Dict[str, object]) -> torch.device:
    use_cuda = bool(_get(cfg, ["device", "use_cuda"], True))
    idx = int(_get(cfg, ["device", "cuda_index"], 0))
    if use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")

def build_model(cfg: Dict[str, object], device: torch.device, checkpoint: Optional[str]) -> T_SiamTPN:
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
    video_dir = os.path.dirname(video_path)
    obj_dir = os.path.join(video_dir, "object_images")
    if os.path.isdir(obj_dir):
        imgs = sorted(glob.glob(os.path.join(obj_dir, "img_*.jpg")))
        if imgs:
            return imgs[0]
    return None

def benchmark_video(model: T_SiamTPN,
                    cfg: Dict[str, object],
                    device: torch.device,
                    video_path: str,
                    checkpoint: Optional[str],
                    max_frames: int = -1,
                    save_vis_dir: Optional[str] = None) -> Dict[str, object]:
    tpl_size = _get(cfg, ["templates", "crop_sizes", "template"], [3, 128, 128])
    sch_size = _get(cfg, ["templates", "crop_sizes", "search"], [3, 256, 256])
    template_hw = (int(tpl_size[1]), int(tpl_size[2]))
    search_hw = (int(sch_size[1]), int(sch_size[2]))
    mean = tuple(_get(cfg, ["data", "image_mean"], [0.485, 0.456, 0.406]))
    std = tuple(_get(cfg, ["data", "image_std"], [0.229, 0.224, 0.225]))
    dynamic_count = int(_get(cfg, ["templates", "dynamic_count"], 4))
    static_count = int(_get(cfg, ["templates", "static_count"], 1))
    max_templates = static_count + dynamic_count
    alpha = float(_get(cfg, ["templates", "temporal", "smoothing_alpha"], 0.8))
    cd_thresh = float(_get(cfg, ["templates", "temporal", "correction", "center_distance_thresh"], 0.25))
    sr_thresh = float(_get(cfg, ["templates", "temporal", "correction", "scale_ratio_thresh"], 0.25))
    upd_interval = int(_get(cfg, ["templates", "temporal", "update", "interval"], 5))
    conf_thresh = float(_get(cfg, ["templates", "temporal", "update", "confidence_thresh"], 0.5))
    update_enabled = bool(_get(cfg, ["templates", "temporal", "update", "enabled"], True))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    ok, first_bgr = cap.read()
    if not ok or first_bgr is None:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_rgb = _ensure_rgb(first_bgr)
    H0, W0 = first_rgb.shape[:2]

    static_tpl_path = _find_object_image_near(video_path)
    if static_tpl_path and os.path.exists(static_tpl_path):
        tpl_bgr = cv2.imread(static_tpl_path, cv2.IMREAD_COLOR)
        if tpl_bgr is None:
            cap.release()
            raise RuntimeError(f"Failed to load static template: {static_tpl_path}")
        static_tpl_rgb = _ensure_rgb(tpl_bgr)
    else:
        # Fallback: center crop of first frame
        w = W0 * 0.1
        h = H0 * 0.1
        init_bbox_xyxy = (W0 * 0.45, H0 * 0.45, W0 * 0.45 + w, H0 * 0.45 + h)
        static_tpl_rgb = _crop_resize(first_rgb, init_bbox_xyxy, out_size=template_hw)

    static_tpl_t = _normalize(static_tpl_rgb, mean, std).unsqueeze(0).to(device)
    dynamic_tpl_tensors: List[torch.Tensor] = []

    prev_smooth_box01: Optional[Tuple[float, float, float, float]] = None
    stable_box01: Optional[Tuple[float, float, float, float]] = None

    fps_hist: List[float] = []
    best_score_hist: List[float] = []
    search_context_factor = 2.0

    # Initialize bbox to center for first iteration
    w = W0 * 0.1
    h = H0 * 0.1
    current_bbox_xyxy = (W0 * 0.45, H0 * 0.45, W0 * 0.45 + w, H0 * 0.45 + h)

    # Optional video writer
    writer = None
    if save_vis_dir:
        os.makedirs(save_vis_dir, exist_ok=True)
        out_path = os.path.join(save_vis_dir, os.path.basename(os.path.dirname(video_path)) + "_out.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(_get(cfg, ["data", "sample_fps"], 30)), (W0, H0))
        if not writer.isOpened():
            writer = None

    frame_idx = 0
    while True:
        if max_frames > 0 and frame_idx >= max_frames:
            break
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = _ensure_rgb(frame_bgr)
        Hf, Wf = frame_rgb.shape[:2]

        # Build search crop
        search_xyxy = _expand_bbox(current_bbox_xyxy, factor=search_context_factor, img_size=(Hf, Wf))
        search_patch_rgb = _crop_resize(frame_rgb, search_xyxy, out_size=search_hw)
        search_t = _normalize(search_patch_rgb, mean, std).unsqueeze(0).to(device)

        # Assemble templates
        templates_imgs: List[torch.Tensor] = [static_tpl_t]
        for dt in dynamic_tpl_tensors[:dynamic_count]:
            templates_imgs.append(dt.unsqueeze(0).to(device))

        t0 = time.time()
        with torch.no_grad():
            outputs = model(search_img=search_t, templates_imgs=templates_imgs)
            cls_list: List[torch.Tensor] = outputs["cls"]
            reg_list: List[torch.Tensor] = outputs["reg"]

            best_score = -1.0
            best_box01: Optional[Tuple[float, float, float, float]] = None
            for s in range(len(cls_list)):
                logits = cls_list[s]
                prob = F.softmax(logits, dim=1)[0, 1]
                score, idx = torch.max(prob.view(-1), dim=0)
                Hs, Ws = prob.shape
                iy = int(idx.item() // Ws)
                ix = int(idx.item() % Ws)
                reg = reg_list[s][0, :, iy, ix]
                cx, cy, w1, h1 = [float(v.item()) for v in reg]
                if score.item() > best_score:
                    best_score = float(score.item())
                    best_box01 = (cx, cy, w1, h1)

        if best_box01 is None:
            # keep current box projected to search
            cx, cy, w1, h1 = (0.5*search_hw[1], 0.5*search_hw[0], 0.2*search_hw[1], 0.2*search_hw[0])
            pred_box01 = (cx / search_hw[1], cy / search_hw[0], w1 / search_hw[1], h1 / search_hw[0])
        else:
            pred_box01 = best_box01

        # Temporal smoothing
        if prev_smooth_box01 is None:
            smooth_box01 = pred_box01
        else:
            psx, psy, psw, psh = prev_smooth_box01
            cx, cy, w1, h1 = pred_box01
            smooth_box01 = (alpha * psx + (1 - alpha) * cx,
                            alpha * psy + (1 - alpha) * cy,
                            alpha * psw + (1 - alpha) * w1,
                            alpha * psh + (1 - alpha) * h1)

        # Correction
        if stable_box01 is None:
            corrected_box01 = smooth_box01
            stable_box01 = smooth_box01
        else:
            s_cx, s_cy, s_w, s_h = stable_box01
            c_cx, c_cy, c_w, c_h = smooth_box01
            center_dist = float(np.sqrt((s_cx - c_cx) ** 2 + (s_cy - c_cy) ** 2))
            scale_r = float(np.sqrt(max(1e-6, (c_w * c_h) / max(1e-6, s_w * s_h))))
            if (center_dist > cd_thresh) or (abs(scale_r - 1.0) > sr_thresh):
                corrected_box01 = stable_box01
            else:
                corrected_box01 = smooth_box01

        cx, cy, w1, h1 = corrected_box01
        box_px = (cx * search_hw[1], cy * search_hw[0], w1 * search_hw[1], h1 * search_hw[0])
        orig_xyxy = _map_box_from_crop((box_px[0], box_px[1], box_px[2], box_px[3]), search_xyxy, search_hw)
        current_bbox_xyxy = orig_xyxy

        t1 = time.time()
        fps = 1.0 / max(1e-6, (t1 - t0))
        fps_hist.append(fps)
        best_score_hist.append(float(best_score))

        if writer is not None:
            vis = frame_bgr.copy()
            x1, y1, x2, y2 = map(int, orig_xyxy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{fps:.2f} FPS | {best_score:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2, cv2.LINE_AA)
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    avg_fps = float(np.mean(fps_hist)) if fps_hist else 0.0
    avg_score = float(np.mean(best_score_hist)) if best_score_hist else 0.0
    return {
        "video": video_path,
        "frames": len(fps_hist),
        "avg_fps": avg_fps,
        "avg_score": avg_score,
    }

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Headless benchmark for T-SiamTPN")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--videos_root", type=str, default="public_test/samples", help="Root folder containing */drone_video.mp4")
    ap.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (.pth)")
    ap.add_argument("--report", type=str, default="reports/public_test_benchmark.csv", help="CSV report path")
    ap.add_argument("--max_frames", type=int, default=-1, help="Limit frames per video")
    ap.add_argument("--save_vis_dir", type=str, default=None, help="Optional directory to save visualization videos")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.report), exist_ok=True) if os.path.dirname(args.report) else None
    cfg = load_config(args.config)
    device = get_device(cfg)
    print(f"Device: {device}")

    model = build_model(cfg, device, args.checkpoint)

    pattern = os.path.join(args.videos_root, "*", "drone_video.mp4")
    videos = sorted(glob.glob(pattern))
    if not videos:
        print(f"No videos found at {pattern}")
        sys.exit(1)

    rows: List[Dict[str, object]] = []
    for vp in videos:
        print(f"Benchmarking {vp} ...")
        res = benchmark_video(model, cfg, device, vp, args.checkpoint, max_frames=args.max_frames, save_vis_dir=args.save_vis_dir)
        rows.append(res)
        print(f"  -> frames={res['frames']} avg_fps={res['avg_fps']:.2f} avg_score={res['avg_score']:.3f}")

    # Write CSV
    fieldnames = ["video", "frames", "avg_fps", "avg_score"]
    with open(args.report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote report: {args.report}")

if __name__ == "__main__":
    main()