"""
UAV Observing Dataset for T-SiamTPN (Temporal Siamese Transformer Pyramid Networks)

This dataset parses:
- videos in observing/train/samples/{video_id}/drone_video.mp4
- object images in observing/train/samples/{video_id}/object_images/img_*.jpg
- annotations in observing/train/annotations/annotations.json

For each annotated frame, it returns:
- search_img: Tensor [3, Hs, Ws] normalized (ImageNet stats) per config
- templates: List[Tensor] with first static template and up to dynamic_count dynamic templates
- target_box_cxcywh: Tensor [4] with bbox in search crop coordinates (pixels)
- meta: dict containing video_id, frame_idx

Cropping strategy:
- Template crop: tight bbox crop resized to template size
- Search crop: context-enlarged bbox (factor=2.0) resized to search size
This is lightweight and suitable for embedded training and validation.

Note:
- Dynamic templates are generated from up to 4 previous annotated frames of the same video
- Static template is taken from object_images/img_1.jpg (fallback to img_2.jpg or img_3.jpg if missing)
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import os
import json
import glob

import numpy as np

import cv2  # Prefer system OpenCV on Jetson Nano
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB and ensure contiguous float32 in [0,1]."""
    if img_bgr is None:
        raise RuntimeError("Received None image from OpenCV reader")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return img_rgb


def _normalize(img_rgb: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    """Normalize numpy RGB HxWx3 to torch [3,H,W] using mean/std."""
    img_chw = np.transpose(img_rgb, (2, 0, 1))  # [3,H,W]
    tensor = torch.from_numpy(img_chw)  # float32
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def _read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a specific frame from a video using OpenCV random access."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    # Random seek; note: some codecs perform approximate seeks
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def _load_image(path: str) -> np.ndarray:
    """Load image via OpenCV, returning BGR array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def _crop_resize(img_rgb: np.ndarray, bbox_xyxy: Tuple[float, float, float, float], out_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop region [x1,y1,x2,y2] from img_rgb and resize to out_size (H,W).
    Out-of-bounds are handled by clamping the crop.
    """
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w - 1, int(np.floor(x1))))
    y1 = max(0, min(h - 1, int(np.floor(y1))))
    x2 = max(0, min(w - 1, int(np.ceil(x2))))
    y2 = max(0, min(h - 1, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        # Fallback: center crop full image
        cx = w // 2
        cy = h // 2
        half_w = min(w, h) // 4
        x1 = max(0, cx - half_w)
        x2 = min(w - 1, cx + half_w)
        y1 = max(0, cy - half_w)
        y2 = min(h - 1, cy + half_w)
    patch = img_rgb[y1:y2, x1:x2]
    patch = cv2.resize(patch, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
    return patch


def _expand_bbox(xyxy: Tuple[float, float, float, float], factor: float, img_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """Expand bbox by factor around center; clamp to image bounds."""
    x1, y1, x2, y2 = xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1)
    h = (y2 - y1)
    new_w = w * factor
    new_h = h * factor
    nx1 = cx - 0.5 * new_w
    ny1 = cy - 0.5 * new_h
    nx2 = cx + 0.5 * new_w
    ny2 = cy + 0.5 * new_h
    H, W = img_size
    nx1 = max(0, min(W - 1, nx1))
    ny1 = max(0, min(H - 1, ny1))
    nx2 = max(0, min(W - 1, nx2))
    ny2 = max(0, min(H - 1, ny2))
    return float(nx1), float(ny1), float(nx2), float(ny2)


def _map_box_to_crop(orig_xyxy: Tuple[float, float, float, float],
                     crop_xyxy: Tuple[float, float, float, float],
                     dst_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Map original box coordinates (xyxy) into the crop coordinate space resized to dst_size (H,W).
    Returns bbox in cxcywh (pixels).
    """
    ox1, oy1, ox2, oy2 = orig_xyxy
    cx1, cy1, cx2, cy2 = crop_xyxy
    crop_w = max(1.0, cx2 - cx1)
    crop_h = max(1.0, cy2 - cy1)

    # Normalize inside crop [0,1]
    nx1 = (ox1 - cx1) / crop_w
    ny1 = (oy1 - cy1) / crop_h
    nx2 = (ox2 - cx1) / crop_w
    ny2 = (oy2 - cy1) / crop_h

    # Scale to dst_size pixels
    H, W = dst_size
    sx = W
    sy = H
    x1p = nx1 * sx
    y1p = ny1 * sy
    x2p = nx2 * sx
    y2p = ny2 * sy

    cx = 0.5 * (x1p + x2p)
    cy = 0.5 * (y1p + y2p)
    w = max(0.0, x2p - x1p)
    h = max(0.0, y2p - y1p)
    return float(cx), float(cy), float(w), float(h)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class UAVDataset(Dataset):
    """
    Observing UAV dataset providing pairs of templates and search frames for T-SiamTPN.

    Config keys used:
      data.train_root
      data.annotations
      data.image_mean
      data.image_std
      data.template_selection
      templates.crop_sizes.template
      templates.crop_sizes.search
      templates.dynamic_count

    Augment:
      data.augment.flip, data.augment.jitter, data.augment.gray (currently light support)
    """

    def __init__(self,
                 train_root: str,
                 annotations_path: str,
                 image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 template_size: Tuple[int, int] = (128, 128),  # (H,W)
                 search_size: Tuple[int, int] = (256, 256),    # (H,W)
                 dynamic_count: int = 4,
                 template_selection: str = "static+rolling",
                 augment_cfg: Optional[Dict[str, bool]] = None,
                 search_context_factor: float = 2.0,
                 ):
        super().__init__()
        self.train_root = train_root
        self.annotations_path = annotations_path
        self.mean = image_mean
        self.std = image_std
        self.template_size = template_size
        self.search_size = search_size
        self.dynamic_count = max(0, int(dynamic_count))
        self.template_selection = template_selection
        self.augment_cfg = augment_cfg or {"flip": True, "jitter": True, "gray": False}
        self.search_context_factor = float(search_context_factor)

        self.samples: List[Dict[str, object]] = []
        self._video_to_annos: Dict[str, List[Dict[str, int]]] = {}

        self._load_annotations()
        self._build_index()

    # ----------------------
    # Annotation parsing
    # ----------------------
    def _load_annotations(self):
        """Parse annotations JSON to map video_id -> list of bbox dicts."""
        with open(self.annotations_path, "r") as f:
            data = json.load(f)
        # Expected structure: list of { "video_id": ..., "annotations": [ { "bboxes": [ { frame,x1,y1,x2,y2 }, ... ] }, ... ] }
        for entry in data:
            vid = entry.get("video_id")
            annos = entry.get("annotations", [])
            if vid is None:
                continue
            bboxes_all: List[Dict[str, int]] = []
            for seg in annos:
                bbs = seg.get("bboxes", [])
                for b in bbs:
                    if {"frame", "x1", "y1", "x2", "y2"}.issubset(b.keys()):
                        bboxes_all.append({"frame": int(b["frame"]),
                                           "x1": int(b["x1"]), "y1": int(b["y1"]),
                                           "x2": int(b["x2"]), "y2": int(b["y2"])})
            # Sort by frame for rolling dynamic template selection
            bboxes_all.sort(key=lambda d: d["frame"])
            self._video_to_annos[vid] = bboxes_all

    def _build_index(self):
        """Build flat samples list with video path, object images, and per-frame bbox."""
        for vid, bboxes in self._video_to_annos.items():
            video_dir = os.path.join(self.train_root, "samples", vid)
            video_path = os.path.join(video_dir, "drone_video.mp4")
            obj_dir = os.path.join(video_dir, "object_images")
            obj_imgs = sorted(glob.glob(os.path.join(obj_dir, "img_*.jpg")))
            self.samples.extend([
                {
                    "video_id": vid,
                    "video_path": video_path,
                    "object_images": obj_imgs,
                    "frame_idx": bb["frame"],
                    "bbox_xyxy": (float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])),
                    # store index within list for dynamic template lookup
                    "anno_index": i,
                }
                for i, bb in enumerate(bboxes)
            ])

    # ----------------------
    # Dataset API
    # ----------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        vid = s["video_id"]
        video_path = s["video_path"]
        frame_idx = int(s["frame_idx"])
        bbox_xyxy = tuple(s["bbox_xyxy"])
        anno_index = int(s["anno_index"])
        obj_imgs: List[str] = s["object_images"]

        # Read search frame
        frame_bgr = _read_frame(video_path, frame_idx)
        frame_rgb = _ensure_rgb(frame_bgr)

        # Build search crop with context
        H, W = frame_rgb.shape[:2]
        search_crop_xyxy = _expand_bbox(bbox_xyxy, factor=self.search_context_factor, img_size=(H, W))
        search_patch_rgb = _crop_resize(frame_rgb, search_crop_xyxy, out_size=self.search_size)
        target_cxcywh = _map_box_to_crop(bbox_xyxy, search_crop_xyxy, dst_size=self.search_size)

        # Static template from object_images (fallback chain)
        static_tpl_bgr = None
        for p in obj_imgs[:3]:  # try first 3
            try:
                static_tpl_bgr = _load_image(p)
                break
            except FileNotFoundError:
                continue
        if static_tpl_bgr is None:
            # fallback: use the search patch itself (rare)
            static_tpl_bgr = frame_bgr
        static_tpl_rgb = _ensure_rgb(static_tpl_bgr)

        # Template crop: tight bbox crop (from same current frame) resized to template size
        template_crop_rgb = _crop_resize(frame_rgb, bbox_xyxy, out_size=self.template_size)

        # Prefer object image as static if present; else use bbox crop
        static_rgb = static_tpl_rgb if len(obj_imgs) > 0 else template_crop_rgb
        # Ensure static template is resized to the configured template size (H,W) for batch collation
        static_rgb_resized = cv2.resize(static_rgb, (self.template_size[1], self.template_size[0]), interpolation=cv2.INTER_LINEAR)
        static_norm = _normalize(static_rgb_resized, self.mean, self.std)

        # Dynamic templates: take up to dynamic_count previous annotated frames for this video
        dynamic_norm_list: List[torch.Tensor] = []
        if self.dynamic_count > 0 and self.template_selection.startswith("static"):
            prev_annos = self._video_to_annos.get(vid, [])
            # Collect max dynamic_count previous entries
            start = max(0, anno_index - self.dynamic_count)
            for j in range(start, anno_index):
                pb = prev_annos[j]
                try:
                    prev_rgb = _ensure_rgb(_read_frame(video_path, int(pb["frame"])))
                    prev_tpl_rgb = _crop_resize(prev_rgb,
                                               (float(pb["x1"]), float(pb["y1"]), float(pb["x2"]), float(pb["y2"])),
                                               out_size=self.template_size)
                    dynamic_norm_list.append(_normalize(prev_tpl_rgb, self.mean, self.std))
                except Exception:
                    # Skip failures (codec/seek)
                    continue

        # Normalize search
        search_norm = _normalize(search_patch_rgb, self.mean, self.std)

        # Light augmentation (applied to search only for simplicity)
        if self.augment_cfg.get("flip", False):
            if np.random.rand() < 0.5:
                search_norm = torch.flip(search_norm, dims=[2])  # horizontal flip W-axis
                # Adjust target cx after flip
                target_cxcywh = (float(self.search_size[1]) - target_cxcywh[0], target_cxcywh[1], target_cxcywh[2], target_cxcywh[3])

        if self.augment_cfg.get("jitter", False):
            # Color jitter: brightness/contrast light perturbation
            if np.random.rand() < 0.3:
                delta = (np.random.rand(1, 1, 1).astype(np.float32) - 0.5) * 0.1
                search_norm = search_norm + torch.from_numpy(delta).to(search_norm.dtype)
                search_norm = torch.clamp(search_norm, min=(-torch.tensor(self.mean)/torch.tensor(self.std)).min().item(),
                                          max=((1.0 - torch.tensor(self.mean))/torch.tensor(self.std)).max().item())

        if self.augment_cfg.get("gray", False):
            if np.random.rand() < 0.1:
                # Convert to grayscale and back: average channels
                g = search_norm.mean(dim=0, keepdim=True)
                search_norm = g.repeat(3, 1, 1)

        # Assemble templates: first static then dynamics (may be empty)
        templates_list: List[torch.Tensor] = [static_norm] + dynamic_norm_list

        out = {
            "search_img": search_norm,                              # [3,Hs,Ws]
            "templates": templates_list,                            # List[[3,Ht,Wt]]
            "target_box_cxcywh": torch.tensor(target_cxcywh),       # [4] in pixels of search crop
            "meta": {
                "video_id": vid,
                "frame_idx": frame_idx,
                "video_path": video_path,
                "anno_index": anno_index,
            }
        }
        return out

    # ----------------------
    # Config constructor
    # ----------------------
    @staticmethod
    def from_config(cfg: Dict[str, object]) -> "UAVDataset":
        """Build dataset from nested/flat dict config."""
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

        train_root = str(_get(["data", "train_root"], "observing/train"))
        annotations = str(_get(["data", "annotations"], "observing/train/annotations/annotations.json"))
        mean = tuple(_get(["data", "image_mean"], [0.485, 0.456, 0.406]))
        std = tuple(_get(["data", "image_std"], [0.229, 0.224, 0.225]))
        tpl_size = _get(["templates", "crop_sizes", "template"], [3, 128, 128])
        sch_size = _get(["templates", "crop_sizes", "search"], [3, 256, 256])
        dynamic_count = int(_get(["templates", "dynamic_count"], 4))
        template_selection = str(_get(["data", "template_selection"], "static+rolling"))
        augment = _get(["data", "augment"], {"flip": True, "jitter": True, "gray": False})

        # sizes are [C,H,W]
        template_hw = (int(tpl_size[1]), int(tpl_size[2]))
        search_hw = (int(sch_size[1]), int(sch_size[2]))

        return UAVDataset(
            train_root=train_root,
            annotations_path=annotations,
            image_mean=tuple(mean),
            image_std=tuple(std),
            template_size=template_hw,
            search_size=search_hw,
            dynamic_count=dynamic_count,
            template_selection=template_selection,
            augment_cfg=augment,
            search_context_factor=2.0,
        )


# -----------------------------------------------------------------------------
# Simple self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal smoke test: build dataset and fetch one sample
    import yaml
    cfg_path = "configs/default.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            "data": {
                "train_root": "observing/train",
                "annotations": "observing/train/annotations/annotations.json",
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "template_selection": "static+rolling",
                "augment": {"flip": True, "jitter": True, "gray": False},
            },
            "templates": {
                "crop_sizes": {
                    "template": [3, 128, 128],
                    "search": [3, 256, 256],
                },
                "dynamic_count": 4,
            }
        }

    ds = UAVDataset.from_config(cfg)
    print(f"Dataset len={len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print("search_img:", tuple(sample["search_img"].shape))
        print("templates count:", len(sample["templates"]))
        print("target_box_cxcywh:", sample["target_box_cxcywh"])
        print("meta:", sample["meta"])