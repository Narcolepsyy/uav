#!/usr/bin/env python3
"""
Training script for T-SiamTPN on the Observing UAV dataset.

Features:
- Loads config (YAML) from configs/default.yaml or provided --config
- Builds dataset via src/data/uav_dataset.py (UAVDataset.from_config)
- Builds model via src/models/t_siamtpn.py (T_SiamTPN.from_config)
- Loss aggregation via src/losses/losses.py (LossAggregator.from_config)
- Mixed precision training (CUDA amp) if enabled in config
- Cosine LR scheduler with optional warmup
- Periodic checkpoint saving and logging

Notes for embedded (Jetson Nano):
- Adjust batch_size and num_workers in configs/default.yaml if memory constrained
- Ensure torch/torchvision are installed via NVIDIA JetPack
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import os
import sys
import time
import argparse
import math

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Make sure we can import project modules when running as a script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.uav_dataset import UAVDataset  # type: ignore
from src.models.t_siamtpn import T_SiamTPN  # type: ignore
from src.losses.losses import from_config as LossesFromConfig  # type: ignore
from src.utils.logger import set_seed, log_system_info  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train T-SiamTPN on Observing UAV dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir in config.training.output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.training.epochs")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg: Dict[str, Any]) -> torch.device:
    use_cuda = bool(_get(cfg, ["device", "use_cuda"], True))
    cuda_index = int(_get(cfg, ["device", "cuda_index"], 0))
    if use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


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


def collate_batch(samples: List[Dict[str, Any]], max_templates: int, template_size: Tuple[int, int]) -> Dict[str, Any]:
    """
    Collate function:
    - Stacks search images -> [B,3,Hs,Ws]
    - Pads/aligns templates list to length max_templates (1 static + dynamic_count) across batch,
      producing a list of length L where each element is [B,3,Ht,Wt]
    - Stacks target boxes [B,4] in pixel coords of search
    """
    B = len(samples)
    # Search images
    search_imgs = torch.stack([s["search_img"] for s in samples], dim=0)  # [B,3,Hs,Ws]
    # Target boxes
    targets = torch.stack([s["target_box_cxcywh"].float() for s in samples], dim=0)  # [B,4]

    # Templates
    Ht, Wt = template_size
    # Determine device later; create zeros on CPU and move later
    def zero_tpl() -> torch.Tensor:
        return torch.zeros((3, Ht, Wt), dtype=torch.float32)

    # Prepare per-slot stacks
    templates_slots: List[torch.Tensor] = []
    for slot in range(max_templates):
        per_slot = []
        for s in samples:
            tpls: List[torch.Tensor] = s["templates"]
            if slot < len(tpls):
                per_slot.append(tpls[slot])
            else:
                per_slot.append(zero_tpl())
        templates_slots.append(torch.stack(per_slot, dim=0))  # [B,3,Ht,Wt]

    meta = [s["meta"] for s in samples]
    return {
        "search_img": search_imgs,
        "templates_imgs": templates_slots,
        "target_box_px": targets,
        "meta": meta,
    }


def build_dataloader(cfg: Dict[str, Any]) -> Tuple[DataLoader, int, Tuple[int, int], Tuple[int, int]]:
    ds = UAVDataset.from_config(cfg)
    batch_size = int(_get(cfg, ["training", "batch_size"], 16))
    num_workers = int(_get(cfg, ["training", "num_workers"], 4))
    static_count = int(_get(cfg, ["templates", "static_count"], 1))
    dynamic_count = int(_get(cfg, ["templates", "dynamic_count"], 4))
    max_templates = static_count + dynamic_count

    tpl_size_list = _get(cfg, ["templates", "crop_sizes", "template"], [3, 128, 128])
    sch_size_list = _get(cfg, ["templates", "crop_sizes", "search"], [3, 256, 256])
    template_size = (int(tpl_size_list[1]), int(tpl_size_list[2]))  # (H,W)
    search_size = (int(sch_size_list[1]), int(sch_size_list[2]))    # (H,W)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, max_templates=max_templates, template_size=template_size),
        drop_last=False,
    )
    return loader, max_templates, template_size, search_size


def build_model(cfg: Dict[str, Any], device: torch.device) -> T_SiamTPN:
    model = T_SiamTPN.from_config(cfg)
    model.to(device)
    return model


def build_losses(cfg: Dict[str, Any]) -> nn.Module:
    agg = LossesFromConfig(cfg)
    return agg


def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> Tuple[optim.Optimizer, Any]:
    lr = float(_get(cfg, ["training", "learning_rate"], 1e-3))
    wd = float(_get(cfg, ["training", "weight_decay"], 1e-4))
    opt_name = str(_get(cfg, ["training", "optimizer"], "adamw")).lower()
    if opt_name == "adamw":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        momentum = 0.9
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
    else:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Cosine scheduler
    epochs = int(_get(cfg, ["training", "epochs"], 50))
    scheduler_name = str(_get(cfg, ["training", "lr_scheduler"], "cosine")).lower()
    if scheduler_name == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)
    return opt, sched


def train_one_epoch(epoch: int,
                    model: T_SiamTPN,
                    loader: DataLoader,
                    losses: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    search_size: Tuple[int, int],
                    cfg: Dict[str, Any],
                    scaler: torch.amp.GradScaler | None = None) -> Dict[str, float]:
    model.train()
    log_interval = int(_get(cfg, ["training", "log_interval"], 50))
    grad_clip_norm = float(_get(cfg, ["training", "grad_clip_norm"], 5.0))
    use_amp = bool(_get(cfg, ["device", "mixed_precision"], True)) and device.type == "cuda"

    running = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_iou": 0.0}
    count = 0

    for i, batch in enumerate(loader, start=1):
        search_img = batch["search_img"].to(device, non_blocking=True)       # [B,3,Hs,Ws]
        templates_imgs = [t.to(device, non_blocking=True) for t in batch["templates_imgs"]]  # list of [B,3,Ht,Wt]
        target_box_px = batch["target_box_px"].to(device)                    # [B,4]

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(search_img=search_img, templates_imgs=templates_imgs)
                loss_dict = losses(outputs, target_box_px, search_size)
                loss = loss_dict["loss_total"]
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(search_img=search_img, templates_imgs=templates_imgs)
            loss_dict = losses(outputs, target_box_px, search_size)
            loss = loss_dict["loss_total"]
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        # Accumulate running metrics
        for k in running.keys():
            running[k] += float(loss_dict[k].item())
        count += 1

        if i % log_interval == 0:
            avg = {k: running[k] / max(count, 1) for k in running.keys()}
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} Iter {i}/{len(loader)} LR {lr:.6f} "
                  f"Loss {avg['loss_total']:.4f} (cls {avg['loss_cls']:.4f}, l1 {avg['loss_l1']:.4f}, iou {avg['loss_iou']:.4f})")

    return {k: running[k] / max(count, 1) for k in running.keys()}


def save_checkpoint(output_dir: str,
                    epoch: int,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    cfg: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
    }
    path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Overrides
    if args.output_dir is not None:
        cfg.setdefault("training", {}).setdefault("output_dir", args.output_dir)
    if args.epochs is not None:
        cfg.setdefault("training", {}).setdefault("epochs", args.epochs)

    # Seed and system info
    seed = int(_get(cfg, ["training", "seed"], 42))
    set_seed(seed)
    log_system_info()

    device = get_device(cfg)
    print(f"Using device: {device}")

    # Build components
    loader, max_templates, template_size, search_size = build_dataloader(cfg)
    model = build_model(cfg, device)
    losses = build_losses(cfg)
    optimizer, scheduler = build_optimizer(cfg, model)

    epochs = int(_get(cfg, ["training", "epochs"], 50))
    output_dir = str(_get(cfg, ["training", "output_dir"], "runs"))
    save_interval = int(_get(cfg, ["training", "save_interval_epochs"], 5))

    use_amp = bool(_get(cfg, ["device", "mixed_precision"], True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    print(f"Max templates per sample: {max_templates} | Template size: {template_size} | Search size: {search_size}")

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_metrics = train_one_epoch(epoch, model, loader, losses, optimizer, device, search_size, cfg, scaler)
        print(f"Epoch {epoch} Summary: Loss {epoch_metrics['loss_total']:.4f} "
              f"(cls {epoch_metrics['loss_cls']:.4f}, l1 {epoch_metrics['loss_l1']:.4f}, iou {epoch_metrics['loss_iou']:.4f})")

        # Step scheduler after epoch
        try:
            scheduler.step()
        except Exception:
            pass

        if epoch == 1 or (epoch % save_interval) == 0 or epoch == epochs:
            save_checkpoint(output_dir, epoch, model, optimizer, cfg)

    print("Training complete.")


if __name__ == "__main__":
    main()