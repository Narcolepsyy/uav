"""
Lightweight logging utilities for T-SiamTPN.

Features:
- Consistent logger configuration for training/inference
- Timed scopes for measuring code blocks
- FPS meter for throughput reporting
- Resource profiling (CPU/RAM/GPU if available)

This module has minimal dependencies and is embedded-friendly.
"""

from __future__ import annotations
import logging
import sys
import time
import os
import random
import datetime
from typing import Optional, Dict, Any

import psutil

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

def configure_logger(name: str = "T-SiamTPN",
                     level: str = "INFO",
                     to_stdout: bool = True) -> logging.Logger:
    """
    Create and configure a logger with standard format.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    if isinstance(level, str):
        logger.setLevel(LEVELS.get(level.upper(), logging.INFO))
    else:
        logger.setLevel(level)

    if to_stdout:
        has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        if not has_stream:
            sh = logging.StreamHandler(sys.stdout)
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            sh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(sh)
    return logger

def set_global_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure root logger level and format.
    """
    logging.basicConfig(
        level=LEVELS.get(level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("T-SiamTPN")

class Timer:
    """
    Context manager to measure elapsed time and optionally log it.
    """
    def __init__(self, logger: Optional[logging.Logger] = None, label: str = "scope"):
        self.logger = logger or logging.getLogger("T-SiamTPN")
        self.label = label
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self.logger.info(f"[Timer] {self.label} took {self.elapsed_ms:.2f} ms")
        return False

class FPSMeter:
    """
    Simple FPS meter with exponential smoothing.
    """
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self._last_t = None
        self._fps = 0.0

    def reset(self):
        self._last_t = None
        self._fps = 0.0

    def update(self, frames: int = 1) -> float:
        now = time.perf_counter()
        if self._last_t is None:
            self._last_t = now
            return self._fps
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0:
            return self._fps
        inst_fps = frames / dt
        self._fps = self.alpha * self._fps + (1 - self.alpha) * inst_fps
        return self._fps

def set_seed(seed: int = 42):
    """
    Set RNG seeds for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # local import to avoid hard dep here
        np.random.seed(seed)
    except Exception:
        pass
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resource_snapshot() -> Dict[str, Any]:
    """
    Collect a snapshot of system resources.
    Returns a dict with CPU%, RAM usage, and GPU memory if torch+CUDA available.
    """
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    snap = {
        "cpu_percent": float(cpu),
        "ram_percent": float(vm.percent),
        "ram_used_mb": float(vm.used) / (1024 * 1024),
        "ram_total_mb": float(vm.total) / (1024 * 1024),
    }
    if _HAS_TORCH and torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            used = torch.cuda.memory_allocated(idx) / (1024 * 1024)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)
            snap.update({
                "gpu_index": int(idx),
                "gpu_used_mb": float(used),
                "gpu_total_mb": float(total),
            })
        except Exception:
            pass
    return snap

def log_system_info(logger: Optional[logging.Logger] = None):
    """
    Log basic system information useful for reproducibility.
    """
    lg = logger or logging.getLogger("T-SiamTPN")
    lg.info(f"Python: {sys.version.split()[0]}")
    lg.info(f"psutil version: {psutil.__version__}")
    if _HAS_TORCH:
        lg.info(f"torch version: {torch.__version__}")
        lg.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            lg.info(f"GPU: {props.name}, Memory: {props.total_memory/(1024*1024):.1f} MB")

def log_resources(prefix: str = "", logger: Optional[logging.Logger] = None):
    """
    Log current resource utilization.
    """
    lg = logger or logging.getLogger("T-SiamTPN")
    snap = resource_snapshot()
    msg = (f"{prefix}CPU: {snap.get('cpu_percent', 0):.2f}% | "
           f"RAM: {snap.get('ram_percent', 0):.2f}% "
           f"({snap.get('ram_used_mb', 0):.1f}/{snap.get('ram_total_mb', 0):.1f} MB)")
    if "gpu_used_mb" in snap:
        msg += (f" | GPU[{snap.get('gpu_index', 0)}]: "
                f"{snap.get('gpu_used_mb', 0):.1f}/{snap.get('gpu_total_mb', 0):.1f} MB")
    lg.info(msg)

__all__ = [
    "configure_logger",
    "set_global_logging",
    "Timer",
    "FPSMeter",
    "set_seed",
    "resource_snapshot",
    "log_system_info",
    "log_resources",
]