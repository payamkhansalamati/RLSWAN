"""
utils.py
========
Shared utility helpers: seed management, metric normalisation, CSV I/O,
console logging and small numerical primitives used across all modules.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
_handler.setLevel(logging.DEBUG)

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator / denominator, or *fallback* when denominator ≈ 0."""
    if abs(denominator) < 1e-8:
        return fallback
    return numerator / denominator


def rms(arr: np.ndarray) -> float:
    """Root-mean-square of a 1-D array."""
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def clip_scalar(value: float, lo: float, hi: float) -> float:
    """Scalar clip without NumPy overhead."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Normalisation for radar / bar charts
# ---------------------------------------------------------------------------

def normalise_score(
    value: float,
    worst: float,
    best: float,
    higher_is_better: bool = True,
) -> float:
    """
    Map *value* to [0, 1] where 1 is the best attainable performance.

    Parameters
    ----------
    value:
        Raw metric value to normalise.
    worst:
        The worst expected value (maps to 0).
    best:
        The best expected value (maps to 1).
    higher_is_better:
        If True, larger raw values are better (e.g. episode return).
        If False, smaller raw values are better (e.g. collision rate).
    """
    if abs(best - worst) < 1e-10:
        return 0.5
    if higher_is_better:
        score = (value - worst) / (best - worst)
    else:
        # Invert: low metric value → high score
        score = (worst - value) / (worst - best)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Normalisation reference bounds
# (used consistently in evaluate.py and plots.py)
# ---------------------------------------------------------------------------

# Each entry: (worst, best, higher_is_better)
NORM_BOUNDS: Dict[str, Tuple[float, float, bool]] = {
    "collision_rate":        (1.0,  0.0,  False),
    "min_gap":               (0.0,  20.0, True),
    "headway_violation_pct": (100.0, 0.0, False),
    "mean_abs_jerk":         (5.0,   0.0, False),
    "rms_jerk":              (5.0,   0.0, False),
    "mean_abs_accel":        (4.0,   0.0, False),
    "speed_tracking_error":  (10.0,  0.0, False),
    "episode_return":        (-500.0, 0.0, True),
    "accel_oscillation":     (3.0,   0.0, False),
    "jerk_variance":         (10.0,  0.0, False),
}


def compute_normalised_scores(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Apply normalisation bounds to a dict of raw metrics.

    Returns a dict with the same keys, values in [0, 1].
    """
    scores: Dict[str, float] = {}
    for key, value in raw.items():
        if key in NORM_BOUNDS:
            worst, best, hib = NORM_BOUNDS[key]
            scores[key] = normalise_score(value, worst, best, hib)
        else:
            scores[key] = float(value)
    return scores


def aggregate_radar_scores(normalised: Dict[str, float]) -> Dict[str, float]:
    """
    Collapse normalised metric dict into 4 radar dimensions:
    Safety, Comfort, Performance, Stability.
    """
    safety_keys = ["collision_rate", "min_gap", "headway_violation_pct"]
    comfort_keys = ["mean_abs_jerk", "rms_jerk", "mean_abs_accel"]
    perf_keys = ["speed_tracking_error", "episode_return"]
    stab_keys = ["accel_oscillation", "jerk_variance"]

    def _mean(keys: List[str]) -> float:
        vals = [normalised[k] for k in keys if k in normalised]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "Safety":      _mean(safety_keys),
        "Comfort":     _mean(comfort_keys),
        "Performance": _mean(perf_keys),
        "Stability":   _mean(stab_keys),
    }


# ---------------------------------------------------------------------------
# CSV / DataFrame I/O
# ---------------------------------------------------------------------------

def save_dict_as_csv(data: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a CSV file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def load_csv_as_df(path: Path) -> pd.DataFrame:
    """Load a CSV written by save_dict_as_csv into a DataFrame."""
    return pd.read_csv(path)


def save_json(obj: Any, path: Path) -> None:
    """Serialise *obj* to JSON at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# DataFrame summary helpers
# ---------------------------------------------------------------------------

def dataframe_summary(df: pd.DataFrame, metric_cols: Optional[List[str]] = None) -> str:
    """Return a formatted string summary (mean ± std) of columns in *df*."""
    if metric_cols is None:
        metric_cols = list(df.select_dtypes(include=[np.number]).columns)
    lines = ["=" * 64, f"{'Metric':<32} {'Mean':>10} {'Std':>10}", "-" * 64]
    for col in metric_cols:
        if col in df.columns:
            lines.append(f"{col:<32} {df[col].mean():>10.4f} {df[col].std():>10.4f}")
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for benchmarking blocks."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Model-name helpers
# ---------------------------------------------------------------------------

def agent_name(agent_type: str, seed: int) -> str:
    """Return a canonical model filename stem."""
    return f"{agent_type}_seed{seed}"


def agent_display_name(agent_type: str) -> str:
    """Human-readable name for plots."""
    mapping = {
        "pid":              "PID Baseline",
        "vanilla_ppo":      "Vanilla PPO",
        "context_ppo":      "Context-Aware PPO",
        "hybrid_context_pid":"Hybrid Context+PID",
        "ablation_state":   "PPO+Context State",
        "ablation_reward":  "PPO+Context Reward Only",
    }
    return mapping.get(agent_type, agent_type)


# ---------------------------------------------------------------------------
# Smooth signal generation helper
# ---------------------------------------------------------------------------

def ar1_smooth(
    n: int,
    sigma: float,
    decay: float,
    rng: np.random.Generator,
    init: float = 0.0,
) -> np.ndarray:
    """
    Generate an AR(1) process:  x[t] = decay * x[t-1] + sigma * noise.

    Useful for synthetic speed perturbations and physiological signals.
    """
    x = np.empty(n)
    x[0] = init
    noise = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = decay * x[t - 1] + sigma * noise[t]
    return x


# ---------------------------------------------------------------------------
# Confidence-interval helper
# ---------------------------------------------------------------------------

def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Return (mean, std) for a sequence; safe for length-1 sequences."""
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0 if len(arr) <= 1 else 1))
