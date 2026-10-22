#!/usr/bin/env python3
"""Record matched policy rollouts and render presentation videos."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import imageio_ffmpeg
import gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3
import torch
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from controllers import HybridContextPIDController, PIDController, PPOController
from env_car_following import make_single_env
from evaluate import _load_obs_stats, compute_episode_metrics

OUTPUT = ROOT / "animations"
DATA_DIR = OUTPUT / "data"
VIDEO_DIR = OUTPUT / "rendered"
FPS = 30
POLICY_NAMES = {
    "context_ppo": "Context-aware PPO",
    "hybrid_context_pid": "Hybrid Context-PID (risk-governed context_ppo)",
}
PROFILE_DESCRIPTIONS = {
    "constant": "Constant lead speed (episode-randomized initial speed).",
    "sinusoidal": "Smooth speed oscillation (3 m/s amplitude, 8 s period).",
    "braking": "Abrupt lead braking (-2.5 m/s^2, starting at 40% of episode).",
    "stop_and_go": "Repeated go/stop cycles with gradual deceleration.",
    "random_smooth": "Autocorrelated lead-speed perturbations (AR(1)).",
}
COLORS = {
    "background": "#0b1220",
    "panel": "#121e30",
    "road": "#29384a",
    "lane": "#aeb9c8",
    "ego": "#4ea5ff",
    "lead": "#f5ad4e",
    "context": "#82d6b4",
    "risk": "#ff6b6b",
    "baseline": "#4ea5ff",
    "governed": "#82d6b4",
    "text": "#f4f7fb",
    "muted": "#adbbca",
    "grid": "#34465d",
}
PRESENTATION_PROFILE_OUTPUT = VIDEO_DIR / "animation1_five_profiles_presentation.mp4"
PRESENTATION_COMPARISON_OUTPUT = VIDEO_DIR / "animation2_matched_comparison_presentation.mp4"
PROFILE_SCENE_SECONDS = 10.0
PROFILE_PLAYBACK_RATE = 5.0
PRESENTATION_FRAMES = 50 * FPS


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_reference(config: Config, seed: int) -> Dict[str, Any]:
    model_dir = config.paths.models / f"context_ppo_seed{seed}"
    final = model_dir / "final_model.zip"
    checkpoint = final
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing final context_ppo checkpoint {checkpoint}; its saved "
            "VecNormalize statistics are required for a matched checkpoint/statistics pair."
        )
    vecnorm = model_dir / "vecnorm_stats.pkl"
    if not vecnorm.exists():
        raise FileNotFoundError(f"Missing observation statistics: {vecnorm}")
    return {
        "agent": "context_ppo",
        "seed": seed,
        "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "observation_statistics": str(vecnorm.resolve().relative_to(ROOT)),
        "observation_statistics_sha256": _sha256(vecnorm),
        "selected_checkpoint": checkpoint.name,
        "checkpoint_statistics_pair": "Both files are saved at the end of the same training run.",
        "hybrid_controller": "hybrid_context_pid wraps the same context_ppo checkpoint",
    }


def _load_policy(config: Config, seed: int) -> Tuple[Any, Any]:
    model_dir = config.paths.models / f"context_ppo_seed{seed}"
    model_path = model_dir / "final_model.zip"
    vecnorm_path = model_dir / "vecnorm_stats.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"Missing VecNormalize statistics for context_ppo seed {seed}: {vecnorm_path}"
        )
    return PPO.load(str(model_path)), _load_obs_stats(vecnorm_path)


def _controller(
    agent: str,
    model: Any,
    obs_stats: Any,
    config: Config,
) -> Any:
    if agent == "context_ppo":
        return PPOController(model, obs_stats)
    if agent == "hybrid_context_pid":
        return HybridContextPIDController(
            model, PIDController(config.pid, config.sim), obs_stats, config.hybrid
        )
    raise ValueError(f"Unsupported animation controller: {agent}")


def _record_episode(
    config: Config,
    agent: str,
    model: Any,
    obs_stats: Any,
    seed: int,
    scenario: str,
    profile: str,
) -> pd.DataFrame:
    env = make_single_env(
        config=config,
        use_context=True,
        use_context_reward=True,
        seed=seed,
        lead_profile=profile,
        scenario=scenario,
        normalize_observation=True,
        training_safety_shield=False,
    )
    controller = _controller(agent, model, obs_stats, config)
    controller.reset()
    obs, info = env.reset(seed=seed)
    trajectory: List[Dict[str, Any]] = []
    info = dict(info)
    info["commanded_action_m_s2"] = float("nan")
    if agent == "hybrid_context_pid":
        info.update({k: float(v) for k, v in controller.diagnostics().items()})
    trajectory.append(info)

    done = False
    while not done:
        action = controller.act(obs, info)
        command = float(np.asarray(action).reshape(-1)[0])
        obs, _reward, terminated, truncated, info = env.step(action)
        info = dict(info)
        info["commanded_action_m_s2"] = command
        if agent == "hybrid_context_pid":
            info.update({k: float(v) for k, v in controller.diagnostics().items()})
        trajectory.append(info)
        done = terminated or truncated
    env.close()

    frame = pd.DataFrame(trajectory)
    frame["time_s"] = frame["step"].astype(float) * config.sim.dt
    frame["agent"] = agent
    frame["policy_label"] = POLICY_NAMES[agent]
    frame["lead_profile"] = profile
    frame["scenario"] = scenario
    frame["evaluation_seed"] = seed
    return frame


def _safety_summary(frame: pd.DataFrame, config: Config) -> Dict[str, Any]:
    valid = frame.iloc[1:].copy() if len(frame) > 1 else frame.copy()
    metrics = compute_episode_metrics(valid.to_dict("records"), config.sim)
    ttc = valid["ttc_proxy"].to_numpy(dtype=float)
    collisions = valid["collision"].to_numpy(dtype=float)
    headway = valid["headway"].to_numpy(dtype=float)
    return {
        "steps": int(len(valid)),
        "duration_s": float(valid["time_s"].max()) if len(valid) else 0.0,
        "terminated_on_collision": bool(np.any(collisions > 0)),
        "collision_steps": int(np.sum(collisions > 0)),
        "min_gap_m": float(valid["gap"].min()) if len(valid) else None,
        "min_headway_s": float(headway.min()) if len(valid) else None,
        "headway_violation_steps": int(np.sum(headway < config.sim.safe_headway)),
        "min_ttc_proxy_s": float(ttc.min()) if len(valid) else None,
        "ttc_violation_steps": int(np.sum(ttc < 3.0)),
        "min_time_headway_threshold_s": config.sim.safe_headway,
        "ttc_violation_threshold_s": 3.0,
        "episode_metrics": metrics,
    }


def _compare_conditions(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> Dict[str, Any]:
    fields = [
        "v_lead", "context_stress", "context_fatigue",
        "context_h_pref", "context_comfort", "context_eff",
    ]
    overlap = min(len(left), len(right))
    mismatches: Dict[str, float] = {}
    for field in fields:
        if field not in left or field not in right:
            mismatches[field] = float("inf")
            continue
        a = left[field].to_numpy(dtype=float)[:overlap]
        b = right[field].to_numpy(dtype=float)[:overlap]
        mismatches[field] = float(np.max(np.abs(a - b))) if overlap else 0.0
    regime_equal = (
        left["regime"].astype(str).to_numpy()[:overlap]
        == right["regime"].astype(str).to_numpy()[:overlap]
    ).all()
    initial_mismatches = {}
    for field in ("gap", "v_ego", "v_lead"):
        initial_mismatches[field] = float(
            abs(float(left[field].iloc[0]) - float(right[field].iloc[0]))
        )
    return {
        "same_seed": bool(left["evaluation_seed"].iloc[0] == right["evaluation_seed"].iloc[0]),
        "same_scenario": bool(left["scenario"].iloc[0] == right["scenario"].iloc[0]),
        "same_lead_profile": bool(left["lead_profile"].iloc[0] == right["lead_profile"].iloc[0]),
        "same_initial_conditions": all(value <= 1e-8 for value in initial_mismatches.values()),
        "initial_condition_abs_difference": initial_mismatches,
        "same_lead_speed_through_overlap": mismatches["v_lead"] <= 1e-8,
        "same_context_through_overlap": all(
            mismatches[k] <= 1e-8
            for k in (
                "context_stress", "context_fatigue", "context_h_pref",
                "context_comfort", "context_eff",
            )
        ) and bool(regime_equal),
        "overlap_steps": overlap,
        "max_abs_difference_by_field": mismatches,
        "context_regime_equal_through_overlap": bool(regime_equal),
    }


def _save_trajectory(frame: pd.DataFrame, filename: str) -> Path:
    path = DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.9g")
    return path


def _run_recording(seed: int) -> None:
    config = Config()
    model_ref = _model_reference(config, seed)
    model, obs_stats = _load_policy(config, seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    profiles = list(config.lead.profiles)
    if len(profiles) != 5:
        raise ValueError(
            f"Animation 1 requires five configured lead profiles; found {len(profiles)}: {profiles}"
        )

    profile_runs: List[Dict[str, Any]] = []
    for index, profile in enumerate(profiles):
        episode_seed = 7100 + index
        scenario = "escalation"
        frame = _record_episode(
            config, "hybrid_context_pid", model, obs_stats,
            episode_seed, scenario, profile,
        )
        summary = _safety_summary(frame, config)
        filename = f"animation1_{profile}.csv"
        path = _save_trajectory(frame, filename)
        profile_runs.append({
            "profile": profile,
            "description": PROFILE_DESCRIPTIONS[profile],
            "scenario": scenario,
            "evaluation_seed": episode_seed,
            "trajectory_file": str(path.relative_to(ROOT)),
            "summary": summary,
        })
        print(
            f"Animation 1 {profile}: {summary['steps']} steps, "
            f"min gap={summary['min_gap_m']:.2f} m, "
            f"TTC violations={summary['ttc_violation_steps']}, "
            f"collisions={summary['collision_steps']}"
        )

    search_profiles = profiles
    search_scenarios = ["calm", "stressed", "fatigued", "both", "switching", "escalation"]
    search_seeds = list(range(700, 705))
    search_rows: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    closest: Optional[Tuple[float, Dict[str, Any], pd.DataFrame, pd.DataFrame]] = None
    pairs_evaluated = 0

    for scenario in search_scenarios:
        for profile in search_profiles:
            for pair_seed in search_seeds:
                base = _record_episode(
                    config, "context_ppo", model, obs_stats,
                    pair_seed, scenario, profile,
                )
                hybrid = _record_episode(
                    config, "hybrid_context_pid", model, obs_stats,
                    pair_seed, scenario, profile,
                )
                conditions = _compare_conditions(base, hybrid)
                base_summary = _safety_summary(base, config)
                hybrid_summary = _safety_summary(hybrid, config)
                paired = all(
                    conditions[k] for k in (
                        "same_seed", "same_scenario", "same_lead_profile",
                        "same_initial_conditions", "same_lead_speed_through_overlap",
                        "same_context_through_overlap",
                    )
                )
                base_fails_ttc = base_summary["ttc_violation_steps"] > 0
                hybrid_safe = (
                    hybrid_summary["ttc_violation_steps"] == 0
                    and not hybrid_summary["terminated_on_collision"]
                )
                qualifies = paired and base_fails_ttc and hybrid_safe
                row = {
                    "scenario": scenario,
                    "lead_profile": profile,
                    "evaluation_seed": pair_seed,
                    "conditions_match": paired,
                    "context_ppo_ttc_violation_steps": base_summary["ttc_violation_steps"],
                    "context_ppo_collision_steps": base_summary["collision_steps"],
                    "context_ppo_min_ttc_s": base_summary["min_ttc_proxy_s"],
                    "hybrid_context_pid_ttc_violation_steps": hybrid_summary["ttc_violation_steps"],
                    "hybrid_context_pid_collision_steps": hybrid_summary["collision_steps"],
                    "hybrid_context_pid_min_ttc_s": hybrid_summary["min_ttc_proxy_s"],
                    "qualifies_for_ttc_comparison": qualifies,
                    "condition_check": conditions,
                }
                search_rows.append(row)
                pairs_evaluated += 1

                min_ttc_delta = (
                    float(hybrid_summary["min_ttc_proxy_s"])
                    - float(base_summary["min_ttc_proxy_s"])
                )
                if closest is None or min_ttc_delta > closest[0]:
                    closest = (min_ttc_delta, row, base, hybrid)
                if qualifies:
                    selected = {
                        "row": row,
                        "context_ppo": base,
                        "hybrid_context_pid": hybrid,
                    }
                    break
            if selected:
                break
        if selected:
            break

    search_path = DATA_DIR / "comparison_search.csv"
    pd.DataFrame(search_rows).drop(columns=["condition_check"]).to_csv(
        search_path, index=False
    )
    candidate = selected
    comparison_status = "verified_ttc_violation_contrast"
    if candidate is None:
        if closest is None:
            raise RuntimeError("No comparison scenarios were evaluated.")
        _, closest_row, base, hybrid = closest
        candidate = {
            "row": closest_row,
            "context_ppo": base,
            "hybrid_context_pid": hybrid,
        }
        comparison_status = "no_verified_ttc_violation_contrast"

    comparison_row = candidate["row"]
    comparison_files = {}
    for agent in ("context_ppo", "hybrid_context_pid"):
        filename = f"animation2_{agent}.csv"
        path = _save_trajectory(candidate[agent], filename)
        comparison_files[agent] = str(path.relative_to(ROOT))
    conditions = _compare_conditions(
        candidate["context_ppo"], candidate["hybrid_context_pid"]
    )
    base_summary = _safety_summary(candidate["context_ppo"], config)
    hybrid_summary = _safety_summary(candidate["hybrid_context_pid"], config)

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"

    metadata = {
        "training": {
            "agent": "context_ppo",
            "training_seed": seed,
            "configured_timesteps": config.ppo.total_timesteps,
            "checkpoint_selection": (
                "final_model.zip with vecnorm_stats.pkl; both are saved at training "
                "completion and form a matched model/normalization pair"
            ),
            "model_reference": model_ref,
            "repository_revision": revision,
            "runtime": {
                "python": sys.version,
                "platform": sys.platform,
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                "imageio": importlib.metadata.version("imageio"),
                "imageio_ffmpeg": imageio_ffmpeg.__version__,
            },
        },
        "config": asdict(config),
        "animation1": {
            "controller": "hybrid_context_pid",
            "scenario": "escalation",
            "scenario_description": "Deterministic calm → stressed → fatigued → both context schedule.",
            "runs": profile_runs,
        },
        "animation2": {
            "status": comparison_status,
            "criterion": (
                "Failure: at least one evaluation TTC proxy < 3.0 s. "
                "Safe for this illustrative comparison: zero TTC violations "
                "and zero collision steps (collision is gap <= collision_gap)."
            ),
            "scenario": comparison_row["scenario"],
            "lead_profile": comparison_row["lead_profile"],
            "evaluation_seed": comparison_row["evaluation_seed"],
            "paired_conditions": conditions,
            "context_ppo_summary": base_summary,
            "hybrid_context_pid_summary": hybrid_summary,
            "trajectories": comparison_files,
            "search": {
                "profiles": search_profiles,
                "scenarios": search_scenarios,
                "seeds": search_seeds,
                "matched_pairs_evaluated_until_selection_or_exhaustion": pairs_evaluated,
                "results_file": str(search_path.relative_to(ROOT)),
            },
            "limitations": [
                "This is one illustrative matched scenario, not evidence of general superiority.",
                "The project's risk-governed policy is implemented as hybrid_context_pid, "
                "which wraps the context_ppo checkpoint with PID supervision and filtering.",
                "TTC is the project's proxy: gap / closing speed when ego closes faster than 0.1 m/s; otherwise 999 s.",
            ],
        },
    }
    _write_json(DATA_DIR / "animation_results.json", metadata)
    print(
        f"Comparison: {comparison_status}; {comparison_row['scenario']}/"
        f"{comparison_row['lead_profile']}, seed={comparison_row['evaluation_seed']}; "
        f"matched={all(conditions[k] for k in ('same_initial_conditions', 'same_lead_speed_through_overlap', 'same_context_through_overlap'))}"
    )


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"time_s", "gap", "v_ego", "v_lead", "headway", "ttc_proxy", "accel"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing fields required for rendering: {sorted(missing)}")
    return frame


def _value(frame: pd.DataFrame, key: str, time_s: float) -> float:
    times = frame["time_s"].to_numpy(dtype=float)
    values = frame[key].to_numpy(dtype=float)
    return float(np.interp(time_s, times, values))


def _frame_at(frame: pd.DataFrame, time_s: float) -> int:
    return int(np.searchsorted(frame["time_s"].to_numpy(dtype=float), time_s, side="right"))


def _add_car(ax: Any, x: float, y: float, color: str, label: str) -> List[Any]:
    length, height = 4.5, 1.55
    body = FancyBboxPatch(
        (x - length / 2, y - height / 2), length, height,
        boxstyle="round,pad=0.08,rounding_size=0.38",
        facecolor=color, edgecolor="#ffffff", linewidth=1.6, zorder=5,
    )
    ax.add_patch(body)
    canopy = Polygon(
        [
            (x - 1.15, y + 0.30), (x - 0.72, y + 0.82),
            (x + 0.72, y + 0.82), (x + 1.17, y + 0.30),
        ],
        closed=True, facecolor="#d6e5f4", edgecolor="#ffffff",
        linewidth=0.8, alpha=0.94, zorder=6,
    )
    ax.add_patch(canopy)
    wheels = []
    for wheel_x in (x - 1.35, x + 1.35):
        wheel = Circle((wheel_x, y - 0.76), 0.20, color="#0b1220", zorder=7)
        ax.add_patch(wheel)
        wheels.append(wheel)
    label_artist = ax.text(
        x, y - 0.03, label, ha="center", va="center", color="#07111d",
        fontsize=13, fontweight="bold", zorder=8,
    )
    return [body, canopy, *wheels, label_artist]


def _make_road(ax: Any, title: str, subtitle: str) -> Dict[str, Any]:
    ax.set_facecolor(COLORS["road"])
    ax.set_xlim(2, 122)
    ax.set_ylim(0, 10)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 121, 10))
    ax.tick_params(axis="x", colors=COLORS["muted"], labelsize=10)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.set_xlabel("Longitudinal position [m] — shared scale", color=COLORS["muted"], fontsize=10)
    ax.axhline(2.0, color=COLORS["lane"], linewidth=2.0, alpha=0.8)
    ax.axhline(8.0, color=COLORS["lane"], linewidth=2.0, alpha=0.8)
    for x in np.arange(2, 122, 8):
        ax.plot([x, x + 4], [5, 5], color="#e3e9f0", linewidth=1.3, alpha=0.7)
    ax.text(0.01, 1.08, title, transform=ax.transAxes, color=COLORS["text"],
            fontsize=16, fontweight="bold", ha="left")
    ax.text(0.01, 0.97, subtitle, transform=ax.transAxes, color=COLORS["muted"],
            fontsize=10, ha="left", va="top")

    ego_x = 10.0
    ego_parts = _add_car(ax, ego_x, 5, COLORS["ego"], "EGO")
    lead_parts = _add_car(ax, ego_x + 4.5 + 20.0, 5, COLORS["lead"], "LEAD")
    ego_speed = ax.text(ego_x, 6.25, "", color=COLORS["text"], ha="center",
                        fontsize=11, fontweight="bold")
    lead_speed = ax.text(ego_x + 24.5, 6.25, "", color=COLORS["text"], ha="center",
                         fontsize=11, fontweight="bold")
    gap_arrow = ax.annotate(
        "", xy=(ego_x + 4.5 + 20.0 - 2.25, 5),
        xytext=(ego_x + 2.25, 5),
        arrowprops={"arrowstyle": "<->", "color": "#ffffff", "lw": 1.8},
        zorder=4,
    )
    gap_label = ax.text(ego_x + 12, 5.6, "", color="#ffffff", ha="center",
                        fontsize=10, fontweight="bold")
    accel_arrow = ax.annotate(
        "", xy=(ego_x, 3.55), xytext=(ego_x + 2.0, 3.55),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["ego"], "lw": 2.4},
        zorder=4,
    )
    accel_label = ax.text(ego_x, 3.05, "", color=COLORS["text"], fontsize=10, ha="center")
    ax.text(0.99, 0.05, "Vehicle drawings schematic; longitudinal separation uses measured gap.",
            transform=ax.transAxes, color=COLORS["muted"], fontsize=8, ha="right")
    return {
        "ax": ax, "ego_parts": ego_parts, "lead_parts": lead_parts,
        "ego_speed": ego_speed, "lead_speed": lead_speed, "gap_arrow": gap_arrow,
        "gap_label": gap_label, "accel_arrow": accel_arrow,
        "accel_label": accel_label, "ego_x": ego_x,
        "lead_initial_x": ego_x + 4.5 + 20.0,
        "ego_drawn_x": ego_x,
        "lead_drawn_x": ego_x + 4.5 + 20.0,
    }


def _update_road(road: Dict[str, Any], frame: pd.DataFrame, time_s: float) -> None:
    ax = road["ax"]
    i = min(_frame_at(frame, time_s), len(frame) - 1)
    gap = _value(frame, "gap", time_s)
    v_ego = _value(frame, "v_ego", time_s)
    v_lead = _value(frame, "v_lead", time_s)
    acceleration = _value(frame, "accel", time_s)
    ego_x = road["ego_x"]
    lead_x = ego_x + 4.5 + gap
    for parts, center_x, key in (
        (road["ego_parts"], ego_x, "ego_drawn_x"),
        (road["lead_parts"], lead_x, "lead_drawn_x"),
    ):
        body, canopy, wheel_a, wheel_b, label_artist = parts
        delta_x = center_x - road[key]
        body.set_x(center_x - 2.25)
        canopy.set_xy(canopy.get_xy() + np.array([delta_x, 0.0]))
        wheel_a.center = (center_x - 1.35, 4.24)
        wheel_b.center = (center_x + 1.35, 4.24)
        label_artist.set_position((center_x, 4.97))
        road[key] = center_x
    road["ego_speed"].set_position((ego_x, 6.25))
    road["ego_speed"].set_text(f"EGO  {v_ego:.1f} m/s")
    road["lead_speed"].set_position((lead_x, 6.25))
    road["lead_speed"].set_text(f"LEAD  {v_lead:.1f} m/s")
    road["gap_arrow"].xy = (lead_x - 2.25, 5)
    road["gap_arrow"].set_position((ego_x + 2.25, 5))
    road["gap_label"].set_position((ego_x + gap / 2 + 2.25, 5.62))
    road["gap_label"].set_text(f"{gap:.1f} m")
    ax_pos = (ego_x, 3.55)
    direction = 1 if acceleration >= 0 else -1
    road["accel_arrow"].xy = (ego_x + direction * 2.2, 3.55)
    road["accel_arrow"].set_position(ax_pos)
    road["accel_label"].set_position((ego_x, 3.05))
    road["accel_label"].set_text(f"accel {acceleration:+.2f} m/s²")


def _style_chart(ax: Any, ylabel: str, threshold: Optional[float] = None) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    ax.spines[:].set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], alpha=0.5, linewidth=0.7)
    ax.set_ylabel(ylabel, color=COLORS["text"], fontsize=10)
    if threshold is not None:
        ax.axhline(threshold, color=COLORS["risk"], linestyle="--", linewidth=1.4)


def _charts(
    axes: Sequence[Any],
    frames: Sequence[pd.DataFrame],
    time_s: float,
    colors: Sequence[str],
) -> List[Any]:
    lines = []
    configs = [
        ("headway", "Headway [s]", 1.5),
        ("ttc_proxy", "TTC [s; cap=12]", 3.0),
        ("v_ego", "Speed [m/s]", None),
    ]
    for ax, (key, ylabel, threshold) in zip(axes, configs):
        _style_chart(ax, ylabel, threshold)
        for frame, color in zip(frames, colors):
            if key == "v_ego":
                ego_line, = ax.plot([], [], color=color, linewidth=2.2, label="Ego")
                lead_line, = ax.plot([], [], color=COLORS["lead"], linewidth=1.5,
                                     linestyle="--", label="Lead")
                lines.extend([(frame, key, ego_line), (frame, "v_lead", lead_line)])
            else:
                line, = ax.plot([], [], color=color, linewidth=2.0)
                lines.append((frame, key, line))
        if key == "ttc_proxy":
            ax.set_ylim(0, 12)
        if key == "headway":
            ax.set_ylim(0, 6)
        if key == "v_ego":
            ax.set_ylim(0, 35)
    axes[-1].set_xlabel("Simulation time [s]", color=COLORS["text"], fontsize=10)
    axes[-1].legend(loc="upper right", frameon=False, labelcolor=COLORS["text"], fontsize=8)
    return lines


def _update_chart_lines(
    lines: Sequence[Tuple[pd.DataFrame, str, Any]],
    time_s: float,
    source_frame: Optional[pd.DataFrame] = None,
) -> None:
    for original_frame, key, line in lines:
        frame = source_frame if source_frame is not None else original_frame
        count = _frame_at(frame, time_s)
        x = frame["time_s"].to_numpy(dtype=float)[:count]
        y = frame[key].to_numpy(dtype=float)[:count]
        if key == "ttc_proxy":
            y = np.clip(y, 0.0, 12.0)
        line.set_data(x, y)


def _create_live_panel(ax: Any, policy: str) -> Dict[str, Any]:
    ax.set_facecolor(COLORS["panel"])
    ax.axis("off")
    title = ax.text(
        0.05, 0.96, policy, transform=ax.transAxes, color=COLORS["text"],
        fontsize=12, fontweight="bold", va="top", wrap=True,
    )
    rows = [
        ("EGO SPEED", COLORS["ego"]),
        ("LEAD SPEED", COLORS["lead"]),
        ("FOLLOWING GAP", COLORS["text"]),
        ("TIME HEADWAY", COLORS["context"]),
        ("TTC PROXY", COLORS["text"]),
        ("EGO ACCELERATION", COLORS["text"]),
        ("HYBRID RISK (raw)", COLORS["context"]),
        ("CONTEXT", COLORS["context"]),
    ]
    values = []
    artists = [title]
    y = 0.82
    for label, color in rows:
        artists.append(ax.text(
            0.06, y, label, transform=ax.transAxes, color=COLORS["muted"],
            fontsize=9, fontweight="bold", va="top",
        ))
        value = ax.text(
            0.06, y - 0.045, "", transform=ax.transAxes, color=color,
            fontsize=12, va="top",
        )
        values.append(value)
        artists.append(value)
        y -= 0.10
    return {"artists": artists, "values": values}


def _update_live_panel(
    panel: Dict[str, Any],
    frame: pd.DataFrame,
    time_s: float,
) -> None:
    gap = _value(frame, "gap", time_s)
    ego = _value(frame, "v_ego", time_s)
    lead = _value(frame, "v_lead", time_s)
    headway = _value(frame, "headway", time_s)
    ttc = _value(frame, "ttc_proxy", time_s)
    accel = _value(frame, "accel", time_s)
    risk = _value(frame, "hybrid_risk_raw", time_s) if "hybrid_risk_raw" in frame else float("nan")
    stress = _value(frame, "context_stress", time_s)
    fatigue = _value(frame, "context_fatigue", time_s)
    preferred_headway = _value(frame, "context_h_pref", time_s)
    metrics = [
        ("EGO SPEED", f"{ego:.2f} m/s", COLORS["ego"]),
        ("LEAD SPEED", f"{lead:.2f} m/s", COLORS["lead"]),
        ("FOLLOWING GAP", f"{gap:.2f} m", COLORS["text"]),
        ("TIME HEADWAY", f"{headway:.2f} s  (threshold 1.5 s)", COLORS["risk"] if headway < 1.5 else COLORS["context"]),
        ("TTC PROXY", f"{ttc:.2f} s  (threshold 3.0 s)" if ttc < 998 else "not closing", COLORS["risk"] if ttc < 3 else COLORS["text"]),
        ("EGO ACCELERATION", f"{accel:+.2f} m/s²", COLORS["text"]),
    ]
    if math.isfinite(risk):
        risk_text = f"{risk:.3f}"
        risk_color = COLORS["risk"] if risk >= 0.3 else COLORS["context"]
    else:
        risk_text = "not available"
        risk_color = COLORS["muted"]
    metrics.append((
        "CONTEXT",
        f"stress {stress:.2f}  |  fatigue {fatigue:.2f}  |  h_pref {preferred_headway:.2f} s",
        COLORS["context"],
    ))
    metrics.insert(6, ("HYBRID RISK (raw)", risk_text, risk_color))
    for artist, (_, value, color) in zip(panel["values"], metrics):
        artist.set_text(value)
        artist.set_color(color)


def _presentation_car(ax: Any, x: float, color: str) -> List[Any]:
    length, height, y = 4.5, 1.65, 3.9
    body = FancyBboxPatch(
        (x - length / 2, y - height / 2),
        length,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.38",
        facecolor=color,
        edgecolor="#ffffff",
        linewidth=2.2,
        zorder=5,
    )
    canopy = Polygon(
        [
            (x - 1.15, y + 0.25), (x - 0.72, y + 0.77),
            (x + 0.72, y + 0.77), (x + 1.17, y + 0.25),
        ],
        closed=True,
        facecolor="#d6e5f4",
        edgecolor="#ffffff",
        linewidth=1.1,
        alpha=0.96,
        zorder=6,
    )
    ax.add_patch(body)
    ax.add_patch(canopy)
    wheels = []
    for wheel_x in (x - 1.35, x + 1.35):
        wheel = Circle((wheel_x, y - 0.78), 0.24, color="#0b1220", zorder=7)
        ax.add_patch(wheel)
        wheels.append(wheel)
    return [body, canopy, *wheels]


def _presentation_road(
    ax: Any,
    title: str,
    subtitle: str,
    x_max: float,
    ego_color: str = COLORS["ego"],
) -> Dict[str, Any]:
    ax.set_facecolor(COLORS["road"])
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 10.0)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    ax.axhline(1.4, color=COLORS["lane"], linewidth=3.0, alpha=0.8, zorder=1)
    ax.axhline(5.8, color=COLORS["lane"], linewidth=3.0, alpha=0.8, zorder=1)
    for lane_x in np.arange(0.0, x_max, 8.0):
        ax.plot(
            [lane_x, min(lane_x + 4.0, x_max)],
            [3.9, 3.9],
            color="#e3e9f0",
            linewidth=2.0,
            alpha=0.72,
            zorder=1,
        )

    title_artist = ax.text(
        0.018, 0.97, title, transform=ax.transAxes, color=COLORS["text"],
        fontsize=19, fontweight="bold", ha="left", va="top",
    )
    subtitle_artist = ax.text(
        0.018, 0.86, subtitle, transform=ax.transAxes, color=COLORS["muted"],
        fontsize=12, ha="left", va="top",
    )
    ego_x = 6.0
    initial_gap = 20.0
    lead_x = ego_x + 4.5 + initial_gap
    ego_parts = _presentation_car(ax, ego_x, ego_color)
    lead_parts = _presentation_car(ax, lead_x, COLORS["lead"])
    ego_name = ax.text(
        ego_x, 7.65, "EGO", color=ego_color, ha="center", va="center",
        fontsize=15, fontweight="bold",
    )
    ego_speed = ax.text(
        ego_x, 6.6, "", color=COLORS["text"], ha="center", va="center",
        fontsize=15, fontweight="bold",
    )
    lead_name = ax.text(
        lead_x, 9.55, "LEAD", color=COLORS["lead"], ha="center", va="center",
        fontsize=15, fontweight="bold",
    )
    lead_speed = ax.text(
        lead_x, 8.55, "", color=COLORS["text"], ha="center", va="center",
        fontsize=15, fontweight="bold",
    )
    gap_arrow = ax.annotate(
        "",
        xy=(lead_x - 2.25, 5.0),
        xytext=(ego_x + 2.25, 5.0),
        arrowprops={"arrowstyle": "<->", "color": "#ffffff", "lw": 2.5},
        zorder=4,
    )
    gap_label = ax.text(
        ego_x + 12.25, 5.55, "", color="#ffffff", ha="center", va="center",
        fontsize=16, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": COLORS["panel"],
              "edgecolor": "#ffffff", "alpha": 0.96},
        zorder=8,
    )
    scale_note = ax.text(
        0.5, 0.035, "Measured gap · scale [m]",
        transform=ax.transAxes, color=COLORS["muted"], fontsize=10, ha="center",
        va="bottom",
    )
    cap_note = ax.text(
        0.98, 0.055, "", transform=ax.transAxes, color=COLORS["risk"],
        fontsize=11, fontweight="bold", ha="right", va="bottom",
    )
    return {
        "ax": ax,
        "title": title_artist,
        "subtitle": subtitle_artist,
        "ego_parts": ego_parts,
        "lead_parts": lead_parts,
        "ego_name": ego_name,
        "lead_name": lead_name,
        "ego_speed": ego_speed,
        "lead_speed": lead_speed,
        "gap_arrow": gap_arrow,
        "gap_label": gap_label,
        "scale_note": scale_note,
        "cap_note": cap_note,
        "ego_x": ego_x,
        "ego_drawn_x": ego_x,
        "lead_drawn_x": lead_x,
        "ego_color": ego_color,
        "x_max": x_max,
    }


def _update_presentation_road(
    road: Dict[str, Any],
    frame: pd.DataFrame,
    time_s: float,
    gap_limit_m: float,
) -> None:
    raw_gap = _value(frame, "gap", time_s)
    gap = max(0.0, raw_gap)
    ego_x = road["ego_x"]
    lead_x = ego_x + 4.5 + gap
    for parts, center_x, key in (
        (road["ego_parts"], ego_x, "ego_drawn_x"),
        (road["lead_parts"], lead_x, "lead_drawn_x"),
    ):
        body, canopy, wheel_a, wheel_b = parts
        delta_x = center_x - road[key]
        body.set_x(center_x - 2.25)
        canopy.set_xy(canopy.get_xy() + np.array([delta_x, 0.0]))
        wheel_a.center = (center_x - 1.35, 3.12)
        wheel_b.center = (center_x + 1.35, 3.12)
        road[key] = center_x

    ego_speed = _value(frame, "v_ego", time_s)
    lead_speed = _value(frame, "v_lead", time_s)
    road["ego_name"].set_position((ego_x, 7.65))
    road["ego_speed"].set_position((ego_x, 6.6))
    road["ego_speed"].set_text(f"{ego_speed:.1f} m/s")
    road["lead_name"].set_position((lead_x, 9.55))
    road["lead_speed"].set_position((lead_x, 8.55))
    road["lead_speed"].set_text(f"{lead_speed:.1f} m/s")
    road["gap_arrow"].xy = (lead_x - 2.25, 5.0)
    road["gap_arrow"].set_position((ego_x + 2.25, 5.0))
    road["gap_label"].set_position((ego_x + 2.25 + gap / 2.0, 5.55))
    road["gap_label"].set_text(f"{raw_gap:.1f} m")
    road["cap_note"].set_text(
        f"Recorded gap at simulation limit ({gap_limit_m:.0f} m)"
        if raw_gap >= gap_limit_m - 1e-6
        else ""
    )


def _presentation_artists(road: Dict[str, Any]) -> List[Any]:
    return [
        *road["ego_parts"], *road["lead_parts"], road["title"], road["subtitle"],
        road["ego_name"], road["lead_name"], road["ego_speed"], road["lead_speed"],
        road["gap_arrow"], road["gap_label"], road["scale_note"], road["cap_note"],
    ]


def _style_presentation_chart(ax: Any, ylabel: str, xlabel: Optional[str] = None) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["muted"], labelsize=13, length=5)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(1.1)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.48, linewidth=0.8)
    ax.set_ylabel(ylabel, color=COLORS["text"], fontsize=15, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS["text"], fontsize=14, labelpad=8)


def _ttc_status(ttc: float) -> Tuple[str, str]:
    if ttc >= 998.0:
        return "TTC: Not closing", COLORS["muted"]
    if ttc < 3.0:
        return f"TTC: {ttc:.1f} s · below 3 s", COLORS["risk"]
    return f"TTC: {ttc:.1f} s", COLORS["text"]


def _headway_status(frame: pd.DataFrame, time_s: float, sim: Any) -> Tuple[str, str]:
    speed = _value(frame, "v_ego", time_s)
    if speed <= sim.headway_speed_floor:
        return "Headway: — (ego stopped)", COLORS["muted"]
    headway = _value(frame, "headway", time_s)
    if headway < sim.safe_headway:
        return f"Headway: {headway:.1f} s · below {sim.safe_headway:.1f} s", COLORS["risk"]
    return f"Headway: {headway:.1f} s", COLORS["context"]


def _road_artists(road: Dict[str, Any]) -> List[Any]:
    return [
        *road["ego_parts"], *road["lead_parts"], road["ego_speed"],
        road["lead_speed"], road["gap_arrow"], road["gap_label"],
        road["accel_arrow"], road["accel_label"],
    ]


def _write_video(
    fig: Any,
    frames: Sequence[pd.DataFrame],
    output: Path,
    draw_frame: Any,
    duration_s: Optional[float] = None,
    frame_count: Optional[int] = None,
    dynamic_artists: Sequence[Any] = (),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    max_time = duration_s if duration_s is not None else max(
        float(frame["time_s"].max()) for frame in frames
    )
    output_frame_count = (
        int(round(max_time * FPS)) if frame_count is None else frame_count
    )
    if output_frame_count <= 0:
        raise ValueError("Video must contain at least one frame.")
    for artist in dynamic_artists:
        artist.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    ordered_artists = sorted(dynamic_artists, key=lambda artist: artist.get_zorder())
    with imageio.get_writer(
        output,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-preset", "medium"],
    ) as writer:
        for frame_idx in range(output_frame_count):
            time_s = min((frame_idx + 0.5) / FPS, max_time)
            draw_frame(time_s, frame_idx, output_frame_count)
            fig.canvas.restore_region(background)
            for artist in ordered_artists:
                if artist.axes is None:
                    fig.draw_artist(artist)
                else:
                    artist.axes.draw_artist(artist)
            fig.canvas.blit(fig.bbox)
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
            if frame_idx % (FPS * 30) == 0:
                print(f"  Rendering {output.name}: {time_s:.1f}/{max_time:.1f} sim s")
    plt.close(fig)


def _render_profile_video(metadata: Dict[str, Any]) -> None:
    sim = Config().sim
    runs = metadata["animation1"]["runs"]
    profile_order = [
        "constant", "sinusoidal", "braking", "stop_and_go", "random_smooth"
    ]
    if [run["profile"] for run in runs] != profile_order:
        raise ValueError(
            "Animation 1 requires the recorded profiles in this order: "
            + ", ".join(profile_order)
        )
    frames = [_load_frame(ROOT / run["trajectory_file"]) for run in runs]
    for run, frame in zip(runs, frames):
        if frame["time_s"].iloc[0] > 0 or frame["time_s"].iloc[-1] < 50.0:
            raise ValueError(
                f"{run['profile']} trajectory does not contain the complete 0–50 s episode."
            )
    max_recorded_gap = max(float(frame["gap"].max()) for frame in frames)
    x_max = max(sim.gap_max, max_recorded_gap) + 17.0

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=COLORS["background"])
    road_ax = fig.add_axes([0.045, 0.455, 0.91, 0.355])
    road = _presentation_road(road_ax, "", "", x_max)
    road["title"].set_visible(False)
    road["subtitle"].set_visible(False)
    road["scale_note"].set_text(
        "Vehicle symbols are schematic · gap and longitudinal positions use recorded metres"
    )

    speed_ax = fig.add_axes([0.075, 0.17, 0.86, 0.21])
    _style_presentation_chart(speed_ax, "Speed [m/s]", "Episode simulation time [s]")
    speed_ax.set_xlim(0.0, 50.0)
    max_speed = max(
        float(max(frame["v_ego"].max(), frame["v_lead"].max())) for frame in frames
    )
    speed_ax.set_ylim(0.0, max(20.0, math.ceil(max_speed / 5.0) * 5.0))
    speed_ax.set_xticks(np.arange(0.0, 51.0, 10.0))
    speed_ax.set_title(
        "Speed history",
        loc="left",
        color=COLORS["text"],
        fontsize=16,
        fontweight="bold",
        pad=8,
    )
    ego_line, = speed_ax.plot(
        [], [], color=COLORS["ego"], linewidth=3.6, label="Ego", zorder=4
    )
    lead_line, = speed_ax.plot(
        [], [], color=COLORS["lead"], linewidth=3.3, linestyle="--",
        label="Lead", zorder=4,
    )
    cursor = speed_ax.axvline(0.0, color="#ffffff", linestyle=":", linewidth=2.0)
    speed_ax.legend(
        loc="upper right", frameon=False, labelcolor=COLORS["text"], fontsize=13,
        ncols=2,
    )

    title = fig.text(
        0.05, 0.975, "", color=COLORS["text"], fontsize=27, fontweight="bold",
        va="top",
    )
    description = fig.text(
        0.05, 0.925, "", color=COLORS["muted"], fontsize=14, va="top",
    )
    video_clock = fig.text(
        0.95, 0.975, "", color=COLORS["context"], fontsize=15,
        fontweight="bold", ha="right", va="top",
    )
    simulation_clock = fig.text(
        0.95, 0.925, "", color=COLORS["text"], fontsize=14,
        ha="right", va="top",
    )
    status_gap = fig.text(
        0.055, 0.421, "", color=COLORS["text"], fontsize=13, fontweight="bold",
    )
    status_ttc = fig.text(0.34, 0.421, "", color=COLORS["text"], fontsize=13)
    status_headway = fig.text(0.59, 0.421, "", color=COLORS["context"], fontsize=13)
    status_context = fig.text(0.83, 0.421, "", color=COLORS["context"], fontsize=13)
    episode_result = fig.text(
        0.05, 0.07, "", color=COLORS["text"], fontsize=14, fontweight="bold",
    )
    limitation = fig.text(
        0.05, 0.025, "Illustrative recorded episodes; not a general performance guarantee.",
        color=COLORS["muted"], fontsize=12, va="bottom",
    )

    profile_titles = {
        "constant": "Constant",
        "sinusoidal": "Sinusoidal",
        "braking": "Braking",
        "stop_and_go": "Stop-and-go",
        "random_smooth": "Smooth random",
    }
    progress_x, progress_y, progress_width, progress_height = 0.05, 0.855, 0.90, 0.035
    cell_width = progress_width / len(runs)
    progress_boxes = []
    progress_labels = []
    for index, run in enumerate(runs):
        box = FancyBboxPatch(
            (progress_x + index * cell_width, progress_y),
            cell_width - 0.006,
            progress_height,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor=COLORS["panel"],
            edgecolor=COLORS["grid"],
            linewidth=1.2,
            zorder=5,
        )
        fig.add_artist(box)
        label = fig.text(
            progress_x + (index + 0.5) * cell_width,
            progress_y + progress_height / 2,
            f"{index + 1}  {profile_titles[run['profile']]}",
            color=COLORS["muted"],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        progress_boxes.append(box)
        progress_labels.append(label)

    def draw(_output_time: float, frame_idx: int, _count: int) -> None:
        profile_index = min(frame_idx // (10 * FPS), len(runs) - 1)
        local_frame = frame_idx % (10 * FPS)
        local_time = (local_frame + 0.5) / FPS * PROFILE_PLAYBACK_RATE
        frame = frames[profile_index]
        run = runs[profile_index]
        _update_presentation_road(road, frame, local_time, sim.gap_max)

        sample_index = min(_frame_at(frame, local_time), len(frame))
        history = frame.iloc[:sample_index]
        ego_line.set_data(history["time_s"], history["v_ego"])
        lead_line.set_data(history["time_s"], history["v_lead"])
        cursor.set_xdata([local_time, local_time])

        profile = run["profile"]
        title.set_text(f"{profile_titles[profile]} lead profile")
        description.set_text(run["description"])
        video_clock.set_text(
            f"VIDEO {frame_idx / FPS:05.2f} / 50.00 s · 5× playback"
        )
        simulation_clock.set_text(f"SIMULATION {local_time:05.2f} / 50.00 s")
        gap = _value(frame, "gap", local_time)
        status_gap.set_text(
            f"Gap: {gap:.1f} m"
            + (" · simulation limit" if gap >= sim.gap_max - 1e-6 else "")
        )
        ttc_text, ttc_color = _ttc_status(_value(frame, "ttc_proxy", local_time))
        status_ttc.set_text(ttc_text)
        status_ttc.set_color(ttc_color)
        headway_text, headway_color = _headway_status(frame, local_time, sim)
        status_headway.set_text(headway_text)
        status_headway.set_color(headway_color)
        row_index = min(_frame_at(frame, local_time), len(frame) - 1)
        regime = str(frame.iloc[row_index]["regime"]).replace("_", " ").title()
        status_context.set_text(f"Context: {regime}")

        summary = run["summary"]
        episode_result.set_text(
            f"Recorded episode: {summary['headway_violation_steps']} headway < "
            f"{sim.safe_headway:.1f} s steps · {summary['ttc_violation_steps']} "
            f"TTC < 3 s steps · {summary['collision_steps']} collisions"
        )
        for index, (box, label) in enumerate(zip(progress_boxes, progress_labels)):
            active = index == profile_index
            box.set_facecolor("#24405e" if active else COLORS["panel"])
            box.set_edgecolor(COLORS["ego"] if active else COLORS["grid"])
            box.set_linewidth(2.2 if active else 1.2)
            label.set_color(COLORS["text"] if active else COLORS["muted"])

    _write_video(
        fig,
        frames,
        PRESENTATION_PROFILE_OUTPUT,
        draw,
        duration_s=50.0,
        frame_count=PRESENTATION_FRAMES,
        dynamic_artists=[
            *_presentation_artists(road), ego_line, lead_line, cursor,
            title, description, video_clock, simulation_clock, status_gap,
            status_ttc, status_headway, status_context, episode_result,
            limitation, *progress_boxes, *progress_labels,
        ],
    )
    _write_json(DATA_DIR / "animation1_presentation_render.json", {
        "output": str(PRESENTATION_PROFILE_OUTPUT.relative_to(ROOT)),
        "fps": FPS,
        "duration_video_seconds": 50.0,
        "frame_count": PRESENTATION_FRAMES,
        "dimensions": [1920, 1080],
        "playback_rate": "5x; each 10 s scene resamples its full 0–50 s recorded episode",
        "profile_scene_video_seconds": 10.0,
        "profile_scene_simulation_seconds": 50.0,
        "profile_order": profile_order,
        "frame_time_mapping": (
            "Each output frame represents a 1/30 s video interval; its midpoint "
            "samples the corresponding recorded simulation time at 5x."
        ),
        "longitudinal_display": (
            "Linear 1:1 metre scale; the simulation's configured 100 m gap limit "
            "is explicitly labeled when reached. No renderer gap cap is applied."
        ),
    })


def _render_comparison_video(metadata: Dict[str, Any]) -> None:
    runs = metadata["animation2"]
    left = _load_frame(ROOT / runs["trajectories"]["context_ppo"])
    right = _load_frame(ROOT / runs["trajectories"]["hybrid_context_pid"])
    sim = Config().sim
    paired = runs["paired_conditions"]
    if not all(paired[key] for key in (
        "same_seed", "same_scenario", "same_lead_profile",
        "same_initial_conditions", "same_lead_speed_through_overlap",
        "same_context_through_overlap",
    )):
        raise ValueError("Refusing direct comparison: recorded conditions do not match.")
    if not np.allclose(
        left["time_s"].to_numpy(dtype=float),
        right["time_s"].to_numpy(dtype=float),
    ):
        raise ValueError("Comparison trajectories do not share an identical time grid.")

    max_time = min(float(left["time_s"].max()), float(right["time_s"].max()))
    gap_limit = sim.gap_max
    x_max = max(
        gap_limit,
        float(left["gap"].max()),
        float(right["gap"].max()),
    ) + 17.0
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=COLORS["background"])
    axes_road = [
        fig.add_axes([0.045, 0.565, 0.445, 0.24]),
        fig.add_axes([0.51, 0.565, 0.445, 0.24]),
    ]
    road_left = _presentation_road(
        axes_road[0],
        "Context-aware PPO",
        "PPO checkpoint without risk-governed PID supervision",
        x_max,
        ego_color=COLORS["baseline"],
    )
    road_right = _presentation_road(
        axes_road[1],
        "Risk-governed context-aware control",
        "Hybrid Context-PID · same PPO checkpoint · PID supervision/filter",
        x_max,
        ego_color=COLORS["governed"],
    )
    for road, color in ((road_left, COLORS["baseline"]), (road_right, COLORS["governed"])):
        road["title"].set_color(color)
        road["title"].set_fontsize(17)
        road["subtitle"].set_fontsize(11)
        road["title"].set_visible(False)
        road["subtitle"].set_visible(False)
        road["scale_note"].set_text("Measured gap [m]")
        road["scale_note"].set_position((0.02, 0.035))
        road["scale_note"].set_ha("left")

    left_panel_title = fig.text(
        0.055, 0.858, "Context-aware PPO",
        color=COLORS["baseline"], fontsize=17, fontweight="bold", va="top",
    )
    right_panel_title = fig.text(
        0.52, 0.858, "Risk-governed context-aware control",
        color=COLORS["governed"], fontsize=17, fontweight="bold", va="top",
    )
    fig.text(
        0.055, 0.83, "Same PPO checkpoint · no risk-governed PID supervisor",
        color=COLORS["muted"], fontsize=11.5, va="top",
    )
    fig.text(
        0.52, 0.83,
        "Hybrid Context-PID · same checkpoint · PID supervision/safety filtering",
        color=COLORS["muted"], fontsize=11.5, va="top",
    )

    frame_borders = []
    for road, color in ((road_left, COLORS["baseline"]), (road_right, COLORS["governed"])):
        border = FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=road["ax"].transAxes,
            boxstyle="square,pad=0",
            facecolor="none",
            edgecolor=color,
            linewidth=2.0,
            zorder=20,
            clip_on=False,
        )
        road["ax"].add_patch(border)
        frame_borders.append(border)

    chart_ax = fig.add_axes([0.09, 0.13, 0.82, 0.28])
    _style_presentation_chart(
        chart_ax, "TTC proxy [s]", "Simulation time [s]"
    )
    chart_ax.set_xlim(0.0, max_time)
    chart_ax.set_ylim(0.0, 12.0)
    chart_ax.set_xticks(np.arange(0.0, max_time + 0.01, 5.0))
    chart_ax.set_yticks(np.arange(0.0, 13.0, 3.0))
    chart_ax.axhline(
        3.0, color=COLORS["risk"], linestyle="--", linewidth=2.5,
        label="3 s TTC threshold", zorder=2,
    )
    left_line, = chart_ax.plot(
        [], [], color=COLORS["baseline"], linewidth=3.6,
        label="Context-aware PPO", zorder=4,
    )
    right_line, = chart_ax.plot(
        [], [], color=COLORS["governed"], linewidth=3.6, linestyle="--",
        label="Risk-governed control", zorder=4,
    )
    cursor = chart_ax.axvline(
        0.0, color="#ffffff", linestyle=":", linewidth=2.0, alpha=0.9, zorder=5
    )
    chart_ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
        fontsize=12,
        ncols=2,
    )

    left_ttc = left["ttc_proxy"].to_numpy(dtype=float)
    right_ttc = right["ttc_proxy"].to_numpy(dtype=float)
    left_times = left["time_s"].to_numpy(dtype=float)
    right_times = right["time_s"].to_numpy(dtype=float)
    left_closing = np.where(left_ttc >= 998.0, np.nan, np.minimum(left_ttc, 12.0))
    right_closing = np.where(right_ttc >= 998.0, np.nan, np.minimum(right_ttc, 12.0))
    left_line.set_data([], [])
    right_line.set_data([], [])
    left_overflow_mask = (left_ttc > 12.0) & (left_ttc < 998.0)
    right_overflow_mask = (right_ttc > 12.0) & (right_ttc < 998.0)
    left_overflow_indices = np.flatnonzero(left_overflow_mask)
    right_overflow_indices = np.flatnonzero(right_overflow_mask)
    if len(left_overflow_indices) > 12:
        left_overflow_indices = left_overflow_indices[
            np.linspace(0, len(left_overflow_indices) - 1, 12, dtype=int)
        ]
    if len(right_overflow_indices) > 12:
        right_overflow_indices = right_overflow_indices[
            np.linspace(0, len(right_overflow_indices) - 1, 12, dtype=int)
        ]
    left_overflow = chart_ax.scatter(
        [], [], marker="^", s=54, color=COLORS["baseline"], zorder=6
    )
    right_overflow = chart_ax.scatter(
        [], [], marker="^", s=54, color=COLORS["governed"], zorder=6
    )
    overflow_note = fig.text(
        0.095, 0.435, "▲ Finite TTC >12 s marked at chart ceiling · not-closing samples omitted",
        color=COLORS["muted"], fontsize=11.5, va="top",
    )

    max_time = max(float(left["time_s"].max()), float(right["time_s"].max()))
    chart_ax.set_xlim(0.0, max_time)
    crossing = left.loc[left["ttc_proxy"] < 3.0, "time_s"]
    critical_time = float(crossing.iloc[0]) if len(crossing) else None
    if critical_time is None:
        raise ValueError("The recorded comparison has no Context-aware PPO TTC violation.")
    event_line = chart_ax.axvline(
        critical_time, color=COLORS["risk"], linestyle="-.", linewidth=2.3,
        visible=False, zorder=3,
    )
    event_point = chart_ax.scatter(
        [], [], s=120, color=COLORS["risk"], edgecolor="#ffffff",
        linewidth=1.2, zorder=8,
    )
    event_label = chart_ax.annotate(
        f"First TTC < 3 s at {critical_time:.1f} s",
        xy=(critical_time, float(left.loc[left["time_s"] == critical_time, "ttc_proxy"].iloc[0])),
        xytext=(critical_time + 2.2, 5.4),
        textcoords="data",
        color=COLORS["text"],
        fontsize=13,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["risk"], "lw": 2.0},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": COLORS["panel"],
              "edgecolor": COLORS["risk"], "alpha": 0.98},
        zorder=9,
    )
    event_label.set_visible(False)
    warning = road_left["ax"].text(
        0.5, 0.22, "TTC below 3 s",
        transform=road_left["ax"].transAxes,
        color=COLORS["text"],
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": COLORS["risk"],
              "edgecolor": "#ffffff", "linewidth": 1.3},
        zorder=25,
    )
    warning.set_visible(False)
    warning.set_animated(True)

    title = fig.text(
        0.05, 0.975, "Context-aware PPO vs risk-governed control",
        color=COLORS["text"], fontsize=28, fontweight="bold", va="top",
    )
    subtitle = fig.text(
        0.05, 0.925,
        "Illustrative matched scenario · same initial conditions, lead trajectory and context",
        color=COLORS["muted"], fontsize=16, va="top",
    )
    clock = fig.text(
        0.95, 0.975, "", color=COLORS["context"], fontsize=15,
        fontweight="bold", ha="right", va="top",
    )
    left_current = fig.text(
        0.05, 0.535, "", color=COLORS["text"], fontsize=12.5, fontweight="bold",
    )
    right_current = fig.text(
        0.515, 0.535, "", color=COLORS["text"], fontsize=12.5, fontweight="bold",
    )
    left_persistent = fig.text(
        0.05, 0.505, "", color=COLORS["muted"], fontsize=12.5, fontweight="bold",
    )
    right_persistent = fig.text(
        0.515, 0.505, "", color=COLORS["muted"], fontsize=12.5, fontweight="bold",
    )
    left_outcome = fig.text(
        0.05, 0.475, "", color=COLORS["text"], fontsize=11.5, fontweight="bold",
    )
    right_outcome = fig.text(
        0.515, 0.475, "", color=COLORS["text"], fontsize=11.5, fontweight="bold",
    )
    limitation = fig.text(
        0.05, 0.045, "", color=COLORS["muted"], fontsize=12.5, va="bottom",
    )
    left_summary = runs["context_ppo_summary"]
    right_summary = runs["hybrid_context_pid_summary"]
    left_outcome.set_text(
        "Outcome: TTC threshold violated; no collision"
        f" · Headway <{sim.safe_headway:.1f} s: {left_summary['headway_violation_steps']} steps"
    )
    right_outcome.set_text(
        "Outcome: No TTC violations or collisions in this episode"
        f" · Headway <{sim.safe_headway:.1f} s: {right_summary['headway_violation_steps']} steps"
    )
    left_progress = float(left_summary["episode_metrics"]["progress"])
    right_progress = float(right_summary["episode_metrics"]["progress"])
    limitation.set_text(
        "Recorded progress: "
        f"Context-aware PPO {left_progress:.1f} m · risk-governed {right_progress:.1f} m"
        "   |   Illustrative recorded episodes; not a general performance guarantee."
    )

    def draw(time_s: float, _index: int, _count: int) -> None:
        for frame, road in ((left, road_left), (right, road_right)):
            t = min(time_s, float(frame["time_s"].max()))
            _update_presentation_road(road, frame, t, gap_limit)

        left_count = _frame_at(left, time_s)
        right_count = _frame_at(right, time_s)
        left_line.set_data(left_times[:left_count], left_closing[:left_count])
        right_line.set_data(right_times[:right_count], right_closing[:right_count])
        left_visible_overflow = left_overflow_indices[
            left_times[left_overflow_indices] <= time_s
        ]
        right_visible_overflow = right_overflow_indices[
            right_times[right_overflow_indices] <= time_s
        ]
        left_overflow.set_offsets(
            np.column_stack((
                left_times[left_visible_overflow],
                np.full(len(left_visible_overflow), 11.75),
            )) if len(left_visible_overflow) else np.empty((0, 2))
        )
        right_overflow.set_offsets(
            np.column_stack((
                right_times[right_visible_overflow],
                np.full(len(right_visible_overflow), 11.35),
            )) if len(right_visible_overflow) else np.empty((0, 2))
        )
        cursor.set_xdata([time_s, time_s])
        clock.set_text(f"SIMULATION {time_s:04.1f} / {max_time:.1f} s")

        for label, frame, time_value in (
            (left_current, left, time_s),
            (right_current, right, time_s),
        ):
            ttc_text, ttc_color = _ttc_status(_value(frame, "ttc_proxy", time_value))
            headway_text, headway_color = _headway_status(frame, time_value, sim)
            label.set_text(f"{ttc_text}   |   {headway_text}")
            label.set_color(
                COLORS["risk"] if ttc_color == COLORS["risk"] or headway_color == COLORS["risk"]
                else COLORS["text"]
            )

        violation_has_occurred = time_s >= critical_time
        left_persistent.set_text(
            "TTC violation occurred" if violation_has_occurred
            else "No TTC violation observed yet"
        )
        left_persistent.set_color(
            COLORS["risk"] if violation_has_occurred else COLORS["muted"]
        )

        sample_index = min(_frame_at(right, time_s), len(right) - 1)
        sample = right.iloc[sample_index]
        interventions = []
        for field, label in (
            ("hybrid_pid_active", "PID supervision"),
            ("hybrid_filter_active", "Safety filter"),
        ):
            if field in right.columns:
                state = "active" if float(sample[field]) > 0.5 else "inactive"
                interventions.append(f"{label}: {state}")
        right_persistent.set_text(" · ".join(interventions))
        right_persistent.set_color(
            COLORS["governed"] if any(item.endswith(": active") for item in interventions)
            else COLORS["muted"]
        )

        event_now = critical_time <= time_s < critical_time + 1.0
        warning.set_visible(event_now)
        event_line.set_visible(violation_has_occurred)
        event_label.set_visible(violation_has_occurred)
        event_point.set_offsets(
            np.array([[
                critical_time,
                float(left.loc[left["time_s"] == critical_time, "ttc_proxy"].iloc[0]),
            ]]) if violation_has_occurred else np.empty((0, 2))
        )
        frame_borders[0].set_edgecolor(
            COLORS["risk"] if event_now else COLORS["baseline"]
        )
        frame_borders[0].set_linewidth(4.0 if event_now else 2.0)

    _write_video(
        fig,
        [left, right],
        PRESENTATION_COMPARISON_OUTPUT,
        draw,
        duration_s=max_time,
        frame_count=int(round(max_time * FPS)),
        dynamic_artists=[
            *_presentation_artists(road_left), *_presentation_artists(road_right),
            frame_borders[0], frame_borders[1], left_line, right_line, cursor,
            left_overflow, right_overflow, overflow_note, event_line, event_point,
            event_label, warning, title, subtitle, clock, left_current,
            right_current, left_persistent, right_persistent, left_outcome,
            right_outcome, limitation,
        ],
    )
    _write_json(DATA_DIR / "animation2_presentation_render.json", {
        "output": str(PRESENTATION_COMPARISON_OUTPUT.relative_to(ROOT)),
        "fps": FPS,
        "duration_seconds": max_time,
        "frame_count": int(round(max_time * FPS)),
        "dimensions": [1920, 1080],
        "playback": "30 fps; synchronized simulation-time frame midpoints",
        "terminal_state_hold": "A policy ending early is held at its terminal state and explicitly labeled.",
        "first_context_ppo_ttc_violation_time_s": critical_time,
        "context_ppo_ttc_violation_steps": left_summary["ttc_violation_steps"],
        "risk_governed_ttc_violation_steps": right_summary["ttc_violation_steps"],
        "context_ppo_headway_violation_steps": left_summary["headway_violation_steps"],
        "risk_governed_headway_violation_steps": right_summary["headway_violation_steps"],
        "context_ppo_collision_steps": left_summary["collision_steps"],
        "risk_governed_collision_steps": right_summary["collision_steps"],
        "pid_supervision_field": "hybrid_pid_active",
        "safety_filter_field": "hybrid_filter_active",
        "measured_episode_progress_m": {
            "context_aware_ppo": left_progress,
            "risk_governed_context_aware_control": right_progress,
        },
        "criterion": runs["criterion"],
        "status": runs["status"],
    })


def _render() -> None:
    metadata_path = DATA_DIR / "animation_results.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Record trajectories before rendering: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    _render_profile_video(metadata)
    _render_comparison_video(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record", help="Record five profiles and search matched comparisons.")
    record.add_argument("--training-seed", type=int, default=0)
    subparsers.add_parser("render", help="Render MP4s from the saved trajectory CSVs.")
    args = parser.parse_args()
    if args.command == "record":
        _run_recording(args.training_seed)
    elif args.command == "render":
        _render()


if __name__ == "__main__":
    main()
