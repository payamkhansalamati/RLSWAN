"""
plots.py
========
All visualisation routines for the car-following case study.

Generated figures (saved as PNG to artifacts/plots/)
------------------------------------------------------
1. training_curve_<agent_type>.png  – per-agent mean episode return vs steps
2. trajectory_<agent_type>.png      – 6-panel time-series plot
3. bar_metrics.png                  – grouped bar chart of key raw metrics
4. radar_scores.png                 – radar / spider chart (Safety, Comfort,
                                       Performance, Stability)
5. scenario_comparison.png          – performance broken down by scenario

All functions are self-contained and can be called independently.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for all systems
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from config import Config
from utils import agent_display_name, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

AGENT_COLORS = {
    "pid":             "#4C72B0",
    "vanilla_ppo":     "#DD8452",
    "context_ppo":     "#55A868",
    "hybrid_context_pid":"#2CA9C2",
    "ablation_state":  "#C44E52",
    "ablation_reward": "#8172B2",
}

AGENT_HATCHES = {
    "pid": "",
    "vanilla_ppo": "",
    "context_ppo": "",
    "hybrid_context_pid": "///",
    "ablation_state": "xx",
    "ablation_reward": "..",
}

AGENT_ORDER = [
    "pid",
    "vanilla_ppo",
    "context_ppo",
    "hybrid_context_pid",
    "ablation_state",
    "ablation_reward",
]
SHOWCASE_AGENT_ORDER = ["pid", "vanilla_ppo", "context_ppo"]

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "whitesmoke",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
}


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved plot → {path}")


def _style_bar_container(bars, agents: List[str]) -> None:
    """Apply consistent edge and hatch styling per agent."""
    for bar, agent in zip(bars, agents):
        bar.set_linewidth(0.8)
        bar.set_edgecolor("white")
        bar.set_hatch(AGENT_HATCHES.get(agent, ""))


def _remove_stale_plot_files(config: Config, expected_files: List[str]) -> None:
    """Delete stale PNG files in the plots directory that will not be regenerated."""
    plots_dir = config.paths.plots
    plots_dir.mkdir(parents=True, exist_ok=True)
    expected = set(expected_files)
    for path in plots_dir.glob("*.png"):
        if path.name not in expected:
            path.unlink(missing_ok=True)
            logger.info(f"  Removed stale plot → {path}")


def _bar_bottom_for_negative_series(means: List[float], stds: Optional[List[float]] = None) -> Optional[float]:
    """
    Return a lower plotting baseline when all bars are non-positive.

    This makes negative-valued bars grow upward from the bottom of the axis
    instead of downward from zero, which is visually easier to read.
    """
    if not means:
        return None
    if not all(np.isfinite(means)):
        return None
    if max(means) > 0.0:
        return None

    lower = min(means)
    if stds is not None and len(stds) == len(means):
        finite_stds = [s for s in stds if np.isfinite(s)]
        if finite_stds:
            lower = min(m - s for m, s in zip(means, stds))

    pad = max(20.0, 0.08 * max(abs(lower), 1.0))
    return lower - pad


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curve(
    agent_type: str,
    config: Config,
) -> None:
    """
    Plot mean episode return vs timestep for all seeds of a given agent.

    Reads SB3 monitor CSV files from
      artifacts/models/<agent_type>_seed<N>/evaluations.npz
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    seeds = config.ppo.seeds
    color = AGENT_COLORS.get(agent_type, "steelblue")
    all_steps: List[np.ndarray] = []
    all_rets:  List[np.ndarray] = []

    for seed in seeds:
        model_dir = config.paths.models / f"{agent_type}_seed{seed}"
        eval_npz = model_dir / "evaluations.npz"
        if not eval_npz.exists():
            continue
        data = np.load(str(eval_npz))
        timesteps = data["timesteps"]
        results   = data["results"]       # shape (n_evals, n_ep_per_eval)
        means = results.mean(axis=1)
        all_steps.append(timesteps)
        all_rets.append(means)
        ax.plot(timesteps, means, alpha=0.3, lw=1, color=color)

    if all_steps:
        # Interpolate onto a common x-axis and plot mean ± std band
        max_len = max(len(s) for s in all_steps)
        min_steps = min(s[0] for s in all_steps)
        max_steps = max(s[-1] for s in all_steps)
        x_common = np.linspace(min_steps, max_steps, 200)
        interp_matrix = []
        for xs, ys in zip(all_steps, all_rets):
            interp_matrix.append(np.interp(x_common, xs, ys))
        interp_matrix = np.array(interp_matrix)
        mean_curve = interp_matrix.mean(axis=0)
        std_curve  = interp_matrix.std(axis=0)
        ax.plot(x_common, mean_curve, lw=2.0, color=color, label=f"Mean ({len(seeds)} seeds)")
        ax.fill_between(x_common, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color=color, label="±1 std")
    else:
        ax.text(0.5, 0.5, "No evaluation data found.\nRun training first.",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)

    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title(f"Training Curve – {agent_display_name(agent_type)}")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    fig.tight_layout()

    out_path = config.paths.plots / f"training_curve_{agent_type}.png"
    _savefig(fig, out_path)


def plot_all_training_curves(config: Config) -> None:
    """Generate training curve plots for all agents."""
    from train import AGENT_CONFIGS
    for agent_type in AGENT_CONFIGS:
        plot_training_curve(agent_type, config)


# ---------------------------------------------------------------------------
# 2. Trajectory time-series plots
# ---------------------------------------------------------------------------

def plot_trajectory(
    trajectory: List[Dict[str, Any]],
    agent_type: str,
    config: Config,
    title_suffix: str = "",
) -> None:
    """
    6-panel time-series plot for one episode trajectory.

    Panels:
      1. Gap (m)
      2. Speed – ego & lead (m/s)
      3. Acceleration (m/s²)
      4. Jerk (m/s³)
      5. Context: stress & fatigue
      6. Context: preferred headway
    """
    _apply_style()
    dt = config.sim.dt
    T = len(trajectory)
    time = np.arange(T) * dt

    keys = ["gap", "v_ego", "v_lead", "accel", "jerk",
            "context_stress", "context_fatigue", "context_h_pref"]
    arrays = {k: np.array([s.get(k, 0.0) for s in trajectory]) for k in keys}

    color = AGENT_COLORS.get(agent_type, "steelblue")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    # Panel 0: Gap
    axes[0].plot(time, arrays["gap"], color=color, lw=1.5)
    axes[0].axhline(config.sim.safe_headway * config.sim.target_speed,
                    ls="--", color="red", lw=0.8, label="Min-safe gap proxy")
    axes[0].set_ylabel("Gap [m]")
    axes[0].set_title("Distance to Lead Vehicle")
    axes[0].legend(fontsize=8)

    # Panel 1: Speed
    axes[1].plot(time, arrays["v_ego"],  color=color, lw=1.5, label="Ego")
    axes[1].plot(time, arrays["v_lead"], color="grey", lw=1.2, ls="--", label="Lead")
    axes[1].set_ylabel("Speed [m/s]")
    axes[1].set_title("Vehicle Speeds")
    axes[1].legend()

    # Panel 2: Acceleration
    axes[2].plot(time, arrays["accel"], color=color, lw=1.5)
    axes[2].axhline(0, color="black", lw=0.5)
    axes[2].set_ylabel("Accel [m/s²]")
    axes[2].set_title("Ego Acceleration")

    # Panel 3: Jerk
    axes[3].plot(time, arrays["jerk"], color=color, lw=1.2, alpha=0.85)
    axes[3].axhline(0, color="black", lw=0.5)
    axes[3].set_ylabel("Jerk [m/s³]")
    axes[3].set_title("Ego Jerk")

    # Panel 4: Context – stress & fatigue
    axes[4].plot(time, arrays["context_stress"],  color="#e74c3c", lw=1.5, label="Stress")
    axes[4].plot(time, arrays["context_fatigue"], color="#9b59b6", lw=1.5, ls="--", label="Fatigue")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("Level [0-1]")
    axes[4].set_title("Inferred Driver State")
    axes[4].legend()
    axes[4].set_xlabel("Time [s]")

    # Panel 5: Preferred headway
    axes[5].plot(time, arrays["context_h_pref"], color="#1abc9c", lw=1.5)
    axes[5].axhline(config.sim.safe_headway, ls="--", color="red", lw=0.8, label="Min safe")
    axes[5].set_ylabel("Headway [s]")
    axes[5].set_title("Preferred Time Headway")
    axes[5].legend()
    axes[5].set_xlabel("Time [s]")

    suf = f" – {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Episode Trajectory: {agent_display_name(agent_type)}{suf}", fontsize=13)
    fig.tight_layout()

    out_path = config.paths.plots / f"trajectory_{agent_type}.png"
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Bar charts
# ---------------------------------------------------------------------------

def plot_bar_metrics(
    config: Config,
) -> None:
    """
    Grouped bar chart comparing key raw metrics across agents.
    Reads aggregate_metrics.csv from artifacts/results/.
    """
    csv_path = config.paths.results / "aggregate_metrics.csv"
    if not csv_path.exists():
        logger.warning(f"Aggregate metrics CSV not found: {csv_path}")
        return

    _apply_style()
    df = pd.read_csv(csv_path)
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]

    metrics_to_plot = [
        ("collision_rate_mean",        "collision_rate_std",        "Collision Rate",         False),
        ("headway_violation_pct_mean", "headway_violation_pct_std", "HW Violation %",         False),
        ("mean_abs_jerk_mean",         "mean_abs_jerk_std",         "Mean |Jerk| [m/s³]",     False),
        ("speed_tracking_error_mean",  "speed_tracking_error_std",  "Speed Error [m/s]",      False),
        ("episode_return_mean",        "episode_return_std",        "Episode Return",          True),
        ("min_gap_mean",               "min_gap_std",               "Min Gap [m]",             True),
    ]

    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 4.2 * n_rows))
    axes = axes.flatten()

    x = np.arange(len(agents))
    width = 0.55

    for idx, (m_col, s_col, label, higher_better) in enumerate(metrics_to_plot):
        ax = axes[idx]
        colors = [AGENT_COLORS.get(a, "steelblue") for a in agents]
        means = [df.loc[df["agent"] == a, m_col].values[0] if m_col in df.columns else 0.0
                 for a in agents]
        stds  = [df.loc[df["agent"] == a, s_col].values[0] if s_col in df.columns else 0.0
                 for a in agents]
        bottom = _bar_bottom_for_negative_series(means, stds)
        if bottom is None:
            heights = means
            bar_bottom = None
        else:
            heights = [m - bottom for m in means]
            bar_bottom = bottom

        bars = ax.bar(
            x,
            heights,
            width,
            bottom=bar_bottom,
            yerr=stds,
            color=colors,
            capsize=4,
            alpha=0.85,
            edgecolor="white",
        )
        _style_bar_container(bars, agents)
        ax.set_xticks(x)
        ax.set_xticklabels([agent_display_name(a) for a in agents],
                           rotation=16, ha="right", fontsize=8.5)
        ax.set_title(label)
        ax.set_ylabel(label)
        # Annotate with value
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Aggregate Metrics Comparison (mean ± std)", fontsize=14)
    fig.tight_layout()
    out_path = config.paths.plots / "bar_metrics.png"
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Radar chart
# ---------------------------------------------------------------------------

def plot_radar(config: Config) -> None:
    """
    Spider / radar chart comparing PID, vanilla_ppo, context_ppo, ablation_state
    on normalised Safety, Comfort, Performance, Stability scores.
    Reads radar_scores.csv from artifacts/results/.
    """
    csv_path = config.paths.results / "radar_scores.csv"
    if not csv_path.exists():
        logger.warning(f"Radar scores CSV not found: {csv_path}")
        return

    _apply_style()
    df = pd.read_csv(csv_path)
    dims = ["Safety", "Comfort", "Performance", "Stability"]
    agents = [a for a in AGENT_ORDER if a in df["agent"].values]

    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]    # close the polygon

    fig, ax = plt.subplots(figsize=(6.8, 6.4), subplot_kw=dict(polar=True))

    for agent in agents:
        row = df[df["agent"] == agent]
        if row.empty:
            continue
        values = [float(row[d].values[0]) for d in dims]
        values += values[:1]
        color = AGENT_COLORS.get(agent, "steelblue")
        line_style = "--" if agent == "hybrid_context_pid" else "-"
        ax.plot(angles, values, "o", lw=2.4, ls=line_style, color=color, label=agent_display_name(agent))
        ax.fill(angles, values, alpha=0.10 if agent == "hybrid_context_pid" else 0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Normalised Performance Radar\n(higher = better)", pad=20, fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.08), fontsize=9, frameon=True)
    ax.grid(color="grey", alpha=0.3)

    fig.tight_layout()
    out_path = config.paths.plots / "radar_scores.png"
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Scenario comparison bar chart
# ---------------------------------------------------------------------------

def plot_scenario_comparison(config: Config) -> None:
    """
    Bar chart comparing episode return across scenarios for each agent.
    Uses episode-level CSV files.
    """
    _apply_style()
    agent_dfs: Dict[str, pd.DataFrame] = {}
    for agent_type in AGENT_ORDER:
        csv_path = config.paths.results / f"episode_metrics_{agent_type}.csv"
        if csv_path.exists():
            agent_dfs[agent_type] = pd.read_csv(csv_path)

    if not agent_dfs:
        logger.warning("No episode CSVs found; skipping scenario comparison plot.")
        return

    scenarios = list(pd.concat(agent_dfs.values())["scenario"].unique())
    agents = [a for a in AGENT_ORDER if a in agent_dfs]

    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(agents))

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    for i, agent_type in enumerate(agents):
        df = agent_dfs[agent_type]
        means, stds = [], []
        for sc in scenarios:
            sub = df[df["scenario"] == sc]["episode_return"]
            means.append(sub.mean() if len(sub) > 0 else 0.0)
            stds.append(sub.std()  if len(sub) > 1 else 0.0)
        offset = (i - len(agents) / 2 + 0.5) * width
        bottom = _bar_bottom_for_negative_series(means, stds)
        if bottom is None:
            heights = means
            bar_bottom = None
        else:
            heights = [m - bottom for m in means]
            bar_bottom = bottom
        bars = ax.bar(
            x + offset,
            heights,
            width,
            bottom=bar_bottom,
            yerr=stds,
            capsize=3,
            label=agent_display_name(agent_type),
            color=AGENT_COLORS.get(agent_type, "steelblue"),
            alpha=0.85, edgecolor="white",
        )
        _style_bar_container(bars, [agent_type] * len(scenarios))

    ax.set_xticks(x)
    ax.set_xticklabels([sc.capitalize() for sc in scenarios])
    ax.set_xlabel("Evaluation Scenario")
    ax.set_ylabel("Mean Episode Return (mean ± std)")
    ax.set_title("Episode Return by Scenario and Agent")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True)
    fig.tight_layout()
    out_path = config.paths.plots / "scenario_comparison.png"
    _savefig(fig, out_path)


def plot_showcase_grouped_metrics(config: Config) -> None:
    """Create grouped bar charts for key showcase metrics by case and agent."""
    csv_path = config.paths.results / "showcase_episode_metrics.csv"
    if not csv_path.exists():
        logger.warning(f"Showcase episode CSV not found: {csv_path}")
        return

    _apply_style()
    df = pd.read_csv(csv_path)
    df = df[df["agent"].isin(SHOWCASE_AGENT_ORDER)]
    if df.empty:
        logger.warning("No showcase rows for PID/vanilla/context agents.")
        return

    metrics = [
        "collision_rate",
        "headway_violation_pct",
        "mean_abs_jerk",
        "headway_pref_error",
    ]
    metric_titles = {
        "collision_rate": "Collision Rate",
        "headway_violation_pct": "Headway Violation %",
        "mean_abs_jerk": "Mean |Jerk| [m/s^3]",
        "headway_pref_error": "Headway Preference Error [s]",
    }

    case_order = list(df["case_id"].drop_duplicates())
    n_cases = len(case_order)
    fig, axes = plt.subplots(n_cases, 1, figsize=(13, 3.2 * n_cases), squeeze=False)

    for idx, case_id in enumerate(case_order):
        ax = axes[idx, 0]
        case_df = df[df["case_id"] == case_id]
        case_display = str(case_df["case_display"].iloc[0])

        x = np.arange(len(metrics))
        width = 0.24
        for j, agent in enumerate(SHOWCASE_AGENT_ORDER):
            a_df = case_df[case_df["agent"] == agent]
            means = [a_df[m].mean() if m in a_df.columns and not a_df.empty else np.nan for m in metrics]
            stds = [a_df[m].std() if m in a_df.columns and len(a_df) > 1 else 0.0 for m in metrics]
            offset = (j - 1) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=AGENT_COLORS.get(agent, "steelblue"),
                alpha=0.88,
                label=agent_display_name(agent),
                edgecolor="white",
            )

        ax.set_title(f"{case_display}")
        ax.set_xticks(x)
        ax.set_xticklabels([metric_titles[m] for m in metrics], rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(ncol=3, fontsize=9)

    fig.suptitle("Showcase Key Metrics (mean +/- std)", fontsize=13)
    fig.tight_layout()
    out_path = config.paths.plots / "showcase_grouped_metrics.png"
    _savefig(fig, out_path)


def _extract_series(traj: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([step.get(key, np.nan) for step in traj], dtype=float)


def plot_showcase_case_trajectory(
    config: Config,
    case_id: str,
    case_display: str,
    trajectories_by_agent: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Plot aligned multi-panel trajectory comparison for one showcase case."""
    if not trajectories_by_agent:
        return

    _apply_style()
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=False)
    axes = axes.flatten()

    # Plot each agent where available.
    for agent in SHOWCASE_AGENT_ORDER:
        traj = trajectories_by_agent.get(agent)
        if not traj:
            continue
        t = np.arange(len(traj)) * config.sim.dt
        color = AGENT_COLORS.get(agent, "steelblue")
        label = agent_display_name(agent)

        axes[0].plot(t, _extract_series(traj, "gap"), color=color, lw=1.5, label=label)
        axes[1].plot(t, _extract_series(traj, "v_ego"), color=color, lw=1.5, label=f"{label} ego")
        axes[2].plot(t, _extract_series(traj, "accel"), color=color, lw=1.3, label=label)
        axes[3].plot(t, _extract_series(traj, "jerk"), color=color, lw=1.1, label=label)
        axes[4].plot(t, _extract_series(traj, "context_h_pref"), color=color, lw=1.3, label=label)

    # Lead speed plotted once from any available trajectory.
    ref_traj = next(iter(trajectories_by_agent.values()))
    t_ref = np.arange(len(ref_traj)) * config.sim.dt
    axes[1].plot(t_ref, _extract_series(ref_traj, "v_lead"), color="black", lw=1.1, ls="--", label="Lead speed")

    # Stress/fatigue shown from context_ppo if available, otherwise first trajectory.
    sf_traj = trajectories_by_agent.get("context_ppo", ref_traj)
    t_sf = np.arange(len(sf_traj)) * config.sim.dt
    axes[5].plot(t_sf, _extract_series(sf_traj, "context_stress"), color="#e74c3c", lw=1.5, label="Stress")
    axes[5].plot(t_sf, _extract_series(sf_traj, "context_fatigue"), color="#8e44ad", lw=1.5, ls="--", label="Fatigue")

    axes[0].set_title("Gap")
    axes[0].set_ylabel("m")
    axes[1].set_title("Speeds")
    axes[1].set_ylabel("m/s")
    axes[2].set_title("Acceleration")
    axes[2].set_ylabel("m/s^2")
    axes[3].set_title("Jerk")
    axes[3].set_ylabel("m/s^3")
    axes[4].set_title("Preferred Headway")
    axes[4].set_ylabel("s")
    axes[5].set_title("Context Levels")
    axes[5].set_ylabel("0-1")

    for ax in axes:
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[5].legend(fontsize=8)

    fig.suptitle(f"Showcase Trajectory Comparison: {case_display}", fontsize=13)
    fig.tight_layout()

    out_map = {
        "case1_stressed_braking": "showcase_case1_stressed_braking.png",
        "case2_fatigued_stopgo": "showcase_case2_fatigued_stopgo.png",
        "case3_both_impairment": "showcase_case3_both.png",
        "case4_escalation": "showcase_escalation.png",
    }
    out_name = out_map.get(case_id, f"showcase_{case_id}.png")
    _savefig(fig, config.paths.plots / out_name)


def plot_showcase_escalation_adaptation(
    config: Config,
    trajectories_by_agent: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Visualize context transitions and action adaptation for escalation case."""
    if not trajectories_by_agent:
        return
    _apply_style()

    ref = trajectories_by_agent.get("context_ppo") or next(iter(trajectories_by_agent.values()))
    t = np.arange(len(ref)) * config.sim.dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, _extract_series(ref, "context_stress"), color="#e74c3c", lw=1.7, label="Stress")
    axes[0].plot(t, _extract_series(ref, "context_fatigue"), color="#8e44ad", lw=1.7, ls="--", label="Fatigue")
    axes[0].plot(t, _extract_series(ref, "context_h_pref") / max(config.sim.headway_pref_norm, 1e-6),
                 color="#16a085", lw=1.5, ls=":", label="Preferred headway (scaled)")
    axes[0].set_ylabel("Context level")
    axes[0].set_title("Deterministic Escalation Context")
    axes[0].legend()

    for agent in SHOWCASE_AGENT_ORDER:
        traj = trajectories_by_agent.get(agent)
        if not traj:
            continue
        t_a = np.arange(len(traj)) * config.sim.dt
        axes[1].plot(
            t_a,
            _extract_series(traj, "accel"),
            color=AGENT_COLORS.get(agent, "steelblue"),
            lw=1.4,
            label=agent_display_name(agent),
        )
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_title("Controller Acceleration Response")
    axes[1].set_ylabel("Acceleration [m/s^2]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()

    fig.tight_layout()
    _savefig(fig, config.paths.plots / "showcase_context_adaptation.png")


# ---------------------------------------------------------------------------
# Reward diagnostics plots (new)
# ---------------------------------------------------------------------------

def plot_reward_components_comparison(config: Config) -> None:
    """
    Compare reward components (base vs context) across all agents.
    
    Reads episode_metrics_<agent>.csv and creates:
      - Grouped bar chart of total base vs context contribution
      - Highlights whether context is serving as gentle shaper vs dominator
    """
    _apply_style()
    
    paths = config.paths
    agents_to_plot = [a for a in AGENT_ORDER if a != "pid"]
    data = None
    
    for agent_type in agents_to_plot:
        csv_path = paths.results / f"episode_metrics_{agent_type}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(str(csv_path))
        if data is None:
            data = {}
        
        # Compute aggregated metrics
        base_mean = df["total_base_reward"].mean() if "total_base_reward" in df.columns else 0.0
        context_mean = df["total_context_reward"].mean() if "total_context_reward" in df.columns else 0.0
        context_scale = df["avg_context_scale"].mean() if "avg_context_scale" in df.columns else 0.0
        
        data[agent_type] = {
            "base": base_mean,
            "context": context_mean,
            "scale": context_scale,
        }
    
    if not data:
        logger.warning("No episode metrics found for reward diagnostics plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    
    # Plot 1: Base vs Context reward magnitude
    agents = list(data.keys())
    base_rewards = [data[a]["base"] for a in agents]
    context_rewards = [abs(c) for c in [data[a]["context"] for a in agents]]
    
    x = np.arange(len(agents))
    width = 0.35
    
    bottom = _bar_bottom_for_negative_series(base_rewards)
    if bottom is None:
        base_heights = base_rewards
        base_bottom = None
    else:
        base_heights = [m - bottom for m in base_rewards]
        base_bottom = bottom

    bars_base = axes[0].bar(x - width/2, base_heights, width, bottom=base_bottom, label="Base Reward", color="#55A868")
    bars_ctx = axes[0].bar(x + width/2, context_rewards, width, label="Context Reward (abs)", color="#DD8452")
    _style_bar_container(bars_base, agents)
    _style_bar_container(bars_ctx, agents)
    axes[0].set_ylabel("Cumulative Reward per Episode")
    axes[0].set_title("Base vs Context Reward Magnitude")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([agent_display_name(a) for a in agents], rotation=15, ha="right")
    axes[0].legend()
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].grid(axis="y", alpha=0.3)
    
    # Plot 2: Context reward scale (should be small, ~0.15)
    context_scales = [data[a]["scale"] for a in agents]
    colors = ["#55A868" if 0.10 <= s <= 0.20 else "#DD8452" for s in context_scales]
    x1 = np.arange(len(agents))
    bars_scale = axes[1].bar(x1, context_scales, color=colors)
    _style_bar_container(bars_scale, agents)
    axes[1].axhline(0.15, color="green", lw=1.5, linestyle="--", label="Target scale (0.15)")
    axes[1].set_ylabel("Effective Context Reward Scale")
    axes[1].set_title("Context Reward Scale Magnitude\n(lower = gentler shaping)")
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels([agent_display_name(a) for a in agents], rotation=15, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    
    # Plot 3: Context contribution ratio
    context_contribution = []
    for agent in agents:
        base = data[agent]["base"]
        context = abs(data[agent]["context"])
        ratio = 100.0 * context / (abs(base) + 1e-6)
        context_contribution.append(min(ratio, 100.0))  # cap at 100% for display
    
    colors = ["#55A868" if r < 30 else "#C44E52" for r in context_contribution]
    x2 = np.arange(len(agents))
    bars_contrib = axes[2].bar(x2, context_contribution, color=colors)
    _style_bar_container(bars_contrib, agents)
    axes[2].axhline(30.0, color="orange", lw=1.5, linestyle="--", label="Warning: > 30%")
    axes[2].set_ylabel("Context Contribution [%]")
    axes[2].set_title("Context as % of Base Reward\n(lower = better balance)")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels([agent_display_name(a) for a in agents], rotation=15, ha="right")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    _savefig(fig, paths.plots / "reward_components_comparison.png")


def plot_reward_diagnostics(config: Config) -> None:
    """
    Generate comprehensive reward diagnostic plots.
    Calls plot_reward_components_comparison.
    """
    logger.info("Generating reward diagnostic plots...")
    plot_reward_components_comparison(config)


def generate_showcase_plots(
    config: Config,
    include_optional_escalation: bool = True,
    traj_seed: int = 700,
) -> None:
    """Generate showcase-specific charts and trajectory comparisons."""
    from evaluate import get_showcase_cases, record_showcase_trajectories

    config.paths.make_dirs()
    expected_files = [
        "showcase_grouped_metrics.png",
        "showcase_case1_stressed_braking.png",
        "showcase_case2_fatigued_stopgo.png",
        "showcase_case3_both.png",
    ]
    if include_optional_escalation:
        expected_files.extend([
            "showcase_escalation.png",
            "showcase_context_adaptation.png",
        ])
    _remove_stale_plot_files(config, expected_files + [p.name for p in config.paths.plots.glob("training_curve_*.png")] + [
        "bar_metrics.png",
        "radar_scores.png",
        "scenario_comparison.png",
        "reward_components_comparison.png",
        "trajectory_pid.png",
        "trajectory_vanilla_ppo.png",
        "trajectory_context_ppo.png",
        "trajectory_hybrid_context_pid.png",
    ])
    plot_showcase_grouped_metrics(config)

    cases = get_showcase_cases(include_optional_escalation=include_optional_escalation)
    trajs = record_showcase_trajectories(
        config=config,
        seed=traj_seed,
        agents=SHOWCASE_AGENT_ORDER,
        include_optional_escalation=include_optional_escalation,
    )

    for case_id, case_def in cases.items():
        case_trajs = trajs.get(case_id, {})
        if not case_trajs:
            continue
        plot_showcase_case_trajectory(
            config=config,
            case_id=case_id,
            case_display=case_def["display"],
            trajectories_by_agent=case_trajs,
        )

    if include_optional_escalation and "case4_escalation" in trajs:
        plot_showcase_escalation_adaptation(config, trajs["case4_escalation"])


# ---------------------------------------------------------------------------
# Master plot runner
# ---------------------------------------------------------------------------

def generate_all_plots(
    config: Config,
    trajectories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    """
    Generate ALL plots.

    Parameters
    ----------
    config:
        Master config.
    trajectories:
        Optional dict of agent_type → trajectory list for time-series plots.
        If not supplied, trajectory plots will not be generated here.
    """
    config.paths.make_dirs()

    expected_files = [
        "bar_metrics.png",
        "radar_scores.png",
        "scenario_comparison.png",
        "reward_components_comparison.png",
        "training_curve_vanilla_ppo.png",
        "training_curve_context_ppo.png",
        "training_curve_ablation_state.png",
        "training_curve_ablation_reward.png",
    ]
    if trajectories:
        expected_files.extend([
            f"trajectory_{agent_type}.png" for agent_type in trajectories.keys()
        ])
    _remove_stale_plot_files(config, expected_files)

    logger.info("Generating training curves...")
    plot_all_training_curves(config)

    logger.info("Generating bar metrics chart...")
    plot_bar_metrics(config)

    logger.info("Generating radar chart...")
    plot_radar(config)

    logger.info("Generating scenario comparison chart...")
    plot_scenario_comparison(config)

    logger.info("Generating reward diagnostic plots...")
    plot_reward_diagnostics(config)

    if trajectories:
        logger.info("Generating trajectory plots...")
        for agent_type, traj in trajectories.items():
            plot_trajectory(traj, agent_type, config, title_suffix="switching scenario")

    logger.info("All plots generated.")
