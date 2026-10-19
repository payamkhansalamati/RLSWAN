"""Evaluation pipeline for the final four-controller system.

Runs PID, vanilla_ppo, context_ppo, and hybrid_context_pid across
configured scenarios and seeds, computes metrics, and saves CSV results.

Metrics computed per episode
-----------------------------
Safety:
  - collision_rate         (1 if collision, else 0)
  - min_gap                [m]
  - headway_violation_pct  % of steps with headway < safe_headway threshold
  - mean_ttc_proxy         [s]  (capped)

Comfort:
  - mean_abs_jerk          [m/s³]
  - rms_jerk               [m/s³]
  - mean_abs_accel         [m/s²]

Performance:
  - speed_tracking_error   mean |v_ego - v_target|  [m/s]
  - episode_return         cumulative reward

Stability:
  - accel_oscillation      mean of |a[t] - a[t-1]|
  - jerk_variance          var(jerk)

All per-episode metrics are saved as CSVs.  Aggregated mean±std across
seeds/episodes are saved as summary CSVs and printed to console.

Usage
-----
    from evaluate import evaluate_all
    results = evaluate_all(config)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config, SimConfig
from controllers import PIDController, PPOController, run_pid_episode, HybridContextPIDController
from env_car_following import CarFollowingEnv, make_single_env
from registry import ALL_CONTROLLERS, TRAINABLE_CONTROLLERS, validate_controllers
from train import load_trained_model
from utils import (
    get_logger,
    set_global_seed,
    mean_std,
    compute_normalised_scores,
    aggregate_radar_scores,
    agent_display_name,
)

logger = get_logger(__name__)

# Evaluation scenarios and the initial regime they enforce
EVAL_SCENARIOS: Dict[str, str] = {
    "calm":      "calm",
    "stressed":  "stressed",
    "fatigued":  "fatigued",
    "switching": "mixed",     # free regime switching – hardest
}


SHOWCASE_CASES: Dict[str, Dict[str, Any]] = {
    "case1_stressed_braking": {
        "display": "Stressed Braking",
        "scenario": "stressed",
        "lead_profile": "braking",
        "hypothesis": "Context-aware PPO keeps larger buffer and fewer safety violations.",
    },
    "case2_fatigued_stopgo": {
        "display": "Fatigued Stop-and-Go",
        "scenario": "fatigued",
        "lead_profile": "stop_and_go",
        "hypothesis": "Context-aware PPO avoids aggressive re-acceleration and high jerk.",
    },
    "case3_both_impairment": {
        "display": "Combined Stress + Fatigue",
        "scenario": "both",
        "lead_profile": "random_smooth",
        "hypothesis": "Context-aware PPO is most conservative and context-compliant.",
    },
    "case4_escalation": {
        "display": "Switching Escalation",
        "scenario": "escalation",
        "lead_profile": "braking",
        "hypothesis": "Context-aware PPO adapts policy across deterministic context transitions.",
    },
}


METRIC_DIRECTIONS: Dict[str, bool] = {
    # True means higher is better; False means lower is better.
    "collision_rate": False,
    "min_gap": True,
    "headway_violation_pct": False,
    "mean_ttc_proxy": True,
    "ttc_violation_pct": False,
    "mean_abs_jerk": False,
    "rms_jerk": False,
    "mean_abs_accel": False,
    "speed_tracking_error": False,
    "progress": True,
    "episode_return": True,
    "accel_oscillation": False,
    "jerk_variance": False,
    "headway_pref_error": False,
    "high_stress_jerk": False,
    "fatigued_pos_accel": False,
    # Reward components
    "total_base_reward": True,
    "total_context_reward": True,
    "avg_context_scale": True,
    "context_contribution_pct": False,  # lower is better (context should be small contribution)
    # Hybrid supervisor diagnostics
    "hybrid_mean_raw_risk":  False,
    "hybrid_p95_raw_risk":   False,
    "hybrid_mean_ema_risk":  False,
    "hybrid_mean_req_brake_risk": False,
    "hybrid_p95_req_brake_risk": False,
    "hybrid_mean_pid_weight": False,
    "hybrid_mean_risk":      False,     # lower is better (less risk)
    "hybrid_pid_active_pct": False,     # lower is better (less PID override needed)
    "hybrid_pid_dominant_pct": False,
    "hybrid_emergency_takeover_pct": False,
    "hybrid_filter_active_pct": False,
    "hybrid_filter_clamped_pct": False,
    "hybrid_mean_action_delta_context_protected": False,
    "hybrid_mode_safe_pct": True,
    "hybrid_mode_caution_pct": False,
    "hybrid_mode_high_risk_pct": False,
    "hybrid_mode_emergency_pct": False,
    # Transition / escalation counts
    "hybrid_n_transitions":   False,
    "hybrid_n_escalations":   False,
    "hybrid_n_escal_sc":      False,
    "hybrid_n_escal_sh":      False,
    "hybrid_n_escal_se":      False,
    "hybrid_n_escal_ch":      False,
    "hybrid_n_escal_he":      False,
    "hybrid_n_deescalations": False,
    # Per-mode statistics
    "hybrid_mean_pid_wt_high":    False,
    "hybrid_mean_risk_raw_high":  False,
}


def get_showcase_cases(include_optional_escalation: bool = True) -> Dict[str, Dict[str, Any]]:
    """Return deterministic showcase cases used for context-sensitive benchmarking."""
    if include_optional_escalation:
        return dict(SHOWCASE_CASES)
    return {k: v for k, v in SHOWCASE_CASES.items() if k != "case4_escalation"}


# ---------------------------------------------------------------------------
# Per-episode metric computation
# ---------------------------------------------------------------------------

def compute_episode_metrics(
    trajectory: List[Dict[str, Any]],
    cfg_sim: SimConfig,
) -> Dict[str, float]:
    """
    Derive scalar metrics from a trajectory (list of info dicts).

    Parameters
    ----------
    trajectory:
        One entry per timestep, each a dict from env.step() info.
    cfg_sim:
        SimConfig for threshold values.

    Returns
    -------
    dict of metric name → float
    """
    if not trajectory:
        return {}

    gaps = np.array([s.get("gap", 0.0) for s in trajectory], dtype=float)
    v_ego = np.array([s.get("v_ego", 0.0) for s in trajectory], dtype=float)
    accels = np.array([s.get("accel", 0.0) for s in trajectory], dtype=float)
    jerks = np.array([s.get("jerk", 0.0) for s in trajectory], dtype=float)
    headways = np.array([s.get("headway", 0.0) for s in trajectory], dtype=float)
    ttcs = np.array([min(s.get("ttc_proxy", 999.0), 60.0) for s in trajectory], dtype=float)
    collisions = np.array([s.get("collision", 0) for s in trajectory], dtype=float)
    rewards = np.array([s.get("reward", 0.0) for s in trajectory], dtype=float)
    h_pref = np.array([s.get("context_h_pref", cfg_sim.safe_headway) for s in trajectory], dtype=float)
    stress = np.array([s.get("context_stress", 0.0) for s in trajectory], dtype=float)
    fatigue = np.array([s.get("context_fatigue", 0.0) for s in trajectory], dtype=float)
    
    # Reward components (new reward structure)
    base_rewards = np.array([s.get("base_reward", 0.0) for s in trajectory], dtype=float)
    context_rewards = np.array([s.get("context_reward", 0.0) for s in trajectory], dtype=float)
    context_rewards_scaled = np.array([s.get("context_reward_scaled", 0.0) for s in trajectory], dtype=float)
    context_scales = np.array([s.get("context_reward_scale", 0.0) for s in trajectory], dtype=float)

    # ---- Safety ----
    collision_rate = float(np.any(collisions > 0))
    min_gap = float(np.min(gaps))
    safe_hw = cfg_sim.safe_headway
    hw_violation_pct = float(100.0 * np.mean(headways < safe_hw))
    mean_ttc = float(np.mean(ttcs))
    ttc_violation_pct = float(100.0 * np.mean(ttcs < 3.0))

    # ---- Comfort ----
    mean_abs_jerk = float(np.mean(np.abs(jerks)))
    rms_jerk = float(np.sqrt(np.mean(jerks ** 2)))
    mean_abs_accel = float(np.mean(np.abs(accels)))

    # ---- Performance ----
    v_target = cfg_sim.target_speed
    speed_err = float(np.mean(np.abs(v_ego - v_target)))
    episode_return = float(np.sum(rewards))
    progress = float(np.sum(v_ego) * cfg_sim.dt)

    # ---- Stability ----
    accel_deltas = np.abs(np.diff(accels))
    accel_osc = float(np.mean(accel_deltas)) if len(accel_deltas) > 0 else 0.0
    jerk_var = float(np.var(jerks))

    # ---- Context compliance ----
    headway_pref_error = float(np.mean(np.abs(headways - h_pref)))
    high_stress_mask = stress >= 0.7
    fatigued_mask = fatigue >= 0.7
    high_stress_jerk = float(np.mean(np.abs(jerks[high_stress_mask]))) if np.any(high_stress_mask) else 0.0
    fatigued_positive_accel = float(np.mean(np.maximum(accels[fatigued_mask], 0.0))) if np.any(fatigued_mask) else 0.0

    # ---- Reward component diagnostics ----
    total_base_reward = float(np.sum(base_rewards))
    total_context_reward = float(np.sum(context_rewards_scaled))
    
    avg_context_scale = float(np.mean(context_scales)) if len(context_scales) > 0 else 0.0
    
    context_contribution_pct = 0.0
    if total_base_reward != 0.0:
        context_contribution_pct = float(
            100.0 * np.abs(total_context_reward) / (np.abs(total_base_reward) + 1e-6)
        )

    # ---- Hybrid supervisor diagnostics ----
    # These are non-zero only for hybrid_context_pid; other agents naturally
    # report zeros so multi-agent tables remain aligned.
    hybrid_risk_raw = np.array([s.get("hybrid_risk_raw", 0.0) for s in trajectory], dtype=float)
    hybrid_risk_ema = np.array([s.get("hybrid_risk_ema", s.get("hybrid_risk", 0.0)) for s in trajectory], dtype=float)
    hybrid_pid_weight = np.array([s.get("hybrid_pid_weight", 0.0) for s in trajectory], dtype=float)
    hybrid_pid_active = np.array([s.get("hybrid_pid_active", s.get("hybrid_in_pid_mode", 0.0)) for s in trajectory], dtype=float)
    hybrid_pid_dominant = np.array([s.get("hybrid_pid_dominant", 0.0) for s in trajectory], dtype=float)
    hybrid_emergency = np.array([s.get("hybrid_emergency_takeover", 0.0) for s in trajectory], dtype=float)
    hybrid_req_brake_risk = np.array([s.get("hybrid_risk_req_brake", 0.0) for s in trajectory], dtype=float)
    hybrid_filter_active = np.array([s.get("hybrid_filter_active", 0.0) for s in trajectory], dtype=float)
    hybrid_filter_clamped = np.array([s.get("hybrid_filter_clamped", 0.0) for s in trajectory], dtype=float)
    hybrid_action_delta_context = np.array([s.get("hybrid_action_delta_context", 0.0) for s in trajectory], dtype=float)

    hybrid_mode_safe = np.array([s.get("hybrid_mode_safe", 0.0) for s in trajectory], dtype=float)
    hybrid_mode_caution = np.array([s.get("hybrid_mode_caution", 0.0) for s in trajectory], dtype=float)
    hybrid_mode_high = np.array([s.get("hybrid_mode_high_risk", 0.0) for s in trajectory], dtype=float)
    hybrid_mode_emergency = np.array([s.get("hybrid_mode_emergency", 0.0) for s in trajectory], dtype=float)

    hybrid_mean_raw_risk = float(np.mean(hybrid_risk_raw))
    hybrid_p95_raw_risk = float(np.percentile(hybrid_risk_raw, 95))
    hybrid_mean_ema_risk = float(np.mean(hybrid_risk_ema))
    hybrid_mean_req_brake_risk = float(np.mean(hybrid_req_brake_risk))
    hybrid_p95_req_brake_risk = float(np.percentile(hybrid_req_brake_risk, 95))
    hybrid_mean_pid_weight = float(np.mean(hybrid_pid_weight))
    hybrid_pid_active_pct = float(100.0 * np.mean(hybrid_pid_active))
    hybrid_pid_dominant_pct = float(100.0 * np.mean(hybrid_pid_dominant))
    hybrid_emergency_takeover_pct = float(100.0 * np.mean(hybrid_emergency))
    hybrid_filter_active_pct = float(100.0 * np.mean(hybrid_filter_active))
    hybrid_filter_clamped_pct = float(100.0 * np.mean(hybrid_filter_clamped))
    hybrid_mode_safe_pct = float(100.0 * np.mean(hybrid_mode_safe))
    hybrid_mode_caution_pct = float(100.0 * np.mean(hybrid_mode_caution))
    hybrid_mode_high_risk_pct = float(100.0 * np.mean(hybrid_mode_high))
    hybrid_mode_emergency_pct = float(100.0 * np.mean(hybrid_mode_emergency))

    # ---- Hybrid transition / escalation counts ----
    hybrid_tran_arr     = np.array([s.get("hybrid_transition", 0.0) for s in trajectory], dtype=float)
    hybrid_escal_sc_arr = np.array([s.get("hybrid_escal_sc",   0.0) for s in trajectory], dtype=float)
    hybrid_escal_sh_arr = np.array([s.get("hybrid_escal_sh",   0.0) for s in trajectory], dtype=float)
    hybrid_escal_se_arr = np.array([s.get("hybrid_escal_se",   0.0) for s in trajectory], dtype=float)
    hybrid_escal_ch_arr = np.array([s.get("hybrid_escal_ch",   0.0) for s in trajectory], dtype=float)
    hybrid_escal_he_arr = np.array([s.get("hybrid_escal_he",   0.0) for s in trajectory], dtype=float)
    hybrid_deescal_arr  = np.array([s.get("hybrid_deescal",    0.0) for s in trajectory], dtype=float)

    hybrid_n_transitions   = float(np.sum(hybrid_tran_arr))
    hybrid_n_escal_sc      = float(np.sum(hybrid_escal_sc_arr))
    hybrid_n_escal_sh      = float(np.sum(hybrid_escal_sh_arr))
    hybrid_n_escal_se      = float(np.sum(hybrid_escal_se_arr))
    hybrid_n_escal_ch      = float(np.sum(hybrid_escal_ch_arr))
    hybrid_n_escal_he      = float(np.sum(hybrid_escal_he_arr))
    hybrid_n_escalations   = (
        hybrid_n_escal_sc + hybrid_n_escal_sh + hybrid_n_escal_se
        + hybrid_n_escal_ch + hybrid_n_escal_he
    )
    hybrid_n_deescalations = float(np.sum(hybrid_deescal_arr))

    # ---- Per-mode mean PID weight and raw risk ----
    def _mode_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        m = mask > 0.5
        return float(np.mean(arr[m])) if m.any() else float("nan")

    hybrid_mean_pid_wt_high  = _mode_mean(hybrid_pid_weight, hybrid_mode_high)
    hybrid_mean_risk_raw_high = _mode_mean(hybrid_risk_raw,  hybrid_mode_high)
    protected_mask = (hybrid_mode_high + hybrid_mode_emergency) > 0.5
    hybrid_mean_action_delta_context_protected = float(
        np.mean(hybrid_action_delta_context[protected_mask])
    ) if np.any(protected_mask) else float("nan")

    return {
        "collision_rate":        collision_rate,
        "min_gap":               min_gap,
        "headway_violation_pct": hw_violation_pct,
        "mean_ttc_proxy":        mean_ttc,
        "ttc_violation_pct":     ttc_violation_pct,
        "mean_abs_jerk":         mean_abs_jerk,
        "rms_jerk":              rms_jerk,
        "mean_abs_accel":        mean_abs_accel,
        "speed_tracking_error":  speed_err,
        "progress":              progress,
        "episode_return":        episode_return,
        "accel_oscillation":     accel_osc,
        "jerk_variance":         jerk_var,
        "headway_pref_error":    headway_pref_error,
        "high_stress_jerk":      high_stress_jerk,
        "fatigued_pos_accel":    fatigued_positive_accel,
        # Reward components
        "total_base_reward":     total_base_reward,
        "total_context_reward":  total_context_reward,
        "avg_context_scale":     avg_context_scale,
        "context_contribution_pct": context_contribution_pct,
        # Hybrid supervisor diagnostics
        "hybrid_mean_raw_risk":       hybrid_mean_raw_risk,
        "hybrid_p95_raw_risk":        hybrid_p95_raw_risk,
        "hybrid_mean_ema_risk":       hybrid_mean_ema_risk,
        "hybrid_mean_req_brake_risk": hybrid_mean_req_brake_risk,
        "hybrid_p95_req_brake_risk":  hybrid_p95_req_brake_risk,
        "hybrid_mean_pid_weight":     hybrid_mean_pid_weight,
        # Backward-compatible alias for older plots/tables
        "hybrid_mean_risk":           hybrid_mean_ema_risk,
        "hybrid_pid_active_pct": hybrid_pid_active_pct,
        "hybrid_pid_dominant_pct":    hybrid_pid_dominant_pct,
        "hybrid_emergency_takeover_pct": hybrid_emergency_takeover_pct,
        "hybrid_filter_active_pct":   hybrid_filter_active_pct,
        "hybrid_filter_clamped_pct":  hybrid_filter_clamped_pct,
        "hybrid_mean_action_delta_context_protected": hybrid_mean_action_delta_context_protected,
        "hybrid_mode_safe_pct":       hybrid_mode_safe_pct,
        "hybrid_mode_caution_pct":    hybrid_mode_caution_pct,
        "hybrid_mode_high_risk_pct":  hybrid_mode_high_risk_pct,
        "hybrid_mode_emergency_pct":  hybrid_mode_emergency_pct,
        # Transition / escalation counts (per episode)
        "hybrid_n_transitions":       hybrid_n_transitions,
        "hybrid_n_escalations":       hybrid_n_escalations,
        "hybrid_n_escal_sc":          hybrid_n_escal_sc,
        "hybrid_n_escal_sh":          hybrid_n_escal_sh,
        "hybrid_n_escal_se":          hybrid_n_escal_se,
        "hybrid_n_escal_ch":          hybrid_n_escal_ch,
        "hybrid_n_escal_he":          hybrid_n_escal_he,
        "hybrid_n_deescalations":     hybrid_n_deescalations,
        # Per-mode statistics
        "hybrid_mean_pid_wt_high":    hybrid_mean_pid_wt_high,
        "hybrid_mean_risk_raw_high":  hybrid_mean_risk_raw_high,
    }


# ---------------------------------------------------------------------------
# Agent rollout runners
# ---------------------------------------------------------------------------

def _load_obs_stats(vecnorm_path: Optional[Path]) -> Any:
    """Load VecNormalize observation statistics from disk when available."""
    if vecnorm_path is None or not vecnorm_path.exists():
        return None
    with open(str(vecnorm_path), "rb") as fh:
        stats = pickle.load(fh)
        return getattr(stats, "obs_rms", None)


def _run_ppo_episode(
    controller: PPOController,
    env: CarFollowingEnv,
    seed: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run one episode with a PPO controller, returning (return, trajectory).
    """
    obs, info = env.reset(seed=seed)
    trajectory = [info]
    episode_return = 0.0
    done = False

    while not done:
        action = controller.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        info["reward"] = float(reward)
        episode_return += reward
        trajectory.append(info)
        done = terminated or truncated

    return episode_return, trajectory


def _run_pid_episode_eval(
    config: Config,
    scenario: str,
    seed: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run one PID episode; uses context-aware env (context visible but not used
    by PID — used only for consistent reward calculation).
    """
    env = make_single_env(
        config=config,
        use_context=True,          # context computed but PID ignores it
        use_context_reward=False,   # PID uses vanilla reward for fair comparison
        seed=seed,
        scenario=scenario,
        normalize_observation=False,
    )
    controller = PIDController(config.pid, config.sim)
    ret, traj = run_pid_episode(env, controller, seed=seed)
    env.close()
    return ret, traj


def _run_hybrid_episode(
    hybrid,                  # HybridContextPIDController, already reset by caller
    config: Config,
    scenario: str,
    seed: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run one episode with a *pre-built* HybridContextPIDController.

    The caller is responsible for calling ``hybrid.reset()`` before each
    episode so that no state leaks between episodes.  The environment is
    created fresh per episode and closed before returning.

    Extra info keys added per timestep
    ------------------------------------
    ``hybrid_risk``        : EMA-smoothed physical risk score in [0, 1].
    ``hybrid_in_pid_mode`` : 1 when PID supervisor is active, 0 otherwise.
    """
    env = make_single_env(
        config=config,
        use_context=True,           # context in obs vector (PPO was trained on it)
        use_context_reward=True,    # reward matches context_ppo training
        seed=seed,
        scenario=scenario,
        normalize_observation=False,
    )

    obs, info = env.reset(seed=seed)
    hybrid.reset()                  # explicit reset — no state leakage
    trajectory = [info]
    episode_return = 0.0
    done = False

    while not done:
        action = hybrid.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        info["reward"] = float(reward)
        diag = hybrid.diagnostics()
        for k, v in diag.items():
            info[k] = float(v)
        # Backward-compatible aliases
        info["hybrid_risk"] = float(hybrid.risk_ema)
        info["hybrid_in_pid_mode"] = int(hybrid.in_pid_mode)
        episode_return += reward
        trajectory.append(info)
        done = terminated or truncated

    env.close()
    return episode_return, trajectory


# ---------------------------------------------------------------------------
# Single agent evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent_type: str,
    config: Config,
    seeds: Optional[List[int]] = None,
    n_episodes_per_scenario: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate one agent type across all scenarios and seeds.

    Returns
    -------
    pd.DataFrame
        One row per episode with columns:
        agent, seed, scenario, episode_idx, + all metric columns.
    """
    validate_controllers([agent_type], allow=ALL_CONTROLLERS, context="evaluation")
    seeds = seeds or config.eval.eval_seeds
    n_ep = (
        config.eval.episodes_per_scenario
        if n_episodes_per_scenario is None
        else n_episodes_per_scenario
    )
    if n_ep < 1:
        raise ValueError("Evaluation episodes per scenario must be at least 1.")
    rows: List[Dict[str, Any]] = []
    scenario_order = list(EVAL_SCENARIOS.keys())

    for seed in seeds:
        set_global_seed(seed)
        logger.info(f"  Evaluating {agent_type} | seed={seed}")

        # Load PPO model if needed.
        # For hybrid_context_pid we load context_ppo's weights (the nominal policy).
        model = vecnorm_path = use_context = use_context_reward = None
        ppo_controller: Optional[PPOController] = None
        hybrid_controller = None
        if agent_type not in ("pid",):
            model_key = "context_ppo" if agent_type == "hybrid_context_pid" else agent_type
            try:
                model, vecnorm_path, use_context, use_context_reward = load_trained_model(
                    model_key, seed, config
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Model missing for evaluation agent '{agent_type}' (seed={seed}). {exc}"
                ) from exc

            ppo_controller = PPOController(model, _load_obs_stats(vecnorm_path))

            # Build hybrid controller once per seed; reuse across all episodes.
            # VecNormalize obs_rms loaded here (once) rather than per episode.
            if agent_type == "hybrid_context_pid":
                pid_ctrl = PIDController(config.pid, config.sim)
                hybrid_controller = HybridContextPIDController(
                    model, pid_ctrl, ppo_controller.obs_stats, config.hybrid
                )

        for scenario_name in EVAL_SCENARIOS:
            scenario_idx = scenario_order.index(scenario_name)
            for ep_idx in range(n_ep):
                ep_seed = seed * 10_000 + scenario_idx * 1_000 + ep_idx

                if agent_type == "pid":
                    _, traj = _run_pid_episode_eval(config, scenario_name, seed=ep_seed)
                elif agent_type == "hybrid_context_pid":
                    _, traj = _run_hybrid_episode(
                        hybrid_controller, config, scenario_name, seed=ep_seed
                    )
                else:
                    env = make_single_env(
                        config=config,
                        use_context=use_context,
                        use_context_reward=use_context_reward,
                        seed=ep_seed,
                        scenario=scenario_name,
                        normalize_observation=True,
                    )
                    _, traj = _run_ppo_episode(ppo_controller, env, seed=ep_seed)
                    env.close()

                metrics = compute_episode_metrics(
                    trajectory=traj[1:],  # skip initial reset info
                    cfg_sim=config.sim,
                )
                row: Dict[str, Any] = {
                    "agent":       agent_type,
                    "seed":        seed,
                    "scenario":    scenario_name,
                    "episode_idx": ep_idx,
                }
                row.update(metrics)
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

AGENT_TYPES = list(ALL_CONTROLLERS)


def evaluate_all(config: Config) -> Dict[str, pd.DataFrame]:
    """
    Run evaluation for all agents and all scenarios.

    Saves CSV files:
      artifacts/results/episode_metrics_<agent>.csv
      artifacts/results/aggregate_metrics.csv
      artifacts/results/radar_scores.csv

    Returns
    -------
    dict mapping agent_type → per-episode DataFrame
    """
    config.paths.make_dirs()
    all_dfs: Dict[str, pd.DataFrame] = {}

    for agent_type in AGENT_TYPES:
        logger.info(f"\nEvaluating: {agent_type}")
        df = evaluate_agent(agent_type, config)
        if df.empty:
            logger.warning(f"  No evaluation data for {agent_type} – skipping.")
            continue
        all_dfs[agent_type] = df

        # Save per-agent CSV
        csv_path = config.paths.results / f"episode_metrics_{agent_type}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved → {csv_path}")

    # Aggregate
    if all_dfs:
        _save_aggregate(all_dfs, config)

    return all_dfs


def _write_csv(df: pd.DataFrame, path: "Path", label: str) -> None:
    """Write *df* to *path*, falling back to a temp file if the target is locked."""
    try:
        df.to_csv(path, index=False)
        logger.info(f"{label} → {path}")
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(alt, index=False)
        logger.warning(
            f"[PermissionError] Cannot write to {path.name} (file may be open in another program). "
            f"Saved to {alt.name} instead. Close the file and rename it manually."
        )


def _save_aggregate(
    all_dfs: Dict[str, pd.DataFrame],
    config: Config,
) -> None:
    """Compute aggregate (mean ± std) metrics and radar scores, save to disk."""
    metric_cols = [
        "collision_rate", "min_gap", "headway_violation_pct",
        "mean_ttc_proxy",
        "mean_abs_jerk", "rms_jerk", "mean_abs_accel",
        "speed_tracking_error", "progress", "episode_return",
        "accel_oscillation", "jerk_variance",
        "ttc_violation_pct", "headway_pref_error", "high_stress_jerk", "fatigued_pos_accel",
        # Reward diagnostics
        "total_base_reward", "total_context_reward", "avg_context_scale", "context_contribution_pct",
        # Hybrid supervisor diagnostics (populated for hybrid_context_pid only)
        "hybrid_mean_raw_risk", "hybrid_p95_raw_risk", "hybrid_mean_ema_risk",
        "hybrid_mean_req_brake_risk", "hybrid_p95_req_brake_risk",
        "hybrid_mean_pid_weight",
        "hybrid_mean_risk", "hybrid_pid_active_pct", "hybrid_pid_dominant_pct",
        "hybrid_emergency_takeover_pct", "hybrid_filter_active_pct",
        "hybrid_filter_clamped_pct", "hybrid_mean_action_delta_context_protected", "hybrid_mode_safe_pct",
        "hybrid_mode_caution_pct", "hybrid_mode_high_risk_pct", "hybrid_mode_emergency_pct",
        # Transition / escalation counts
        "hybrid_n_transitions", "hybrid_n_escalations",
        "hybrid_n_escal_sc", "hybrid_n_escal_sh", "hybrid_n_escal_se", "hybrid_n_escal_ch", "hybrid_n_escal_he",
        "hybrid_n_deescalations",
        # Per-mode stats
        "hybrid_mean_pid_wt_high", "hybrid_mean_risk_raw_high",
    ]
    agg_rows: List[Dict[str, Any]] = []
    radar_rows: List[Dict[str, Any]] = []

    for agent_type, df in all_dfs.items():
        row: Dict[str, Any] = {"agent": agent_type}
        raw_means: Dict[str, float] = {}

        for col in metric_cols:
            if col in df.columns:
                valid = df[col].dropna()
                if valid.empty:
                    continue
                m, s = mean_std(valid.tolist())
                row[f"{col}_mean"] = round(m, 4)
                row[f"{col}_std"] = round(s, 4)
                raw_means[col] = m
        agg_rows.append(row)

        # Normalise and build radar
        normed = compute_normalised_scores(raw_means)
        radar = aggregate_radar_scores(normed)
        radar_row: Dict[str, Any] = {"agent": agent_type, "display": agent_display_name(agent_type)}
        radar_row.update(radar)
        radar_rows.append(radar_row)

    agg_df = pd.DataFrame(agg_rows)
    agg_path = config.paths.results / "aggregate_metrics.csv"
    _write_csv(agg_df, agg_path, "Aggregate metrics")

    radar_df = pd.DataFrame(radar_rows)
    radar_path = config.paths.results / "radar_scores.csv"
    _write_csv(radar_df, radar_path, "Radar scores")

    # Console summary
    _print_summary(agg_df, radar_df)


def _print_summary(agg_df: pd.DataFrame, radar_df: pd.DataFrame) -> None:
    """Print a human-readable summary table to the console."""
    width = 72
    print("\n" + "=" * width)
    print("  EVALUATION SUMMARY")
    print("=" * width)

    # Key metrics table
    key_metrics = [
        ("Collision Rate",        "collision_rate_mean",        "collision_rate_std"),
        ("Min Gap [m]",           "min_gap_mean",               "min_gap_std"),
        ("HW Violation [%]",      "headway_violation_pct_mean", "headway_violation_pct_std"),
        ("Mean |Jerk| [m/s³]",    "mean_abs_jerk_mean",         "mean_abs_jerk_std"),
        ("Speed Error [m/s]",     "speed_tracking_error_mean",  "speed_tracking_error_std"),
        ("Episode Return",        "episode_return_mean",        "episode_return_std"),
    ]

    header = f"{'Metric':<26}" + "".join(f"{a:<20}" for a in agg_df["agent"].tolist())
    print(header)
    print("-" * width)
    for label, mean_col, std_col in key_metrics:
        row_str = f"{label:<26}"
        for _, r in agg_df.iterrows():
            m = r.get(mean_col, float("nan"))
            s = r.get(std_col, float("nan"))
            row_str += f"{m:>7.3f} ± {s:<6.3f}   "
        print(row_str)

    print("\n" + "-" * width)
    print("  RADAR SCORES (normalised, higher = better)")
    print("-" * width)
    radar_dims = ["Safety", "Comfort", "Performance", "Stability"]
    header2 = f"{'Dimension':<16}" + "".join(f"{a:<20}" for a in radar_df["agent"].tolist())
    print(header2)
    print("-" * width)
    for dim in radar_dims:
        row_str = f"{dim:<16}"
        for _, r in radar_df.iterrows():
            row_str += f"{r.get(dim, float('nan')):>8.3f}            "
        print(row_str)
    
    print("\n" + "-" * width)
    print("  REWARD DIAGNOSTICS (new base/context structure)")
    print("-" * width)
    reward_metrics = [
        ("Total Base Reward",     "total_base_reward_mean",    "total_base_reward_std"),
        ("Total Context Reward",  "total_context_reward_mean", "total_context_reward_std"),
        ("Avg Context Scale",     "avg_context_scale_mean",    "avg_context_scale_std"),
        ("Context Contribution",  "context_contribution_pct_mean", "context_contribution_pct_std"),
    ]
    header3 = f"{'Metric':<26}" + "".join(f"{a:<20}" for a in agg_df["agent"].tolist())
    print(header3)
    print("-" * width)
    for label, mean_col, std_col in reward_metrics:
        row_str = f"{label:<26}"
        for _, r in agg_df.iterrows():
            m = r.get(mean_col, float("nan"))
            s = r.get(std_col, float("nan"))
            if "Contribution" in label:
                row_str += f"{m:>6.1f}% ± {s:<5.1f}%    "
            else:
                row_str += f"{m:>7.3f} ± {s:<6.3f}   "
        print(row_str)

    # Show hybrid-specific diagnostics only when hybrid_context_pid is present
    hybrid_agents = set(agg_df["agent"].tolist())
    if "hybrid_context_pid" in hybrid_agents:
        print("\n" + "-" * width)
        print("  HYBRID SUPERVISOR DIAGNOSTICS")
        print("-" * width)
        hybrid_metrics = [
            ("Mean Raw Risk",      "hybrid_mean_raw_risk_mean",      "hybrid_mean_raw_risk_std"),
            ("P95 Raw Risk",       "hybrid_p95_raw_risk_mean",       "hybrid_p95_raw_risk_std"),
            ("Mean EMA Risk",      "hybrid_mean_ema_risk_mean",      "hybrid_mean_ema_risk_std"),
            ("Mean ReqBrake Risk", "hybrid_mean_req_brake_risk_mean", "hybrid_mean_req_brake_risk_std"),
            ("P95 ReqBrake Risk",  "hybrid_p95_req_brake_risk_mean",  "hybrid_p95_req_brake_risk_std"),
            ("Mean PID Weight",    "hybrid_mean_pid_weight_mean",    "hybrid_mean_pid_weight_std"),
            ("PID Active [%]",     "hybrid_pid_active_pct_mean",     "hybrid_pid_active_pct_std"),
            ("PID Dominant [%]",   "hybrid_pid_dominant_pct_mean",   "hybrid_pid_dominant_pct_std"),
            ("Emergency [%]",      "hybrid_emergency_takeover_pct_mean", "hybrid_emergency_takeover_pct_std"),
            ("Filter Active [%]",  "hybrid_filter_active_pct_mean",  "hybrid_filter_active_pct_std"),
            ("Filter Clamped [%]", "hybrid_filter_clamped_pct_mean", "hybrid_filter_clamped_pct_std"),
            ("Action Delta Ctx",   "hybrid_mean_action_delta_context_protected_mean", "hybrid_mean_action_delta_context_protected_std"),
            ("Mode Safe [%]",      "hybrid_mode_safe_pct_mean",      "hybrid_mode_safe_pct_std"),
            ("Mode Caution [%]",   "hybrid_mode_caution_pct_mean",   "hybrid_mode_caution_pct_std"),
            ("Mode High [%]",      "hybrid_mode_high_risk_pct_mean", "hybrid_mode_high_risk_pct_std"),
            ("Mode Emergency [%]", "hybrid_mode_emergency_pct_mean", "hybrid_mode_emergency_pct_std"),
            # Transition counts (per episode)
            ("N Transitions",      "hybrid_n_transitions_mean",      "hybrid_n_transitions_std"),
            ("N Escalations",      "hybrid_n_escalations_mean",      "hybrid_n_escalations_std"),
            ("  safe->caution",    "hybrid_n_escal_sc_mean",         "hybrid_n_escal_sc_std"),
            ("  safe->high",       "hybrid_n_escal_sh_mean",         "hybrid_n_escal_sh_std"),
            ("  safe->emergency",  "hybrid_n_escal_se_mean",         "hybrid_n_escal_se_std"),
            ("  caution->high",    "hybrid_n_escal_ch_mean",         "hybrid_n_escal_ch_std"),
            ("  *->emergency",     "hybrid_n_escal_he_mean",         "hybrid_n_escal_he_std"),
            ("N De-escalations",   "hybrid_n_deescalations_mean",    "hybrid_n_deescalations_std"),
            # Per-mode stats
            ("PID Wt in High",     "hybrid_mean_pid_wt_high_mean",   "hybrid_mean_pid_wt_high_std"),
            ("Raw Risk in High",   "hybrid_mean_risk_raw_high_mean", "hybrid_mean_risk_raw_high_std"),
        ]
        header4 = f"{'Metric':<26}" + "".join(f"{a:<20}" for a in agg_df["agent"].tolist())
        print(header4)
        print("-" * width)
        for label, mean_col, std_col in hybrid_metrics:
            row_str = f"{label:<26}"
            for _, r in agg_df.iterrows():
                m = r.get(mean_col, float("nan"))
                s = r.get(std_col, float("nan"))
                if "%" in label:
                    row_str += f"{m:>6.1f}% ± {s:<5.1f}%    "
                elif label.startswith("N ") or label.startswith("  "):
                    # Transition counts — show as floats with 1 decimal place
                    row_str += f"{m:>6.1f} ± {s:<6.1f}      "
                else:
                    row_str += f"{m:>7.3f} ± {s:<6.3f}   "
            print(row_str)

    # Goal-oriented hybrid judge: compare hybrid to PID safety and context_ppo comfort.
    agents_present = set(agg_df["agent"].tolist())
    if {"pid", "context_ppo", "hybrid_context_pid"}.issubset(agents_present):
        pid_row = agg_df[agg_df["agent"] == "pid"].iloc[0]
        ctx_row = agg_df[agg_df["agent"] == "context_ppo"].iloc[0]
        hyb_row = agg_df[agg_df["agent"] == "hybrid_context_pid"].iloc[0]

        pid_collision = float(pid_row.get("collision_rate_mean", np.nan))
        ctx_collision = float(ctx_row.get("collision_rate_mean", np.nan))
        hyb_collision = float(hyb_row.get("collision_rate_mean", np.nan))

        pid_jerk = float(pid_row.get("mean_abs_jerk_mean", np.nan))
        ctx_jerk = float(ctx_row.get("mean_abs_jerk_mean", np.nan))
        hyb_jerk = float(hyb_row.get("mean_abs_jerk_mean", np.nan))

        collision_gap_to_pid = hyb_collision - pid_collision
        jerk_gap_to_ctx = hyb_jerk - ctx_jerk

        # Relative closeness measures (lower is better).
        col_ref = max(1e-6, abs(ctx_collision - pid_collision))
        jerk_ref = max(1e-6, abs(pid_jerk - ctx_jerk))
        collision_to_pid_ratio = abs(hyb_collision - pid_collision) / col_ref
        jerk_to_ctx_ratio = abs(hyb_jerk - ctx_jerk) / jerk_ref

        if collision_to_pid_ratio <= 0.45 and jerk_to_ctx_ratio <= 0.45:
            judge = "HYBRID SAFETY-COMFORT TARGET MET"
        elif collision_to_pid_ratio > 0.70:
            judge = "HYBRID TOO SOFT"
        elif jerk_to_ctx_ratio > 0.60:
            judge = "HYBRID TOO INTRUSIVE"
        else:
            # Tie-break toward safety objective: if safety is still far from PID, call it soft.
            judge = "HYBRID TOO SOFT" if collision_to_pid_ratio >= jerk_to_ctx_ratio else "HYBRID TOO INTRUSIVE"

        print("\n" + "-" * width)
        print("  HYBRID GOAL CHECK")
        print("-" * width)
        print(f"Collision Gap vs PID      : {collision_gap_to_pid:+.3f} (hybrid - pid)")
        print(f"Jerk Gap vs Context PPO   : {jerk_gap_to_ctx:+.3f} (hybrid - context_ppo)")
        print(f"Emergency [%]             : {hyb_row.get('hybrid_emergency_takeover_pct_mean', float('nan')):.1f}")
        print(f"Mode High [%]             : {hyb_row.get('hybrid_mode_high_risk_pct_mean', float('nan')):.1f}")
        print(
            "PID Wt in High           : "
            f"{hyb_row.get('hybrid_mean_pid_wt_high_mean', float('nan')):.3f} ± "
            f"{hyb_row.get('hybrid_mean_pid_wt_high_std', float('nan')):.3f}"
        )
        print(f"Collision Closeness Ratio : {collision_to_pid_ratio:.3f} (0=at pid, 1=at context_ppo)")
        print(f"Jerk Closeness Ratio      : {jerk_to_ctx_ratio:.3f} (0=at context_ppo, 1=at pid)")
        print(f"AUTO JUDGMENT             : {judge}")

        # Final tuning success criteria: material collision reduction with preserved jerk.
        collision_materially_reduced = hyb_collision <= min(0.30, ctx_collision - 0.08)
        jerk_preserved = abs(jerk_gap_to_ctx) <= 0.02
        emergency_reasonable = float(hyb_row.get("hybrid_emergency_takeover_pct_mean", np.nan)) <= 10.0
        high_mode_active = float(hyb_row.get("hybrid_mode_high_risk_pct_mean", np.nan)) >= 10.0
        pid_high_strong = float(hyb_row.get("hybrid_mean_pid_wt_high_mean", np.nan)) >= 0.90

        if (
            collision_materially_reduced
            and jerk_preserved
            and emergency_reasonable
            and high_mode_active
            and pid_high_strong
        ):
            final_tuning = "FINAL TUNING SUCCESS"
        else:
            final_tuning = "FURTHER TUNING REQUIRED"
        print(f"FINAL TUNING STATUS       : {final_tuning}")

    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Trajectory recorder (for single-episode plots)
# ---------------------------------------------------------------------------

def record_trajectory(
    agent_type: str,
    config: Config,
    seed: int = 100,
    scenario: str = "switching",
    lead_profile: str = "sinusoidal",
) -> List[Dict[str, Any]]:
    """
    Run a single demonstrative episode and return the full trajectory list.

    Used by plots.py to generate distance/speed/jerk/context time plots.
    """
    if agent_type == "pid":
        env = make_single_env(
            config=config,
            use_context=True,
            use_context_reward=False,
            seed=seed,
            lead_profile=lead_profile,
            scenario=scenario,
            normalize_observation=False,
        )
        controller = PIDController(config.pid, config.sim)
        _, traj = run_pid_episode(env, controller, seed=seed)
        env.close()
        return traj

    if agent_type == "hybrid_context_pid":
        # Hybrid is evaluation-only: load context_ppo as nominal policy
        # and wrap it with PID safety supervisor.
        model, vecnorm_path, _, _ = load_trained_model("context_ppo", seed, config)

        obs_stats = _load_obs_stats(vecnorm_path)

        pid_ctrl = PIDController(config.pid, config.sim)
        hybrid = HybridContextPIDController(model, pid_ctrl, obs_stats, config.hybrid)
        _, traj = _run_hybrid_episode(hybrid, config, scenario=scenario, seed=seed)
        return traj

    model, vecnorm_path, use_context, use_context_reward = load_trained_model(
        agent_type, seed, config
    )
    env = make_single_env(
        config=config,
        use_context=use_context,
        use_context_reward=use_context_reward,
        seed=seed,
        lead_profile=lead_profile,
        scenario=scenario,
        normalize_observation=True,
    )
    ppo_controller = PPOController(model, _load_obs_stats(vecnorm_path))
    _, traj = _run_ppo_episode(ppo_controller, env, seed=seed)
    env.close()
    return traj


def _build_diagnostic_obs(
    config: Config,
    use_context: bool,
    gap: float,
    v_ego: float,
    v_lead: float,
    accel: float,
    jerk: float,
    context: Dict[str, float],
) -> np.ndarray:
    """Construct a single normalized observation matching PPO training format."""
    sim = config.sim
    phys = np.array([
        np.clip(gap / sim.gap_norm, 0.0, 2.0),
        np.clip(v_ego / sim.speed_norm, 0.0, 1.2),
        np.clip(v_lead / sim.speed_norm, 0.0, 1.2),
        np.clip(accel / sim.accel_norm, -1.0, 1.0),
        np.clip(jerk / sim.jerk_norm, -2.0, 2.0),
    ], dtype=np.float32)
    if not use_context:
        return phys
    ctx = np.array([
        np.clip(context["stress_level"], 0.0, 1.0),
        np.clip(context["fatigue_level"], 0.0, 1.0),
        np.clip(context["preferred_headway"] / sim.headway_pref_norm, 0.0, 1.2),
        np.clip(context["comfort_priority"], 0.0, 1.0),
        np.clip(context["efficiency_priority"], 0.0, 1.0),
    ], dtype=np.float32)
    return np.concatenate([phys, ctx])


def run_context_sensitivity_diagnostics(
    config: Config,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate action sensitivity to context under fixed physical states."""
    physical_states = [
        {"label": "nominal", "gap": 20.0, "v_ego": 12.0, "v_lead": 12.0, "accel": 0.0, "jerk": 0.0},
        {"label": "closing_in", "gap": 10.0, "v_ego": 14.0, "v_lead": 10.0, "accel": 0.5, "jerk": 0.2},
    ]
    contexts = [
        {
            "label": "calm",
            "stress_level": 0.1,
            "fatigue_level": 0.1,
            "preferred_headway": 1.5,
            "comfort_priority": 0.35,
            "efficiency_priority": 0.9,
        },
        {
            "label": "stressed",
            "stress_level": 0.9,
            "fatigue_level": 0.2,
            "preferred_headway": 2.3,
            "comfort_priority": 0.8,
            "efficiency_priority": 0.35,
        },
        {
            "label": "fatigued",
            "stress_level": 0.2,
            "fatigue_level": 0.9,
            "preferred_headway": 2.3,
            "comfort_priority": 0.75,
            "efficiency_priority": 0.35,
        },
    ]

    rows: List[Dict[str, Any]] = []
    for agent_type in list(TRAINABLE_CONTROLLERS):
        try:
            model, vecnorm_path, use_context, _ = load_trained_model(agent_type, seed, config)
        except FileNotFoundError:
            continue

        obs_stats = None
        if vecnorm_path is not None and vecnorm_path.exists():
            with open(str(vecnorm_path), "rb") as fh:
                stats = pickle.load(fh)
                obs_stats = getattr(stats, "obs_rms", None)

        for phys in physical_states:
            for ctx in contexts:
                obs = _build_diagnostic_obs(
                    config=config,
                    use_context=use_context,
                    gap=phys["gap"],
                    v_ego=phys["v_ego"],
                    v_lead=phys["v_lead"],
                    accel=phys["accel"],
                    jerk=phys["jerk"],
                    context=ctx,
                )
                obs_input = obs.copy()
                if obs_stats is not None:
                    obs_input = (obs_input - obs_stats.mean) / np.sqrt(obs_stats.var + 1e-8)
                    obs_input = np.clip(obs_input, -10.0, 10.0)
                action, _ = model.predict(obs_input, deterministic=True)
                rows.append({
                    "agent": agent_type,
                    "state_label": phys["label"],
                    "context_label": ctx["label"],
                    "gap": phys["gap"],
                    "v_ego": phys["v_ego"],
                    "v_lead": phys["v_lead"],
                    "stress_level": ctx["stress_level"],
                    "fatigue_level": ctx["fatigue_level"],
                    "preferred_headway": ctx["preferred_headway"],
                    "action": float(np.squeeze(action)),
                })

    df = pd.DataFrame(rows)
    out_path = config.paths.results / "context_sensitivity_diagnostics.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Context sensitivity diagnostics → {out_path}")
    return df


def _relative_improvement(context_value: float, baseline_value: float, higher_is_better: bool) -> float:
    """Compute relative improvement (%) of context-aware PPO versus a baseline."""
    denom = max(abs(baseline_value), 1e-8)
    if higher_is_better:
        return 100.0 * (context_value - baseline_value) / denom
    return 100.0 * (baseline_value - context_value) / denom


def evaluate_showcase(
    config: Config,
    agents: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_episodes_per_case: Optional[int] = None,
    include_optional_escalation: bool = True,
) -> pd.DataFrame:
    """
    Run deterministic context-sensitive showcase evaluations.

    Cases are curated to test when context awareness should matter most.
    Saves episode-level rows and summary CSVs under artifacts/results/.
    """
    config.paths.make_dirs()
    agents = agents or ["pid", "vanilla_ppo", "context_ppo"]
    validate_controllers(agents, allow=ALL_CONTROLLERS, context="showcase evaluation")
    seeds = seeds or config.eval.showcase_seeds
    n_ep = n_episodes_per_case or config.eval.showcase_episodes_per_case
    cases = get_showcase_cases(include_optional_escalation=include_optional_escalation)

    rows: List[Dict[str, Any]] = []
    case_ids = list(cases.keys())

    for agent_type in agents:
        logger.info(f"Showcase evaluation: {agent_type}")
        for seed in seeds:
            set_global_seed(seed)

            model = vecnorm_path = use_context = use_context_reward = None
            ppo_controller: Optional[PPOController] = None
            if agent_type != "pid":
                try:
                    model, vecnorm_path, use_context, use_context_reward = load_trained_model(
                        agent_type, seed, config
                    )
                except FileNotFoundError as exc:
                    logger.warning(str(exc))
                    continue
                ppo_controller = PPOController(model, _load_obs_stats(vecnorm_path))

            for case_idx, case_id in enumerate(case_ids):
                case_def = cases[case_id]
                scenario = case_def["scenario"]
                lead_profile = case_def["lead_profile"]

                for ep_idx in range(n_ep):
                    ep_seed = seed * 100_000 + case_idx * 1_000 + ep_idx

                    if agent_type == "pid":
                        env = make_single_env(
                            config=config,
                            use_context=True,
                            use_context_reward=False,
                            seed=ep_seed,
                            scenario=scenario,
                            lead_profile=lead_profile,
                            normalize_observation=False,
                        )
                        controller = PIDController(config.pid, config.sim)
                        _, traj = run_pid_episode(env, controller, seed=ep_seed)
                        env.close()
                    else:
                        env = make_single_env(
                            config=config,
                            use_context=use_context,
                            use_context_reward=use_context_reward,
                            seed=ep_seed,
                            scenario=scenario,
                            lead_profile=lead_profile,
                            normalize_observation=True,
                        )
                        _, traj = _run_ppo_episode(ppo_controller, env, seed=ep_seed)
                        env.close()

                    metrics = compute_episode_metrics(trajectory=traj[1:], cfg_sim=config.sim)
                    row: Dict[str, Any] = {
                        "agent": agent_type,
                        "seed": seed,
                        "case_id": case_id,
                        "case_display": case_def["display"],
                        "scenario": scenario,
                        "lead_profile": lead_profile,
                        "episode_idx": ep_idx,
                    }
                    row.update(metrics)
                    rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Showcase evaluation produced no rows.")
        return df

    episode_path = config.paths.results / "showcase_episode_metrics.csv"
    df.to_csv(episode_path, index=False)
    logger.info(f"Showcase episode metrics → {episode_path}")

    _save_showcase_reports(df, config)
    return df


def _save_showcase_reports(df: pd.DataFrame, config: Config) -> None:
    """Generate showcase summary, win-count, and seed-consistency reports."""
    metric_cols = [
        "collision_rate",
        "min_gap",
        "headway_violation_pct",
        "mean_ttc_proxy",
        "ttc_violation_pct",
        "mean_abs_jerk",
        "rms_jerk",
        "mean_abs_accel",
        "speed_tracking_error",
        "progress",
        "episode_return",
        "accel_oscillation",
        "jerk_variance",
        "headway_pref_error",
        "high_stress_jerk",
        "fatigued_pos_accel",
    ]

    summary_rows: List[Dict[str, Any]] = []
    grouped = df.groupby(["case_id", "case_display", "agent"])
    case_metric_means = (
        df.groupby(["case_id", "agent"])[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    for (case_id, case_display, agent), sub in grouped:
        for metric in metric_cols:
            vals = sub[metric].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=0 if vals.size <= 1 else 1))

            ctx_row = case_metric_means[
                (case_metric_means["case_id"] == case_id) & (case_metric_means["agent"] == "context_ppo")
            ]
            pid_row = case_metric_means[
                (case_metric_means["case_id"] == case_id) & (case_metric_means["agent"] == "pid")
            ]
            van_row = case_metric_means[
                (case_metric_means["case_id"] == case_id) & (case_metric_means["agent"] == "vanilla_ppo")
            ]

            ctx_vs_pid = np.nan
            ctx_vs_van = np.nan
            higher_is_better = METRIC_DIRECTIONS.get(metric, False)
            if not ctx_row.empty and not pid_row.empty:
                ctx_vs_pid = _relative_improvement(
                    float(ctx_row.iloc[0][metric]),
                    float(pid_row.iloc[0][metric]),
                    higher_is_better,
                )
            if not ctx_row.empty and not van_row.empty:
                ctx_vs_van = _relative_improvement(
                    float(ctx_row.iloc[0][metric]),
                    float(van_row.iloc[0][metric]),
                    higher_is_better,
                )

            summary_rows.append(
                {
                    "case_id": case_id,
                    "case_display": case_display,
                    "agent": agent,
                    "metric": metric,
                    "mean": m,
                    "std": s,
                    "higher_is_better": int(higher_is_better),
                    "context_vs_pid_rel_improvement_pct": ctx_vs_pid,
                    "context_vs_vanilla_rel_improvement_pct": ctx_vs_van,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = config.paths.results / "showcase_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Showcase summary → {summary_path}")

    # Win-count table: count metrics where context_ppo beats each opponent.
    win_rows: List[Dict[str, Any]] = []
    for case_id in df["case_id"].unique():
        ctx_vals = case_metric_means[
            (case_metric_means["case_id"] == case_id) & (case_metric_means["agent"] == "context_ppo")
        ]
        if ctx_vals.empty:
            continue
        for opponent in ["pid", "vanilla_ppo", "hybrid_context_pid"]:
            opp_vals = case_metric_means[
                (case_metric_means["case_id"] == case_id) & (case_metric_means["agent"] == opponent)
            ]
            if opp_vals.empty:
                continue

            wins = 0
            losses = 0
            ties = 0
            for metric in metric_cols:
                c_val = float(ctx_vals.iloc[0][metric])
                o_val = float(opp_vals.iloc[0][metric])
                hib = METRIC_DIRECTIONS.get(metric, False)
                if np.isclose(c_val, o_val, atol=1e-9):
                    ties += 1
                elif (c_val > o_val and hib) or (c_val < o_val and not hib):
                    wins += 1
                else:
                    losses += 1

            win_rows.append(
                {
                    "case_id": case_id,
                    "opponent": opponent,
                    "metrics_compared": len(metric_cols),
                    "context_wins": wins,
                    "opponent_wins": losses,
                    "ties": ties,
                }
            )

    wins_df = pd.DataFrame(win_rows)
    wins_path = config.paths.results / "showcase_win_counts.csv"
    wins_df.to_csv(wins_path, index=False)
    logger.info(f"Showcase win counts → {wins_path}")

    # Seed-wise consistency: context_ppo better across seed means.
    seed_mean = (
        df.groupby(["case_id", "seed", "agent"])[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    consistency_rows: List[Dict[str, Any]] = []
    for case_id in df["case_id"].unique():
        for metric in metric_cols:
            for opponent in ["pid", "vanilla_ppo"]:
                better_count = 0
                comparable = 0
                for seed in sorted(df["seed"].unique()):
                    c = seed_mean[
                        (seed_mean["case_id"] == case_id)
                        & (seed_mean["seed"] == seed)
                        & (seed_mean["agent"] == "context_ppo")
                    ]
                    o = seed_mean[
                        (seed_mean["case_id"] == case_id)
                        & (seed_mean["seed"] == seed)
                        & (seed_mean["agent"] == opponent)
                    ]
                    if c.empty or o.empty:
                        continue
                    comparable += 1
                    c_val = float(c.iloc[0][metric])
                    o_val = float(o.iloc[0][metric])
                    hib = METRIC_DIRECTIONS.get(metric, False)
                    if (c_val > o_val and hib) or (c_val < o_val and not hib):
                        better_count += 1

                if comparable == 0:
                    continue
                consistency_rows.append(
                    {
                        "case_id": case_id,
                        "metric": metric,
                        "opponent": opponent,
                        "n_seeds": comparable,
                        "context_better_seeds": better_count,
                        "consistency_ratio": float(better_count / comparable),
                    }
                )

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_path = config.paths.results / "showcase_seed_consistency.csv"
    consistency_df.to_csv(consistency_path, index=False)
    logger.info(f"Showcase seed consistency → {consistency_path}")

    if not wins_df.empty:
        print("\nShowcase metric win-counts (context_ppo vs baselines):")
        print(wins_df.to_string(index=False))


def record_showcase_trajectories(
    config: Config,
    seed: int = 700,
    agents: Optional[List[str]] = None,
    include_optional_escalation: bool = True,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Record one deterministic trajectory per showcase case for selected agents."""
    agents = agents or ["pid", "vanilla_ppo", "context_ppo"]
    cases = get_showcase_cases(include_optional_escalation=include_optional_escalation)
    trajectories: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for case_id, case_def in cases.items():
        trajectories[case_id] = {}
        for agent_type in agents:
            try:
                traj = record_trajectory(
                    agent_type=agent_type,
                    config=config,
                    seed=seed,
                    scenario=case_def["scenario"],
                    lead_profile=case_def["lead_profile"],
                )
                trajectories[case_id][agent_type] = traj
            except FileNotFoundError as exc:
                logger.warning(str(exc))
                continue
    return trajectories
