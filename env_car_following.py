"""
env_car_following.py
====================
Custom Gymnasium environment for 1D longitudinal car-following.

Two observation modes are supported via the ``use_context`` flag:

  A) Vanilla PPO   – physical-only state (5 dims):
       s = [d, v_ego, v_lead, a, j]

  B) Context-aware PPO – augmented state (10 dims):
       s̃ = [d, v_ego, v_lead, a, j, κ_stress, κ_fatigue, h_pref, κ_comfort, κ_eff]

An optional ``context_in_reward`` flag controls whether the reward function
uses the context vector (needed for the ablation study).

Lead-vehicle trajectory profiles
---------------------------------
  constant       – constant speed sampled at episode start
  sinusoidal     – speed varies as a slow sine wave
  braking        – abrupt deceleration mid-episode
  stop_and_go    – repeated stop / restart cycles
  random_smooth  – AR(1) speed perturbations

Termination
-----------
  - Collision:         gap ≤ collision_gap
  - Time horizon:      step == episode_steps

Info dict
---------
Every ``step()`` call returns an info dict containing all raw values needed
for metric logging:  gap, v_ego, v_lead, accel, jerk, headway, context, reward
components.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Config, SimConfig, RewardConfig, LeadConfig, HumanConfig
from context_model import ContextModel, ContextOutput, make_context_model


# ---------------------------------------------------------------------------
# Lead-vehicle trajectory generator
# ---------------------------------------------------------------------------

class LeadVehicleProfile:
    """
    Generates the lead vehicle speed at each timestep according to a chosen
    profile.  All profiles are deterministic given ``rng`` and the profile
    name.

    Parameters
    ----------
    profile:
        One of ("constant", "sinusoidal", "braking", "stop_and_go",
        "random_smooth").
    cfg_lead:
        LeadConfig parameters.
    cfg_sim:
        SimConfig (needed for dt, v bounds, episode length).
    rng:
        NumPy random generator.
    """

    _VALID = {"constant", "sinusoidal", "braking", "stop_and_go", "random_smooth"}

    def __init__(
        self,
        profile: str,
        cfg_lead: LeadConfig,
        cfg_sim: SimConfig,
        rng: np.random.Generator,
    ) -> None:
        if profile not in self._VALID:
            raise ValueError(f"Unknown lead profile: {profile!r}")
        self.profile = profile
        self.cfg_lead = cfg_lead
        self.cfg_sim = cfg_sim
        self.rng = rng
        self._speed_sequence: Optional[np.ndarray] = None

    def reset(self, initial_speed: float) -> None:
        """Pre-generate the full speed sequence for the episode."""
        T = self.cfg_sim.episode_steps
        dt = self.cfg_sim.dt
        self._speed_sequence = self._build(initial_speed, T, dt)

    def speed_at(self, step: int) -> float:
        """Return lead speed at the current step."""
        assert self._speed_sequence is not None, "Call reset() first"
        idx = int(np.clip(step, 0, len(self._speed_sequence) - 1))
        return float(self._speed_sequence[idx])

    # ------------------------------------------------------------------
    # Profile builders
    # ------------------------------------------------------------------

    def _build(self, v0: float, T: int, dt: float) -> np.ndarray:
        if self.profile == "constant":
            return self._constant(v0, T)
        if self.profile == "sinusoidal":
            return self._sinusoidal(v0, T, dt)
        if self.profile == "braking":
            return self._braking(v0, T, dt)
        if self.profile == "stop_and_go":
            return self._stop_and_go(v0, T, dt)
        if self.profile == "random_smooth":
            return self._random_smooth(v0, T)
        raise ValueError(self.profile)

    def _constant(self, v0: float, T: int) -> np.ndarray:
        speed = self.rng.uniform(8.0, 18.0)
        return np.full(T, speed, dtype=np.float32)

    def _sinusoidal(self, v0: float, T: int, dt: float) -> np.ndarray:
        cfg = self.cfg_lead
        base = self.rng.uniform(10.0, 16.0)
        t = np.arange(T) * dt
        speeds = base + cfg.sin_amplitude * np.sin(2 * math.pi * t / cfg.sin_period)
        return np.clip(speeds, self.cfg_sim.v_min, self.cfg_sim.v_max).astype(np.float32)

    def _braking(self, v0: float, T: int, dt: float) -> np.ndarray:
        cfg = self.cfg_lead
        base = self.rng.uniform(12.0, 18.0)
        speeds = np.full(T, base, dtype=np.float32)
        start_idx = int(cfg.brake_start_frac * T)
        brake_steps = int(cfg.brake_duration / dt)
        for i in range(brake_steps):
            t = start_idx + i
            if t < T:
                speeds[t] = max(0.0, speeds[t - 1] + cfg.brake_decel * dt if t > 0 else base)
        # Hold minimum then resume
        min_speed = max(0.0, speeds[start_idx + brake_steps - 1] if (start_idx + brake_steps - 1) < T else 0.0)
        for t in range(start_idx + brake_steps, min(start_idx + 2 * brake_steps, T)):
            speeds[t] = min(base, speeds[t - 1] + 1.5 * dt)
        return np.clip(speeds, self.cfg_sim.v_min, self.cfg_sim.v_max).astype(np.float32)

    def _stop_and_go(self, v0: float, T: int, dt: float) -> np.ndarray:
        cfg = self.cfg_lead
        base = self.rng.uniform(8.0, 14.0)
        speeds = np.zeros(T, dtype=np.float32)
        running = True
        t = 0
        cycle_steps = max(1, int((T / max(1, cfg.sg_cycles))))
        stop_steps = int(cfg.sg_stop_duration / dt)
        go_steps = cycle_steps - stop_steps
        step = 0
        for _ in range(cfg.sg_cycles + 1):
            # Go phase
            for _ in range(go_steps):
                if step < T:
                    speeds[step] = base
                    step += 1
            # Stop phase: gradual decel
            for i in range(stop_steps):
                if step < T:
                    frac = i / max(1, stop_steps)
                    speeds[step] = base * max(0.0, 1.0 - 2 * frac)
                    step += 1
        if step < T:
            speeds[step:] = base
        return np.clip(speeds, self.cfg_sim.v_min, self.cfg_sim.v_max).astype(np.float32)

    def _random_smooth(self, v0: float, T: int) -> np.ndarray:
        cfg = self.cfg_lead
        base = self.rng.uniform(10.0, 16.0)
        perturbation = np.zeros(T)
        noise = self.rng.standard_normal(T)
        for t in range(1, T):
            perturbation[t] = cfg.smooth_decay * perturbation[t - 1] + cfg.smooth_sigma * noise[t]
        return np.clip((base + perturbation).astype(np.float32), self.cfg_sim.v_min, self.cfg_sim.v_max)


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Computes per-step reward as: total = base_reward + context_reward_scale * context_reward
    
    Base reward: safety (collision/headway) + comfort (jerk/accel) + efficiency (speed tracking)
    Context reward: small preference shaper (stressed→safer, fatigued→smoother)
    
    Uses optional warmup: context_scale ramps from 0 → target over first 30% of training.
    """

    def __init__(
        self,
        cfg_sim: SimConfig,
        cfg_rew: RewardConfig,
        use_context_reward: bool = True,
    ) -> None:
        self.cfg_sim = cfg_sim
        self.cfg_rew = cfg_rew
        self.use_context_reward = use_context_reward
        self._logger = logging.getLogger(__name__)
        self._debug_counter = 0
        self._effective_context_scale = cfg_rew.context_reward_scale
        self._ratio_latest = 0.0
        self._ratio_scaled_latest = 0.0
        self._ratio_ema = 0.0
        self._ratio_samples = 0
        self._ema_abs_jerk = 0.0
        self._ema_pos_accel = 0.0

    def reset_episode(self) -> None:
        """Reset per-episode smooth diagnostic states."""
        self._ema_abs_jerk = 0.0
        self._ema_pos_accel = 0.0

    def get_diagnostics(self) -> Dict[str, float]:
        """Return latest reward-balance diagnostics for monitoring callbacks."""
        return {
            "context_base_ratio": self._ratio_latest,
            "context_base_ratio_scaled": self._ratio_scaled_latest,
            "context_base_ratio_ema": self._ratio_ema,
            "context_ratio_samples": float(self._ratio_samples),
            "effective_context_scale": self._effective_context_scale,
        }

    def set_training_fraction(self, fraction: float) -> None:
        """
        Update effective context reward scale based on training progress.
        Used for optional warmup: ramps from 0 to target_scale.
        """
        cfg = self.cfg_rew
        if cfg.context_warmup_enabled:
            if fraction < cfg.context_warmup_fraction:
                progress = fraction / max(cfg.context_warmup_fraction, 1e-6)
                self._effective_context_scale = cfg.context_reward_scale * progress
            else:
                self._effective_context_scale = cfg.context_reward_scale
        else:
            self._effective_context_scale = cfg.context_reward_scale

    def compute(
        self,
        gap: float,
        v_ego: float,
        v_lead: float,
        accel: float,
        jerk: float,
        context: Optional[ContextOutput],
        v_target: float,
        episode_scenario: Optional[str] = None,
        cmd_delta: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total_reward = base_reward + context_reward_scale * context_reward.

        Args:
            cmd_delta: |a_cmd(t) - a_cmd(t-1)| — used for action-change penalty and diagnostics.

        Returns (total_reward, detailed_components_dict).
        """
        cfg = self.cfg_rew
        sim = self.cfg_sim
        eps = sim.eps

        # Derived quantities
        effective_speed = max(v_ego, sim.headway_speed_floor)
        headway_raw = gap / (effective_speed + eps)
        headway = float(np.clip(headway_raw, 0.0, sim.headway_clip_max))
        collision = gap <= sim.collision_gap
        short_gap = max(0.0, 2.0 - gap)
        r_stability = 0.0
        r_comfort_guard = 0.0
        r_action_change = 0.0
        scenario_gain = 0.0

        # Smooth jerk/accel signals for context shaping to avoid reactive oscillations.
        ema_alpha = float(np.clip(cfg.context_ema_alpha, 0.01, 1.0))
        abs_jerk = abs(jerk)
        pos_accel = max(0.0, accel)
        self._ema_abs_jerk = (1.0 - ema_alpha) * self._ema_abs_jerk + ema_alpha * abs_jerk
        self._ema_pos_accel = (1.0 - ema_alpha) * self._ema_pos_accel + ema_alpha * pos_accel

        # ─────────────────────────────────────────────────────────────────────
        # BASE REWARD (dominant): safety + comfort + efficiency
        # ─────────────────────────────────────────────────────────────────────
        
        # Safety: collision and headway violations
        r_collision = -cfg.w_collision if collision else 0.0
        r_gap = -cfg.w_gap * (max(0.0, sim.safe_headway - headway) ** 2)
        r_distance_buffer = -cfg.w_distance_buffer * (short_gap ** 2)
        r_safety = r_collision + r_gap + r_distance_buffer
        
        # Comfort: jerk and acceleration penalties
        # Risk-aware: penalize jerk more strongly in safe driving, allow more during emergencies
        if context is not None:
            stress = context.stress_level
            # Simplified risk estimation: higher stress → higher risk
            risk_score = float(np.clip(stress, 0.0, 1.0))
            # Interpolate jerk multiplier based on risk
            low_mult = cfg.low_risk_jerk_multiplier
            high_mult = cfg.high_risk_jerk_multiplier
            jerk_multiplier = low_mult + risk_score * (high_mult - low_mult)
        else:
            jerk_multiplier = 1.0  # vanilla: use baseline jerk penalty

        r_comfort = -(cfg.alpha_accel * abs(accel) + cfg.jerk_weight * jerk_multiplier * abs(jerk))

        # Hard comfort guard to strongly discourage high-jerk maneuvers for all agents.
        jerk_excess = max(0.0, abs_jerk - cfg.comfort_guard_jerk_threshold)
        r_comfort_guard = -cfg.comfort_guard_weight * (jerk_excess ** 2)
        r_comfort += r_comfort_guard
        
        # Efficiency: speed tracking
        r_efficiency = -cfg.efficiency_weight * abs(v_ego - v_target)

        # Action-change penalty: small quadratic penalty on command rate-of-change.
        # Applies to all agents, discourages oscillatory control (base reward term).
        r_action_change = -cfg.action_change_penalty_weight * (cmd_delta ** 2)

        # Base reward: combine safety, comfort, efficiency, action-change
        base_reward = r_safety + r_comfort + r_efficiency + r_action_change

        # Handle collision: override with dominant penalty
        if collision:
            base_reward = float(np.clip(r_collision, cfg.total_reward_clip_min, cfg.total_reward_clip_max))
            context_reward = 0.0
            total_reward = base_reward
            reward_clipped = float(base_reward != r_collision)
        else:
            # ─────────────────────────────────────────────────────────────────────
            # CONTEXT REWARD (small shaper): preference modulation
            # ─────────────────────────────────────────────────────────────────────
            
            context_reward = 0.0
            if self.use_context_reward and context is not None and self._effective_context_scale > 0.0:
                kappa_stress = context.stress_level
                kappa_fatigue = context.fatigue_level
                h_pref = context.preferred_headway
                comp_clip = max(cfg.context_component_clip, 1e-6)

                scenario_name = (episode_scenario or "").lower()
                if scenario_name == "stressed":
                    scenario_gain = cfg.scenario_gain_stressed
                elif scenario_name == "fatigued":
                    scenario_gain = cfg.scenario_gain_fatigued
                elif scenario_name == "both":
                    scenario_gain = cfg.scenario_gain_both
                elif scenario_name == "switching":
                    scenario_gain = cfg.scenario_gain_switching
                else:
                    scenario_gain = cfg.scenario_gain_default
                
                # Soft preference terms use smooth bounded bonuses plus small penalties.
                headway_error = abs(headway - h_pref)
                # Gate positive context bonuses when jerk is high to avoid rewarding oscillatory behavior.
                bonus_jerk_gate = float(np.exp(-self._ema_abs_jerk / max(cfg.context_bonus_jerk_gate_scale, 1e-6)))
                headway_pref_bonus = float(np.exp(-headway_error)) * bonus_jerk_gate

                # Stressed driver: one-sided headway SHORTFALL penalty (not exact-match chasing)
                # Only penalizes when headway is below (h_pref - deadband); no upward pressure.
                r_stress = 0.0
                if kappa_stress > 0.05:
                    r_stress += -cfg.stress_jerk_penalty_weight * kappa_stress * abs(jerk)
                    headway_shortfall = max(
                        0.0, h_pref - headway - cfg.context_headway_deadband
                    )
                    r_stress += -cfg.stress_headway_shortfall_weight * kappa_stress * (headway_shortfall ** 2)
                r_stress = float(np.clip(r_stress, -comp_clip, comp_clip))
                
                # Fatigued driver: reward smooth low-accel actions and penalize only aggressive accel.
                r_fatigue = 0.0
                if kappa_fatigue > 0.05:
                    accel_smooth_bonus = float(np.exp(-self._ema_pos_accel)) * bonus_jerk_gate
                    aggressive_excess = max(0.0, self._ema_pos_accel - cfg.fatigue_aggressive_accel_threshold)
                    r_fatigue += cfg.fatigue_smooth_bonus_weight * kappa_fatigue * accel_smooth_bonus
                    r_fatigue += -cfg.fatigue_accel_penalty_weight * kappa_fatigue * aggressive_excess
                r_fatigue = float(np.clip(r_fatigue, -comp_clip, comp_clip))
                
                # Both stressed and fatigued: gentle bonus, with a tiny short-gap guard.
                r_both = 0.0
                if kappa_stress > 0.5 and kappa_fatigue > 0.5:
                    both_level = min(kappa_stress, kappa_fatigue)
                    short_gap_violation = max(0.0, sim.safe_headway - headway)
                    r_both += cfg.both_context_extra_caution_weight * both_level * headway_pref_bonus
                    r_both += -cfg.both_short_gap_penalty_weight * both_level * short_gap_violation
                r_both = float(np.clip(r_both, -comp_clip, comp_clip))

                # Always-on tiny damping so context shaping cannot promote oscillations.
                r_stability = -cfg.context_jerk_stability_weight * abs(jerk)
                r_stability = float(np.clip(r_stability, -comp_clip, comp_clip))
                
                context_reward_raw = scenario_gain * (r_stress + r_fatigue + r_both + r_stability)
                
                # Clip context reward to prevent domination
                context_reward = float(np.clip(
                    context_reward_raw,
                    -cfg.context_reward_clip,
                    cfg.context_reward_clip,
                ))
            
            # Total reward: base + scaled context
            total_raw = base_reward + self._effective_context_scale * context_reward
            total_reward = float(np.clip(total_raw, cfg.total_reward_clip_min, cfg.total_reward_clip_max))
            reward_clipped = float(total_reward != total_raw)

        # ─────────────────────────────────────────────────────────────────────
        # DIAGNOSTIC COMPONENTS
        # ─────────────────────────────────────────────────────────────────────
        
        components = {
            # Base reward components
            "base_safety": r_safety,
            "base_comfort": r_comfort,
            "base_comfort_guard": r_comfort_guard,
            "base_efficiency": r_efficiency,
            "base_action_change": r_action_change,
            "base_reward": base_reward,
            # Context reward
            "context_reward": context_reward,
            "context_reward_scale": self._effective_context_scale,
            "context_reward_scaled": self._effective_context_scale * context_reward,
            "context_base_ratio": abs(context_reward) / (abs(base_reward) + 1e-6),
            "context_base_ratio_scaled": abs(self._effective_context_scale * context_reward) / (abs(base_reward) + 1e-6),
            "context_stability": r_stability if (self.use_context_reward and context is not None and self._effective_context_scale > 0.0) else 0.0,
            "context_scenario_gain": scenario_gain if (self.use_context_reward and context is not None and self._effective_context_scale > 0.0) else 0.0,
            # Context variables
            "context_stress": context.stress_level if context else 0.0,
            "context_fatigue": context.fatigue_level if context else 0.0,
            "context_h_pref": context.preferred_headway if context else 0.0,
            "ema_abs_jerk": self._ema_abs_jerk,
            "ema_pos_accel": self._ema_pos_accel,
            # Physical state
            "gap": gap,
            "headway": headway,
            "v_ego": v_ego,
            "cmd_delta": cmd_delta,
            # Totals
            "total_raw": base_reward + self._effective_context_scale * context_reward,
            "total": total_reward,
            "reward_clipped": reward_clipped,
        }

        # Update running diagnostics for callback-side monitoring.
        self._ratio_latest = components["context_base_ratio"]
        self._ratio_scaled_latest = components["context_base_ratio_scaled"]
        self._ratio_samples += 1
        if self._ratio_samples == 1:
            self._ratio_ema = self._ratio_latest
        else:
            self._ratio_ema = 0.98 * self._ratio_ema + 0.02 * self._ratio_latest

        # Debug logging if enabled
        if cfg.reward_debug and context is not None:
            self._debug_counter += 1
            if self._debug_counter % max(1, cfg.reward_debug_interval) == 0:
                self._logger.info(
                    "Reward decomp | base=%.2f context=%.2f scale=%.3f total=%.2f | "
                    "stress=%.2f fatigue=%.2f h_pref=%.2f",
                    base_reward,
                    context_reward,
                    self._effective_context_scale,
                    total_reward,
                    context.stress_level,
                    context.fatigue_level,
                    context.preferred_headway,
                )

        return total_reward, components


# ---------------------------------------------------------------------------
# Core Gymnasium Environment
# ---------------------------------------------------------------------------

class CarFollowingEnv(gym.Env):
    """
    1D longitudinal car-following environment.

    Observation space
    -----------------
    Mode A (use_context=False):  Box(5,)  = [d, v_ego, v_lead, a, j]
    Mode B (use_context=True):   Box(10,) = [d, v_ego, v_lead, a, j,
                                              κ_stress, κ_fatigue, h_pref,
                                              κ_comfort, κ_eff]

    Action space
    ------------
    Box(1,) in [a_min, a_max] m/s²   (continuous longitudinal acceleration)

    Episode info dict keys
    ----------------------
    gap, v_ego, v_lead, accel, jerk, headway, ttc_proxy,
    collision, regime, context_stress, context_fatigue,
    context_h_pref, context_comfort, context_eff,
    r_safe, r_comfort, r_eff, r_context, reward, reward_components

    Parameters
    ----------
    config:
        Master Config dataclass.
    use_context:
        Whether to include context vector in the observation.
    use_context_reward:
        Whether to use context-modulated reward terms.
    lead_profile:
        One of the named profiles or None for random selection each episode.
    scenario:
        Evaluation scenario name ("calm", "stressed", "fatigued", "both",
        "switching", "escalation", "mixed"). Controls how the context
        regime is initialised / allowed to switch.
    seed:
        Episode-level seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        use_context: bool = True,
        use_context_reward: bool = True,
        lead_profile: Optional[str] = None,
        scenario: str = "mixed",
        seed: int = 0,
        normalize_observation: bool = True,
        training_safety_shield: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.use_context = use_context
        self.use_context_reward = use_context_reward
        self.normalize_observation = normalize_observation
        self.training_safety_shield = training_safety_shield
        self.lead_profile_name = lead_profile
        self.scenario = scenario
        self._base_seed = seed

        cfg_sim = config.sim
        cfg_rew = config.reward
        cfg_lead = config.lead
        cfg_human = config.human

        # Gymnasium action space
        self.action_space = spaces.Box(
            low=np.float32(cfg_sim.a_min),
            high=np.float32(cfg_sim.a_max),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation space
        if normalize_observation:
            phys_lo = np.array([0.0, 0.0, 0.0, -1.0, -2.0], dtype=np.float32)
            phys_hi = np.array([2.0, 1.2, 1.2, 1.0, 2.0], dtype=np.float32)
            if use_context:
                ctx_lo = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                ctx_hi = np.array([1.0, 1.0, 1.2, 1.0, 1.0], dtype=np.float32)
                obs_lo = np.concatenate([phys_lo, ctx_lo])
                obs_hi = np.concatenate([phys_hi, ctx_hi])
            else:
                obs_lo = phys_lo
                obs_hi = phys_hi
        else:
            phys_lo = np.array([0.0, 0.0, 0.0, cfg_sim.a_min, -10.0], dtype=np.float32)
            phys_hi = np.array([
                cfg_sim.gap_max,
                cfg_sim.v_max,
                cfg_sim.v_max,
                cfg_sim.a_max,
                10.0,
            ], dtype=np.float32)
            if use_context:
                ctx_lo = np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
                ctx_hi = np.array([1.0, 1.0, 5.0, 1.0, 1.0], dtype=np.float32)
                obs_lo = np.concatenate([phys_lo, ctx_lo])
                obs_hi = np.concatenate([phys_hi, ctx_hi])
            else:
                obs_lo = phys_lo
                obs_hi = phys_hi

        self.observation_space = spaces.Box(
            low=obs_lo, high=obs_hi, dtype=np.float32
        )

        # Sub-components
        self._rng = np.random.default_rng(seed)
        self._lead = LeadVehicleProfile(
            profile=lead_profile if lead_profile is not None else "constant",
            cfg_lead=cfg_lead,
            cfg_sim=cfg_sim,
            rng=self._rng,
        )
        self._context_model = make_context_model(cfg_human, seed=seed, scenario=scenario)
        self._reward_calc = RewardCalculator(cfg_sim, cfg_rew, use_context_reward)

        # Action smoothing parameters (from reward config)
        self._action_smoothing_enabled = cfg_rew.action_smoothing_enabled
        self._action_smoothing_alpha_vanilla = cfg_rew.action_smoothing_alpha_vanilla
        self._action_smoothing_alpha_context = cfg_rew.action_smoothing_alpha_context

        # Episode state (initialised in reset)
        self._step_count: int = 0
        self._gap: float = cfg_sim.initial_gap
        self._v_ego: float = cfg_sim.initial_ego_speed
        self._v_lead: float = cfg_sim.initial_lead_speed
        self._accel: float = 0.0
        self._prev_accel: float = 0.0
        self._jerk: float = 0.0
        self._prev_action_cmd: float = 0.0
        self._context: Optional[ContextOutput] = None
        self._episode_rng: Optional[np.random.Generator] = None
        self._curriculum_stage: int = 1
        self._episode_scenario: str = scenario
        self._safety_shield_active: bool = False
        # Hysteresis safeguard state (anti-chattering)
        self._context_guard_active: bool = False
        # Per-episode intervention diagnostics
        self._guard_engage_count: int = 0
        self._guard_disengage_count: int = 0
        self._guard_active_steps: int = 0
        self._recovery_hold_total_steps: int = 0  # track total recovery hold time per episode
        self._cmd_change_history: list = []  # |Δa_cmd| per step for p95 diagnostic
        # Recovery hold: suppress acceleration after braking to reduce chatter
        self._recovery_hold_steps: int = 0

    def set_curriculum_stage(self, stage: int) -> None:
        """Set current curriculum stage for training env resets."""
        self._curriculum_stage = int(np.clip(stage, 1, 3))

    def set_training_progress(self, progress: float) -> None:
        """
        Update training progress for both curriculum and reward warmup.
        
        Args:
            progress: Training progress as fraction of total timesteps (0.0 to 1.0)
        """
        # Curriculum stage mapping
        cur = self.config.curriculum
        if not cur.enabled:
            self._curriculum_stage = 3
        elif progress < cur.stage1_fraction:
            self._curriculum_stage = 1
        elif progress < cur.stage2_fraction:
            self._curriculum_stage = 2
        else:
            self._curriculum_stage = 3
        
        # Reward warmup scaling
        self._reward_calc.set_training_fraction(progress)

    def get_reward_diagnostics(self) -> Dict[str, float]:
        """Expose reward-balance diagnostics to vectorized training callbacks."""
        return self._reward_calc.get_diagnostics()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and return (observation, info)."""
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            # Advance the RNG so each episode is different even without explicit seed
            seed_in = int(self._rng.integers(0, 2**31))
            self._rng = np.random.default_rng(seed_in)

        self._episode_rng = self._rng

        # Choose lead profile and context scenario
        lead_name, episode_scenario = self._select_episode_setup()
        self._episode_scenario = episode_scenario
        self._lead = LeadVehicleProfile(
            profile=lead_name,
            cfg_lead=self.config.lead,
            cfg_sim=self.config.sim,
            rng=self._rng,
        )

        # Randomise initial conditions slightly
        cfg = self.config.sim
        self._gap = cfg.initial_gap + self._rng.uniform(-3.0, 3.0)
        self._v_ego = cfg.initial_ego_speed + self._rng.uniform(-2.0, 2.0)
        self._v_lead = cfg.initial_lead_speed + self._rng.uniform(-2.0, 2.0)
        self._accel = 0.0
        self._prev_accel = 0.0
        self._jerk = 0.0
        self._prev_action_cmd = 0.0
        self._step_count = 0

        # Reset lead trajectory starting from initial lead speed
        self._lead.reset(initial_speed=self._v_lead)

        # Reset context model
        context_seed = int(self._rng.integers(0, 2**31 - 1))
        context_scenario = "mixed" if episode_scenario == "switching" else episode_scenario
        self._context_model = make_context_model(self.config.human, seed=context_seed, scenario=context_scenario)
        initial_regime = None
        if episode_scenario in ("calm", "stressed", "fatigued", "both"):
            initial_regime = episode_scenario
        self._context = self._context_model.reset(initial_regime=initial_regime)
        self._safety_shield_active = False
        self._context_guard_active = False
        self._guard_engage_count = 0
        self._guard_disengage_count = 0
        self._guard_active_steps = 0
        self._recovery_hold_total_steps = 0
        self._cmd_change_history = []
        self._recovery_hold_steps = 0
        self._reward_calc.reset_episode()

        info = self._build_info(reward_components=None)
        return self._build_obs(), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the simulation one timestep.

        Returns (obs, reward, terminated, truncated, info).
        """
        cfg = self.config.sim
        dt = cfg.dt

        # Clip and extract action
        a_cmd = float(np.clip(action[0], cfg.a_min, cfg.a_max))
        prev_cmd = self._prev_action_cmd

        # Training-time safety shield (applies to all agents; no context required)
        self._safety_shield_active = False
        if self.training_safety_shield and self._gap < cfg.emergency_gap and self._v_ego > self._v_lead:
            a_cmd = min(a_cmd, cfg.emergency_brake)
            self._safety_shield_active = True

        # ── Context-aware safeguard with HYSTERESIS (context PPO only) ──────────
        # Prevents rapid on/off chattering near risk thresholds by using separate
        # engage and release thresholds (release must be looser than engage).
        if self.use_context and self.use_context_reward and self._context is not None:
            cfg_rew = self.config.reward
            rel_speed = self._v_ego - self._v_lead
            ttc = self._gap / rel_speed if rel_speed > 0.1 else 999.0
            stress = self._context.stress_level
            fatigue = self._context.fatigue_level

            # Engage thresholds (tighter — guard activates here)
            engage_gap = cfg.emergency_gap + 1.5 * stress + 1.0 * fatigue
            engage_ttc = 2.2 + 0.8 * stress
            # Release thresholds (looser — guard deactivates only when clearly safe)
            release_gap = engage_gap + cfg_rew.context_guard_hysteresis_gap
            release_ttc = engage_ttc + cfg_rew.context_guard_hysteresis_ttc

            prev_guard = self._context_guard_active
            in_risk = (self._gap < engage_gap) or (ttc < engage_ttc)
            out_of_risk = (self._gap > release_gap) and (ttc > release_ttc)

            if in_risk and not self._context_guard_active:
                self._context_guard_active = True
                self._guard_engage_count += 1
            elif out_of_risk and self._context_guard_active:
                self._context_guard_active = False
                self._guard_disengage_count += 1

            if self._context_guard_active:
                brake_limit = cfg.emergency_brake - 0.6 * stress - 0.4 * fatigue
                a_cmd = min(a_cmd, max(cfg.a_min, brake_limit))
                self._safety_shield_active = True
                self._guard_active_steps += 1

            # ── Compute continuous risk_score for adaptive limits ────────────
            # risk_score ∈ [0, 1]: 0 = fully safe, 1 = high risk
            gap_risk = float(np.clip(1.0 - (self._gap - engage_gap) / max(release_gap - engage_gap, 1e-3), 0.0, 1.0))
            ttc_risk = float(np.clip(1.0 - (ttc - engage_ttc) / max(release_ttc - engage_ttc, 1e-3), 0.0, 1.0))
            risk_score = max(gap_risk, ttc_risk)

            # ── Recovery hold: suppress acceleration after braking to reduce chatter ──
            # When guard disengages or strong braking occurs, hold tighter control briefly.
            if self._context_guard_active and self._recovery_hold_steps == 0:
                # Guard just engaged or is active → start recovery hold
                self._recovery_hold_steps = cfg_rew.recovery_hold_steps
            elif a_cmd < cfg.emergency_brake + 0.1 and prev_cmd >= cfg.emergency_brake:
                # Just braked heavily → start recovery hold
                self._recovery_hold_steps = cfg_rew.recovery_hold_steps

            # Apply recovery hold effect: tighter slew limits, stronger smoothing
            in_recovery = self._recovery_hold_steps > 0
            if in_recovery:
                # Reduce upward slew during recovery; keep downward slew for safety
                max_up = cfg_rew.slew_up_low_risk * cfg_rew.recovery_hold_slew_up_scale
                max_down = cfg_rew.slew_down_low_risk
                recovery_alpha_bonus = cfg_rew.recovery_hold_alpha_bonus
                self._recovery_hold_total_steps += 1
                self._recovery_hold_steps -= 1
            else:
                # Normal operation: interpolate slew limits based on risk
                max_up = cfg_rew.slew_up_low_risk + risk_score * (cfg_rew.slew_up_high_risk - cfg_rew.slew_up_low_risk)
                max_down = cfg_rew.slew_down_low_risk + risk_score * (cfg_rew.slew_down_high_risk - cfg_rew.slew_down_low_risk)
                recovery_alpha_bonus = 0.0

            # ── Command deadzone: suppress micro-oscillations ───────────────
            # If proposed change is below deadzone AND risk is low, keep previous command
            deadzone = cfg_rew.cmd_deadzone_low_risk if risk_score < 0.3 else cfg_rew.cmd_deadzone_mid_risk
            if abs(a_cmd - prev_cmd) < deadzone:
                cmd_to_use = prev_cmd
            else:
                cmd_to_use = a_cmd

            # Apply slew-rate limit
            a_cmd = float(np.clip(cmd_to_use, prev_cmd - max_down, prev_cmd + max_up))

            # ── Adaptive action smoothing ─────────────────────────────────────
            # Low risk → heavy smoothing; high risk → lighter smoothing.
            # Add extra smoothing during recovery hold.
            alpha_lo = cfg_rew.context_smoothing_alpha_low_risk   # e.g. 0.85
            alpha_hi = cfg_rew.context_smoothing_alpha_high_risk   # e.g. 0.35
            smooth_alpha = alpha_lo + risk_score * (alpha_hi - alpha_lo)
            if in_recovery:
                smooth_alpha = min(1.0, smooth_alpha + recovery_alpha_bonus)
            if self._action_smoothing_enabled:
                a_cmd = smooth_alpha * prev_cmd + (1.0 - smooth_alpha) * a_cmd
        else:
            # Vanilla / ablation: apply fixed light smoothing only; no recovery hold or deadzone
            if self._action_smoothing_enabled:
                a_cmd = self._action_smoothing_alpha_vanilla * prev_cmd + (
                    1.0 - self._action_smoothing_alpha_vanilla
                ) * a_cmd

        # Track command change for diagnostics and the action-change reward penalty
        cmd_delta = abs(a_cmd - prev_cmd)
        self._cmd_change_history.append(cmd_delta)
        self._prev_action_cmd = a_cmd

        # Lead vehicle update
        v_lead_new = self._lead.speed_at(self._step_count)

        # Ego vehicle kinematics
        v_ego_new = float(np.clip(self._v_ego + a_cmd * dt, cfg.v_min, cfg.v_max))

        # Distance update: Δgap = (v_lead - v_ego) * dt
        gap_new = self._gap + (v_lead_new - v_ego_new) * dt
        gap_new = float(np.clip(gap_new, -1.0, cfg.gap_max))

        # Jerk
        self._prev_accel = self._accel
        actual_accel = (v_ego_new - self._v_ego) / dt
        jerk = (actual_accel - self._prev_accel) / dt

        # Update state
        self._v_ego = v_ego_new
        self._v_lead = v_lead_new
        self._gap = gap_new
        self._accel = actual_accel
        self._jerk = jerk
        self._step_count += 1

        # Advance context model
        self._context = self._context_model.step()

        # Compute reward
        reward, r_comp = self._reward_calc.compute(
            gap=self._gap,
            v_ego=self._v_ego,
            v_lead=self._v_lead,
            accel=self._accel,
            jerk=self._jerk,
            context=self._context if (self.use_context_reward and self._context is not None) else None,
            v_target=self.config.sim.target_speed,
            episode_scenario=self._episode_scenario,
            cmd_delta=cmd_delta,
        )

        # Termination
        terminated = self._gap <= cfg.collision_gap
        truncated = self._step_count >= cfg.episode_steps

        info = self._build_info(reward_components=r_comp)
        return self._build_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        pass  # No visual rendering required

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Construct the observation vector."""
        cfg = self.config.sim
        gap_obs = float(np.clip(self._gap, 0.0, cfg.gap_max))
        phys = np.array([
            gap_obs,
            self._v_ego,
            self._v_lead,
            self._accel,
            self._jerk,
        ], dtype=np.float32)

        if self.normalize_observation:
            phys = np.array([
                np.clip(gap_obs / cfg.gap_norm, 0.0, 2.0),
                np.clip(self._v_ego / cfg.speed_norm, 0.0, 1.2),
                np.clip(self._v_lead / cfg.speed_norm, 0.0, 1.2),
                np.clip(self._accel / cfg.accel_norm, -1.0, 1.0),
                np.clip(self._jerk / cfg.jerk_norm, -2.0, 2.0),
            ], dtype=np.float32)

        if not self.use_context or self._context is None:
            return phys

        ctx = self._context.to_array()
        if self.normalize_observation:
            ctx = np.array([
                np.clip(ctx[0], 0.0, 1.0),
                np.clip(ctx[1], 0.0, 1.0),
                np.clip(ctx[2] / cfg.headway_pref_norm, 0.0, 1.2),
                np.clip(ctx[3], 0.0, 1.0),
                np.clip(ctx[4], 0.0, 1.0),
            ], dtype=np.float32)
        return np.concatenate([phys, ctx], axis=0)

    def _select_episode_setup(self) -> Tuple[str, str]:
        """Return (lead_profile, scenario) for the next episode."""
        if self.lead_profile_name is not None and self.scenario != "curriculum":
            return self.lead_profile_name, self.scenario

        if self.scenario != "curriculum":
            lead_name = self.lead_profile_name or str(self._rng.choice(self.config.lead.profiles))
            return lead_name, self.scenario

        cur = self.config.curriculum
        if self._curriculum_stage == 1:
            profiles = cur.stage1_profiles
            scenarios = cur.stage1_scenarios
        elif self._curriculum_stage == 2:
            profiles = cur.stage2_profiles
            scenarios = cur.stage2_scenarios
        else:
            profiles = cur.stage3_profiles
            scenarios = cur.stage3_scenarios

        lead_name = self.lead_profile_name or str(self._rng.choice(profiles))
        scenario_name = str(self._rng.choice(scenarios))
        return lead_name, scenario_name

    def _build_info(
        self, reward_components: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Collect all diagnostics into an info dict."""
        cfg = self.config.sim
        headway = self._gap / (self._v_ego + cfg.eps)
        # TTC proxy: gap / (v_ego - v_lead) when ego is faster
        rel_speed = self._v_ego - self._v_lead
        ttc = self._gap / rel_speed if rel_speed > 0.1 else 999.0

        ctx = self._context
        info: Dict[str, Any] = {
            "gap":             self._gap,
            "v_ego":           self._v_ego,
            "v_lead":          self._v_lead,
            "accel":           self._accel,
            "jerk":            self._jerk,
            "headway":         headway,
            "ttc_proxy":       ttc,
            "collision":       int(self._gap <= cfg.collision_gap),
            "regime":          self._context_model.current_regime,
            "episode_scenario": self._episode_scenario,
            "curriculum_stage": self._curriculum_stage,
            "safety_shield_active": int(self._safety_shield_active),
            "step":            self._step_count,
            # Intervention-chatter diagnostics (updated each step)
            "guard_engage_count":    self._guard_engage_count,
            "guard_disengage_count": self._guard_disengage_count,
            "guard_active_steps":    self._guard_active_steps,
            "guard_active_frac":     (
                self._guard_active_steps / max(1, self._step_count)
            ),
            "mean_abs_cmd_change": float(np.mean(self._cmd_change_history)) if self._cmd_change_history else 0.0,
            "p95_cmd_change":      float(np.percentile(self._cmd_change_history, 95)) if self._cmd_change_history else 0.0,
            "recovery_hold_frac":   (
                self._recovery_hold_total_steps / max(1, self._step_count)
            ),
        }

        # Context values (safe when context is None for vanilla env)
        if ctx is not None:
            info.update({
                "context_stress":   ctx.stress_level,
                "context_fatigue":  ctx.fatigue_level,
                "context_h_pref":   ctx.preferred_headway,
                "context_comfort":  ctx.comfort_priority,
                "context_eff":      ctx.efficiency_priority,
            })
        else:
            info.update({
                "context_stress":  0.0,
                "context_fatigue": 0.0,
                "context_h_pref":  cfg.safe_headway,
                "context_comfort": 0.5,
                "context_eff":     0.5,
            })

        # Add enhanced jerk diagnostics (from reward components if available)
        info.update({
            "ema_abs_jerk": reward_components.get("ema_abs_jerk", 0.0) if reward_components else 0.0,
            "ema_pos_accel": reward_components.get("ema_pos_accel", 0.0) if reward_components else 0.0,
        })

        if reward_components is not None:
            structured_components = {
                # New reward decomposition
                "base_safety": reward_components.get("base_safety", 0.0),
                "base_comfort": reward_components.get("base_comfort", 0.0),
                "base_efficiency": reward_components.get("base_efficiency", 0.0),
                "base_action_change": reward_components.get("base_action_change", 0.0),
                "base_reward": reward_components.get("base_reward", 0.0),
                "context_reward": reward_components.get("context_reward", 0.0),
                "context_reward_scale": reward_components.get("context_reward_scale", 0.0),
                "context_reward_scaled": reward_components.get("context_reward_scaled", 0.0),
                "context_stability": reward_components.get("context_stability", 0.0),
                "context_base_ratio": reward_components.get("context_base_ratio", 0.0),
                "context_base_ratio_scaled": reward_components.get("context_base_ratio_scaled", 0.0),
                "cmd_delta": reward_components.get("cmd_delta", 0.0),
                # Legacy aliases for downstream compatibility
                "r_safe": reward_components.get("base_safety", reward_components.get("r_safe", 0.0)),
                "r_comfort": reward_components.get("base_comfort", reward_components.get("r_comfort", 0.0)),
                "r_eff": reward_components.get("base_efficiency", reward_components.get("r_eff", 0.0)),
                "r_context": reward_components.get("context_reward", reward_components.get("r_context", 0.0)),
                # Shared totals
                "total_raw": reward_components.get("total_raw", 0.0),
                "total": reward_components.get("total", 0.0),
                "reward_clipped": reward_components.get("reward_clipped", 0.0),
            }
            info.update({
                # New primary reward fields
                "base_safety": reward_components.get("base_safety", 0.0),
                "base_comfort": reward_components.get("base_comfort", 0.0),
                "base_efficiency": reward_components.get("base_efficiency", 0.0),
                "base_reward": reward_components.get("base_reward", 0.0),
                "context_reward": reward_components.get("context_reward", 0.0),
                "context_reward_scale": reward_components.get("context_reward_scale", 0.0),
                "context_reward_scaled": reward_components.get("context_reward_scaled", 0.0),
                "context_stability": reward_components.get("context_stability", 0.0),
                "context_base_ratio": reward_components.get("context_base_ratio", 0.0),
                "context_base_ratio_scaled": reward_components.get("context_base_ratio_scaled", 0.0),
                # Legacy fields
                "r_safe":    reward_components.get("base_safety", reward_components.get("r_safe", 0.0)),
                "r_comfort": reward_components.get("base_comfort", reward_components.get("r_comfort", 0.0)),
                "r_eff":     reward_components.get("base_efficiency", reward_components.get("r_eff", 0.0)),
                "r_context": reward_components.get("context_reward", reward_components.get("r_context", 0.0)),
                "r_distance_buffer": reward_components.get("r_distance_buffer", 0.0),
                "reward_raw": reward_components.get("total_raw", 0.0),
                "reward_clipped": reward_components.get("reward_clipped", 0.0),
                "reward":    reward_components.get("total", 0.0),
                "reward_components": structured_components,
            })

        return info


# ---------------------------------------------------------------------------
# Environment factory functions (used by train.py and evaluate.py)
# ---------------------------------------------------------------------------

def make_env(
    config: Config,
    use_context: bool,
    use_context_reward: bool,
    seed: int,
    lead_profile: Optional[str] = None,
    scenario: str = "mixed",
    normalize_observation: bool = True,
    training_safety_shield: bool = False,
) -> CarFollowingEnv:
    """Factory for a single CarFollowingEnv (compatible with SB3 make_vec_env)."""
    def _init() -> CarFollowingEnv:
        env = CarFollowingEnv(
            config=config,
            use_context=use_context,
            use_context_reward=use_context_reward,
            lead_profile=lead_profile,
            scenario=scenario,
            seed=seed,
            normalize_observation=normalize_observation,
            training_safety_shield=training_safety_shield,
        )
        return env
    return _init


def make_single_env(
    config: Config,
    use_context: bool,
    use_context_reward: bool,
    seed: int,
    lead_profile: Optional[str] = None,
    scenario: str = "mixed",
    normalize_observation: bool = True,
    training_safety_shield: bool = False,
) -> CarFollowingEnv:
    """Return a concrete (unwrapped) CarFollowingEnv instance."""
    return CarFollowingEnv(
        config=config,
        use_context=use_context,
        use_context_reward=use_context_reward,
        lead_profile=lead_profile,
        scenario=scenario,
        seed=seed,
        normalize_observation=normalize_observation,
        training_safety_shield=training_safety_shield,
    )
