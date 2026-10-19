"""
config.py
=========
Centralised configuration for the Context-Aware Car-Following project.

All hyper-parameters are exposed here via Python dataclasses so that the rest
of the code-base can import a single configuration object and every script
remains consistent.  Change values here – or supply them via CLI overrides –
rather than scattering magic numbers through the modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """File-system locations used throughout the project."""
    root: Path = Path("artifacts")
    models: Path = field(init=False)
    results: Path = field(init=False)
    plots: Path = field(init=False)

    def __post_init__(self) -> None:
        self.models = self.root / "models"
        self.results = self.root / "results"
        self.plots = self.root / "plots"

    def make_dirs(self) -> None:
        """Create all required output directories."""
        for d in (self.models, self.results, self.plots):
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Simulation / Environment
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Physical simulation parameters."""
    dt: float = 0.1                    # timestep  [s]
    episode_steps: int = 500           # max steps per episode
    initial_gap: float = 20.0          # initial distance to lead vehicle [m]
    initial_ego_speed: float = 12.0    # ego speed at episode start [m/s]
    initial_lead_speed: float = 12.0   # lead speed at episode start [m/s]
    target_speed: float = 15.0         # nominal desired speed [m/s]
    v_min: float = 0.0                 # minimum allowed speed [m/s]
    v_max: float = 33.0                # maximum allowed speed [m/s]
    a_min: float = -3.0                # minimum acceleration command [m/s²]
    a_max: float = 2.0                 # maximum acceleration command [m/s²]
    gap_min: float = 1.0               # hard minimum gap (collision boundary) [m]
    gap_max: float = 100.0             # observation clipping for gap [m]
    safe_headway: float = 1.5          # minimum safe time-headway threshold [s]
    headway_speed_floor: float = 0.1   # floor speed for reward headway computation [m/s]
    headway_clip_max: float = 8.0      # max headway used inside reward terms [s]
    emergency_gap: float = 4.0         # training-time safety shield trigger [m]
    emergency_brake: float = -2.5      # shielded minimum braking [m/s²]
    eps: float = 1e-6                  # small value to avoid division by zero
    collision_gap: float = 0.0         # gap at which a collision is declared [m]

    # Manual observation scaling for PPO
    gap_norm: float = 50.0
    speed_norm: float = 30.0
    accel_norm: float = 5.0
    jerk_norm: float = 10.0
    headway_pref_norm: float = 3.0


# ---------------------------------------------------------------------------
# Human-context model
# ---------------------------------------------------------------------------

@dataclass
class HumanConfig:
    """Parameters for the synthetic driver-state and context-interpreter."""
    # Latent regime transition probabilities per step  (calm / stressed / fatigued / both)
    regime_hold_steps: int = 80        # average steps before a regime switch
    noise_std: float = 0.03            # noise added to inferred context values

    # Physiological baseline ranges  [min, max]
    hr_calm: Tuple[float, float] = (65.0, 75.0)
    hr_stressed: Tuple[float, float] = (85.0, 100.0)
    hr_fatigued: Tuple[float, float] = (60.0, 70.0)
    hr_both: Tuple[float, float] = (88.0, 105.0)

    hrv_calm: Tuple[float, float] = (55.0, 70.0)
    hrv_stressed: Tuple[float, float] = (20.0, 35.0)
    hrv_fatigued: Tuple[float, float] = (35.0, 50.0)
    hrv_both: Tuple[float, float] = (15.0, 30.0)

    resp_calm: Tuple[float, float] = (14.0, 16.0)
    resp_stressed: Tuple[float, float] = (18.0, 22.0)
    resp_fatigued: Tuple[float, float] = (12.0, 15.0)
    resp_both: Tuple[float, float] = (19.0, 24.0)

    fatigue_calm: Tuple[float, float] = (0.05, 0.15)
    fatigue_stressed: Tuple[float, float] = (0.10, 0.25)
    fatigue_fatigued: Tuple[float, float] = (0.55, 0.80)
    fatigue_both: Tuple[float, float] = (0.60, 0.85)

    # Headway preferences per regime [s]
    headway_pref_calm: float = 1.4
    headway_pref_stressed: float = 2.2
    headway_pref_fatigued: float = 2.2
    headway_pref_both: float = 3.0

    regimes: Tuple[str, ...] = ("calm", "stressed", "fatigued", "both")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """
    Reward function weights with explicit base/context separation.
    
    Core design:
      total_reward = base_reward + context_reward_scale * context_reward
    
    where base_reward captures safety/comfort/efficiency (shared objective),
    and context_reward = small preference shaper (stressed: safer, fatigued: smoother).
    """
    # ─────────────────────────────────────────────────────────────────────
    # BASE REWARD WEIGHTS (safety, comfort, efficiency)
    # ─────────────────────────────────────────────────────────────────────
    
    # Safety: collision and headway violations (DOMINANT)
    w_collision: float = 200.0         # terminal collision penalty
    w_gap: float = 7.0                 # mild reduction so context preference has room
    w_distance_buffer: float = 1.5     # mild reduction to avoid over-dominant base penalties
    
    # Comfort: jerk and acceleration (moderate)
    alpha_accel: float = 0.15          # accel contribution to comfort
    jerk_weight: float = 0.17          # balanced jerk penalty
    comfort_guard_weight: float = 0.20
    comfort_guard_jerk_threshold: float = 0.8
    
    # Efficiency: speed tracking (minor)
    efficiency_weight: float = 0.3
    
    # ─────────────────────────────────────────────────────────────────────
    # CONTEXT REWARD SCALE (key parameter: keep small!)
    # ─────────────────────────────────────────────────────────────────────
    
    context_reward_scale: float = 0.40  # increased to push contribution back to 10-15% range
                                        # set to 0.0 to disable context shaping
    
    # ─────────────────────────────────────────────────────────────────────
    # CONTEXT REWARD COMPONENTS (preference shapers, not dominators)
    # ─────────────────────────────────────────────────────────────────────
    
    # Stressed: prefer smoother control and larger headway
    stress_jerk_penalty_weight: float = 0.20   # mild stress-time jerk suppression
    # One-sided headway shortfall penalty (replaces symmetric exp(-|error|) bonus)
    # Only penalizes when headway < h_pref - deadband; no pressure to over-chase
    stress_headway_shortfall_weight: float = 0.40  # weight on quadratic shortfall penalty
    context_headway_deadband: float = 0.25          # [s] slightly larger to reduce pressure in safe regions
    # Legacy: stress_headway_bonus_weight kept for backward compat but unused in new logic
    stress_headway_bonus_weight: float = 0.0   # disabled: replaced by one-sided shortfall

    # Fatigued: prefer less aggressive positive acceleration
    fatigue_accel_penalty_weight: float = 0.12  # moderate penalty for aggressive accel
    fatigue_smooth_bonus_weight: float = 0.60   # further reduced to avoid excessive smoothness chasing
    fatigue_aggressive_accel_threshold: float = 0.6

    # Both: combine caution from stress + comfort from fatigue
    both_context_extra_caution_weight: float = 0.30  # reduced combined-condition bonus for stability
    both_short_gap_penalty_weight: float = 0.05      # small penalty only when gap too short
    context_jerk_stability_weight: float = 0.16      # balanced always-on damping for jerk control

    # Structural controls to prevent context spikes and oscillatory incentives
    context_bonus_jerk_gate_scale: float = 0.9        # smaller -> stronger bonus suppression at high jerk
    context_component_clip: float = 1.2               # mild headroom without allowing spikes
    context_ema_alpha: float = 0.20                   # smoothing for jerk/accel-driven context terms

    # ─────────────────────────────────────────────────────────────────────
    # HYSTERESIS FOR CONTEXT SAFETY SAFEGUARD (anti-chattering)
    # ─────────────────────────────────────────────────────────────────────
    # Guard engages when risk > engage threshold; stays on until risk drops below release threshold.
    # release > engage to prevent on/off toggling near threshold.
    context_guard_hysteresis_ttc: float = 0.6   # [s] release margin above engage_ttc
    context_guard_hysteresis_gap: float = 1.0   # [m] release margin above engage_gap

    # ─────────────────────────────────────────────────────────────────────
    # ADAPTIVE SLEW-RATE LIMITING (rate-limit command changes by risk level)
    # ─────────────────────────────────────────────────────────────────────
    # Second-pass refinement: stricter in low/mid-risk to reduce oscillation
    slew_up_low_risk: float = 0.05    # [m/s²/step] reduced for less oscillation
    slew_down_low_risk: float = 0.25  # [m/s²/step] slightly increased for smoother braking
    slew_up_high_risk: float = 0.05   # [m/s²/step] keep minimal for safety
    slew_down_high_risk: float = 0.70 # [m/s²/step] increased for emergency braking authority

    # ─────────────────────────────────────────────────────────────────────
    # ADAPTIVE ACTION SMOOTHING (risk-sensitive alpha for context mode)
    # ─────────────────────────────────────────────────────────────────────
    # Second-pass: heavier smoothing in low/mid-risk to suppress jerk
    action_smoothing_enabled: bool = True
    action_smoothing_alpha_vanilla: float = 0.30        # fixed for vanilla PPO
    action_smoothing_alpha_context: float = 0.55        # fallback if adaptive is disabled
    context_smoothing_alpha_low_risk: float = 0.85      # heavily damped in safe driving
    context_smoothing_alpha_mid_risk: float = 0.65      # more damped in moderate risk
    context_smoothing_alpha_high_risk: float = 0.35     # allow responsiveness only at true risk

    # ─────────────────────────────────────────────────────────────────────
    # COMMAND DEADZONE FOR MICRO-OSCILLATION SUPPRESSION
    # ─────────────────────────────────────────────────────────────────────
    # Suppress tiny adjustments that contribute little to safety but inflate jerk
    cmd_deadzone_low_risk: float = 0.02  # [m/s²] ignore changes < 0.02 when safe
    cmd_deadzone_mid_risk: float = 0.01  # [m/s²] slightly relax deadzone at medium risk

    # ─────────────────────────────────────────────────────────────────────
    # RECOVERY HOLD: BRAKE-RELEASE DAMPER
    # ─────────────────────────────────────────────────────────────────────
    # After guard engages or strong braking, hold tighter control for a brief period
    # to prevent chatter on acceleration restart
    recovery_hold_steps: int = 6              # [steps] how long to hold after braking
    recovery_hold_slew_up_scale: float = 0.5  # fraction of normal slew_up allowed during hold
    recovery_hold_alpha_bonus: float = 0.15   # additional smoothing alpha during recovery hold

    # ─────────────────────────────────────────────────────────────────────
    # RISK-AWARE JERK PENALTY MULTIPLIERS (safety-first: penalize jerk more in safe regimes)
    # ─────────────────────────────────────────────────────────────────────
    low_risk_jerk_multiplier: float = 1.5    # penalize jerk 50% more strongly when safe
    mid_risk_jerk_multiplier: float = 1.0    # baseline jerk penalty at medium risk
    high_risk_jerk_multiplier: float = 0.5   # allow more jerk during true emergency

    # ─────────────────────────────────────────────────────────────────────
    # BASE REWARD: ACTION-CHANGE PENALTY (discourages all agents from oscillating)
    # ─────────────────────────────────────────────────────────────────────
    action_change_penalty_weight: float = 0.06  # increased modestly for stronger smoothing pressure

    # Scenario-dependent context gains (lower in switching/both, higher in stressed)
    scenario_gain_stressed: float = 1.05
    scenario_gain_fatigued: float = 0.95
    scenario_gain_both: float = 0.70
    scenario_gain_switching: float = 0.85
    scenario_gain_default: float = 0.85
    
    # ─────────────────────────────────────────────────────────────────────
    # NUMERICAL STABILITY
    # ─────────────────────────────────────────────────────────────────────
    
    context_reward_clip: float = 4.0   # allow larger bounded context contribution
    total_reward_clip_min: float = -220.0
    total_reward_clip_max: float = 10.0
    
    # ─────────────────────────────────────────────────────────────────────
    # OPTIONAL WARMUP (ramp context reward scale during training)
    # ─────────────────────────────────────────────────────────────────────
    
    context_warmup_enabled: bool = True
    context_warmup_fraction: float = 0.3  # ramp from 0 to context_reward_scale
                                           # during first 30% of training
    
    # ─────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────
    
    reward_debug: bool = False
    reward_debug_interval: int = 2000
    
    # Vanilla PPO fixed weights (for fair comparison)
    beta_eff: float = 1.0
    beta_comfort: float = 1.0


@dataclass
class CurriculumConfig:
    """Stage-based training curriculum parameters."""
    enabled: bool = True
    stage1_fraction: float = 0.30
    stage2_fraction: float = 0.70
    stage1_profiles: Tuple[str, ...] = ("constant", "sinusoidal")
    stage2_profiles: Tuple[str, ...] = ("constant", "sinusoidal", "braking", "stop_and_go")
    stage3_profiles: Tuple[str, ...] = ("constant", "sinusoidal", "braking", "stop_and_go", "random_smooth")
    stage1_scenarios: Tuple[str, ...] = ("calm",)
    stage2_scenarios: Tuple[str, ...] = ("calm", "stressed", "fatigued")
    stage3_scenarios: Tuple[str, ...] = ("switching", "stressed", "fatigued", "both")
    training_safety_shield: bool = True

    # Context PPO efficiency / comfort weights (modulated per step by kappa)
    # (kappa_eff and kappa_comfort come from the context vector at runtime)


# ---------------------------------------------------------------------------
# Lead-vehicle trajectory
# ---------------------------------------------------------------------------

@dataclass
class LeadConfig:
    """Parameters for scripted lead-vehicle trajectories."""
    profiles: List[str] = field(default_factory=lambda: [
        "constant",
        "sinusoidal",
        "braking",
        "stop_and_go",
        "random_smooth",
    ])
    # Sinusoidal params
    sin_amplitude: float = 3.0         # speed oscillation amplitude [m/s]
    sin_period: float = 8.0            # period [s]
    # Braking event params
    brake_decel: float = -2.5          # deceleration magnitude [m/s²]
    brake_duration: float = 3.0        # duration of brake event [s]
    brake_start_frac: float = 0.40     # fraction of episode at which brake starts
    # Stop-and-go params
    sg_cycles: int = 3                 # number of stop-go cycles per episode
    sg_stop_duration: float = 4.0      # time stopped [s]
    # Random smooth params
    smooth_sigma: float = 0.8          # per-step noise in speed [m/s]
    smooth_decay: float = 0.95         # AR(1) coefficient


# ---------------------------------------------------------------------------
# PID Baseline
# ---------------------------------------------------------------------------

@dataclass
class PIDConfig:
    """Parameters for the fixed-gap PID baseline controller."""
    h_pid: float = 2.0                 # desired time headway [s]
    standstill_gap: float = 4.0        # desired standstill spacing [m]
    kp: float = 0.5                    # proportional gain
    ki: float = 0.02                   # integral gain
    kd: float = 0.3                    # derivative gain
    windup_limit: float = 5.0          # integral anti-windup clip
    a_min: float = -3.0                # output clip [m/s²]
    a_max: float = 2.0


# ---------------------------------------------------------------------------
# Hybrid supervisor controller
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    """
    Parameters for HybridContextPIDController (v3 — sharp fallback design).

    Supervisor principle (hierarchical fallback):
        safe      -> pure context_ppo  (PID weight = 0.0 exactly)
        caution   -> near-pure context_ppo, tiny PID nudge (hybrid_pid_weight_caution)
        high_risk -> strong PID dominance (hybrid_pid_weight_high → 1.0), no weak blending
        emergency -> full PID takeover (weight = 1.0 exactly)

    Risk formula (max-based, NOT weighted average):
        risk_raw = max(r_ttc, r_gap, hybrid_headway_weight * r_headway)
    This ensures a single critical signal (TTC or gap) immediately drives takeover.

    Escalation: fast — triggered by risk_raw spike.
    De-escalation: slow — requires BOTH risk_raw AND risk_ema below exit threshold.

    Direct safe → high_risk transition (skipping caution) when risk_raw spikes
    past hybrid_high_enter in a single step.
    """
    # ------------------------------------------------------------------
    # Risk smoothing
    # ------------------------------------------------------------------
    # Higher alpha = EMA tracks more slowly = slower de-escalation
    hybrid_ema_alpha: float = 0.85

    # ------------------------------------------------------------------
    # Mode ENTRY thresholds (0..1, raw risk)
    # Escalation is driven by risk_raw for speed.
    # ------------------------------------------------------------------
    hybrid_caution_enter: float = 0.34      # safe -> caution (kept high to keep caution minimal)
    hybrid_high_enter: float = 0.18         # -> high_risk (from safe OR caution)
    hybrid_emergency_enter: float = 0.72    # -> emergency

    # ------------------------------------------------------------------
    # Mode EXIT thresholds (0..1)
    # De-escalation requires BOTH risk_raw < exit AND risk_ema < exit.
    # ------------------------------------------------------------------
    hybrid_caution_exit: float = 0.22       # caution -> safe
    hybrid_high_exit: float = 0.12          # high_risk -> caution
    hybrid_emergency_exit: float = 0.56     # emergency -> high_risk
    hybrid_mode_hysteresis_margin: float = 0.06

    # ------------------------------------------------------------------
    # Emergency hard-override triggers (physical, bypass risk score)
    # ------------------------------------------------------------------
    hybrid_emergency_ttc: float = 1.25      # TTC [s]  hard-override threshold
    hybrid_emergency_gap: float = 5.4       # gap [m]  hard-override threshold
    hybrid_emergency_gap_shortfall: float = 0.22   # severe shortfall ratio trigger
    hybrid_emergency_closing_speed: float = 1.2    # [m/s] trigger for shortfall emergency
    # Stricter direct safe->emergency gate to avoid reactive emergency overuse.
    hybrid_safe_direct_emergency_ttc: float = 0.95
    hybrid_safe_direct_emergency_gap: float = 4.8

    # ------------------------------------------------------------------
    # PID blend weights per mode
    # ------------------------------------------------------------------
    hybrid_pid_weight_caution: float = 0.00   # no contamination in caution by default
    hybrid_pid_weight_high: float = 0.95      # fixed strong PID dominance in high_risk
    hybrid_pid_dominant_threshold: float = 0.90  # label threshold for diagnostics

    # ------------------------------------------------------------------
    # Risk formula parameters
    # ------------------------------------------------------------------
    # TTC/gap dominate escalation; headway is secondary only.
    hybrid_ttc_weight: float = 1.60
    hybrid_gap_weight: float = 1.40
    hybrid_req_brake_weight: float = 1.80
    hybrid_headway_weight: float = 0.15
    hybrid_req_brake_critical: float = 2.2   # [m/s^2] required braking level treated as severe
    hybrid_high_enter_req_brake: float = 0.45

    # Reference values used in risk component computation
    pid_min_gap: float = 4.0               # standstill gap reference [m]
    hybrid_risk_headway_ref: float = 2.2   # time-headway reference for gap-shortfall risk [s]
                                           # (intentionally tighter than PID h_pid=2.0)
    ttc_critical: float = 5.0             # TTC at which r_ttc starts rising (→ 1 at TTC=0) [s]
    headway_critical: float = 1.3         # headway at which r_headway = 1 [s]

    # Safety filter / hold logic
    hybrid_filter_enable_high: bool = True
    hybrid_filter_enable_emergency: bool = True
    hybrid_high_hold_steps: int = 8
    hybrid_emergency_hold_steps: int = 12


# ---------------------------------------------------------------------------
# PPO Training
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Stable-Baselines3 PPO hyperparameters and training settings.

    ``checkpoint_eval_episodes`` controls each training-time evaluation used
    to select ``best_model.zip``; it is independent of final evaluation size.
    """
    total_timesteps: int = 500_000      # total environment steps per run
    n_envs: int = 8                     # parallel training environments
    checkpoint_eval_episodes: int = 5   # episodes per evaluation during training
    n_steps: int = 2048                 # rollout steps per env before update
    batch_size: int = 256
    n_epochs: int = 10
    # Modest stability retune (kept close to prior values).
    learning_rate: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    ent_coef: float = 0.002
    vf_coef: float = 0.6
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [128, 128])
    activation_fn: str = "tanh"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    normalize_obs: bool = True
    normalize_rew: bool = True
    verbose: int = 0                   # SB3 verbosity


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Final evaluation settings, separate from checkpoint selection."""
    # eval_seeds must match the model training seeds so load_trained_model
    # can locate the saved checkpoint for each seed.
    eval_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    scenarios: List[str] = field(default_factory=lambda: [
        "calm",
        "stressed",
        "fatigued",
        "switching",
    ])
    episodes_per_scenario: int = 50    # final evaluation episodes per scenario
    showcase_seeds: List[int] = field(default_factory=lambda: [700, 701, 702, 703, 704])
    showcase_episodes_per_case: int = 4


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Aggregated configuration used by all modules."""
    paths: PathConfig = field(default_factory=PathConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    human: HumanConfig = field(default_factory=HumanConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    lead: LeadConfig = field(default_factory=LeadConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        self.paths.make_dirs()


# Singleton default config used as a convenient import
default_config: Config = Config()
