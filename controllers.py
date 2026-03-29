"""
controllers.py
==============
Deterministic baseline controllers for the car-following task.

Implemented controllers
-----------------------
1. PIDController
   Fixed-gap PID with anti-windup.  Competes against PPO agents.

2. ConstantSpeedController
   Echo of lead speed — useful as a sanity-check / lower bound.

Both controllers share the same ``act(obs, info)`` interface so they can
be dropped into the evaluation loop without modification.

        # PID design
        # ----------
        # Desired gap:        d_des = h_pid * v_ego + d_standstill
        # Distance error:     e_d   = gap - d_des
        # Goal: reduce e_d to zero.
        #
        # Control law (PID on distance error):
        #   u = Kp * e_d + Ki * ∫e_d dt + Kd * de_d/dt
        #
        # This yields intuitive behavior:
        #   - gap > desired  -> e_d > 0 -> accelerate
        #   - gap < desired  -> e_d < 0 -> brake
        #
        # Output is clipped to [a_min, a_max] with anti-windup integration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import HybridConfig, PIDConfig, SimConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseController:
    """Protocol-compatible base class for all controllers."""

    def reset(self) -> None:
        """Reset any internal state at the start of a new episode."""

    def act(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Return a single-element action array given the current observation.

        Parameters
        ----------
        obs:
            Observation vector from the environment.
        info:
            Optional info dict from the environment (contains raw values).

        Returns
        -------
        action:
            numpy array of shape (1,) with the acceleration command [m/s²].
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------

class PIDController(BaseController):
    """
    Fixed-gap PID controller for longitudinal car-following.

    The controller maintains a desired time-gap (time headway) relative to the
    lead vehicle using proportional, integral, and derivative terms.

    Obs layout assumed (indices):
        0: gap      (d_t)
        1: v_ego    (v_ego_t)
        2: v_lead   (v_lead_t)
        3: accel    (a_t)
        4: jerk     (j_t)
    Any additional context dimensions (indices 5–9) are ignored.

    Parameters
    ----------
    pid_cfg:
        PIDConfig from config.py.
    sim_cfg:
        SimConfig for dt and acceleration bounds.
    """

    def __init__(self, pid_cfg: PIDConfig, sim_cfg: SimConfig) -> None:
        self.cfg = pid_cfg
        self.sim = sim_cfg
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def act(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Compute PID acceleration command from current observation."""
        gap = float(obs[0])
        v_ego = float(obs[1])
        dt = self.sim.dt

        # Desired spacing policy: time-gap + standstill offset.
        d_desired = self.cfg.h_pid * max(v_ego, 0.0) + self.cfg.standstill_gap

        # Positive error means too far (can accelerate), negative means too close (brake).
        error = gap - d_desired

        if not self._initialized:
            self._prev_error = error
            self._initialized = True

        derivative = (error - self._prev_error) / max(dt, 1e-6)

        # Candidate integral and unclipped control for anti-windup gating.
        integral_candidate = self._integral + error * dt
        integral_candidate = float(
            np.clip(integral_candidate, -self.cfg.windup_limit, self.cfg.windup_limit)
        )

        u_unclipped = (
            self.cfg.kp * error
            + self.cfg.ki * integral_candidate
            + self.cfg.kd * derivative
        )
        u = float(np.clip(u_unclipped, self.cfg.a_min, self.cfg.a_max))

        # Anti-windup: only commit integration when either unsaturated,
        # or when the error would reduce saturation.
        saturating_high = u_unclipped > self.cfg.a_max
        saturating_low = u_unclipped < self.cfg.a_min
        reduce_high = error < 0.0
        reduce_low = error > 0.0
        if (not saturating_high and not saturating_low) or (saturating_high and reduce_high) or (saturating_low and reduce_low):
            self._integral = integral_candidate

        self._prev_error = error
        return np.array([u], dtype=np.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def integral(self) -> float:
        return self._integral


# ---------------------------------------------------------------------------
# Constant-speed controller (sanity check)
# ---------------------------------------------------------------------------

class ConstantSpeedController(BaseController):
    """
    Trivial controller: always commands zero acceleration.

    Useful as a lower bound / sanity check.
    """

    def reset(self) -> None:
        pass

    def act(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hybrid context-PPO + PID safety supervisor
# ---------------------------------------------------------------------------

class HybridContextPIDController:
    """
        Hybrid supervisor around context_ppo (nominal) and PID (safety).

        Risk is computed from raw physical values in ``info`` only:
        gap, v_ego, v_lead, headway, ttc_proxy (or reconstructed TTC).

        Supervisor modes (with hysteresis):
            safe -> caution -> high_risk -> emergency

        - ``risk_raw`` drives fast escalation and emergency takeover.
        - ``risk_ema`` smooths mild/moderate blending in caution mode.
        - Emergency mode forces full PID takeover (no timid blending).
    """

    def __init__(
        self,
        ppo_model: Any,
        pid_controller: PIDController,
        obs_stats: Any,           # VecNormalize obs_rms (or None)
        cfg: HybridConfig,
    ) -> None:
        self.model = ppo_model
        self.pid = pid_controller
        self.obs_stats = obs_stats    # RunningMeanStd from VecNormalize pickle
        self.cfg = cfg

        self._risk_raw: float = 0.0
        self._risk_ema: float = 0.0
        self._pid_weight: float = 0.0
        self._mode: str = "safe"
        self._emergency_takeover: bool = False
        self._components: Dict[str, float] = {
            "r_ttc": 0.0,
            "r_gap": 0.0,
            "r_req_brake": 0.0,
            "r_headway": 0.0,
            "r_closing": 0.0,
        }
        self._high_hold_steps_remaining: int = 0
        self._emergency_hold_steps_remaining: int = 0
        self._filter_active: bool = False
        self._filter_clamped: bool = False
        self._action_delta_context: float = 0.0

        # Per-step transition flags (reset at the start of each compute_risk call)
        self._transition_this_step: bool = False
        self._escal_sc: bool = False   # safe → caution
        self._escal_sh: bool = False   # safe → high_risk  (skipping caution)
        self._escal_se: bool = False   # safe → emergency
        self._escal_ch: bool = False   # caution → high_risk
        self._escal_he: bool = False   # * → emergency  (any escalation into emergency)
        self._deescal: bool = False    # any transition to a lower mode

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state at the start of a new episode."""
        self._risk_raw = 0.0
        self._risk_ema = 0.0
        self._pid_weight = 0.0
        self._mode = "safe"
        self._emergency_takeover = False
        self._components = {
            "r_ttc": 0.0,
            "r_gap": 0.0,
            "r_req_brake": 0.0,
            "r_headway": 0.0,
            "r_closing": 0.0,
        }
        self._high_hold_steps_remaining = 0
        self._emergency_hold_steps_remaining = 0
        self._filter_active = False
        self._filter_clamped = False
        self._action_delta_context = 0.0
        self._transition_this_step = False
        self._escal_sc = False
        self._escal_sh = False
        self._escal_se = False
        self._escal_ch = False
        self._escal_he = False
        self._deescal = False
        self.pid.reset()

    def _extract_physical_state(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]],
    ) -> Tuple[float, float, float, float, float, float]:
        """Extract raw physical state from info with obs fallback."""
        eps = 1e-6
        src = info or {}
        gap = float(src.get("gap", float(obs[0])))
        v_ego = float(src.get("v_ego", float(obs[1])))
        v_lead = float(src.get("v_lead", float(obs[2])))
        dv = v_ego - v_lead

        headway = float(src.get("headway", gap / max(v_ego, eps)))
        ttc_info = src.get("ttc_proxy", None)
        if ttc_info is not None:
            ttc = float(ttc_info)
        elif dv > 0.0:
            ttc = gap / max(dv, eps)
        else:
            ttc = 1e6

        return gap, v_ego, v_lead, dv, headway, ttc

    def _compute_raw_risk(
        self,
        gap: float,
        v_ego: float,
        dv: float,
        headway: float,
        ttc: float,
    ) -> Tuple[float, Dict[str, float], bool, bool]:
        """
        Compute raw risk using a max-based formula so the most critical
        single signal (TTC or gap) immediately drives escalation.

        risk_raw = max(r_ttc, r_gap, hybrid_headway_weight * r_headway)

        TTC and gap-shortfall are primary.  Headway is secondary (dampened).
        """
        eps = 1e-6

        # TTC risk: only when ego is closing on lead vehicle
        if dv > 0.0:
            r_ttc = float(np.clip(1.0 - ttc / max(self.cfg.ttc_critical, eps), 0.0, 1.0))
        else:
            r_ttc = 0.0

        # Gap-shortfall risk: use a safety-oriented reference gap (tighter than PID desired)
        # risk_gap_ref = pid_min_gap + hybrid_risk_headway_ref * v_ego
        # At typical driving (gap~20m, v=12m/s): risk_gap_ref~22m, so gap shortfall~2m → r_gap~0.09
        risk_gap_ref = self.cfg.pid_min_gap + self.cfg.hybrid_risk_headway_ref * max(v_ego, 0.0)
        gap_shortfall = max(0.0, risk_gap_ref - gap)
        r_gap = float(np.clip(gap_shortfall / max(risk_gap_ref, eps), 0.0, 1.0))

        # Required-brake risk: approximate braking needed to avoid violating a
        # safe terminal gap. This surfaces physically unrecoverable states earlier.
        available_distance = max(gap - risk_gap_ref, eps)
        if dv > 0.0:
            a_req = (dv * dv) / max(2.0 * available_distance, eps)
            r_req_brake = float(
                np.clip(a_req / max(self.cfg.hybrid_req_brake_critical, eps), 0.0, 1.0)
            )
        else:
            r_req_brake = 0.0

        # Headway risk: secondary signal, dampened by hybrid_headway_weight
        r_headway = float(np.clip(1.0 - headway / max(self.cfg.headway_critical, eps), 0.0, 1.0))

        # Closing speed risk (for diagnostics only, not used in risk_raw)
        r_closing = float(np.clip(dv / max(5.0, eps), 0.0, 1.0)) if dv > 0.0 else 0.0

        # Weighted-max combination: TTC and gap dominate; headway is secondary.
        raw_risk = max(
            self.cfg.hybrid_ttc_weight * r_ttc,
            self.cfg.hybrid_gap_weight * r_gap,
            self.cfg.hybrid_req_brake_weight * r_req_brake,
            self.cfg.hybrid_headway_weight * r_headway,
        )
        raw_risk = float(np.clip(raw_risk, 0.0, 1.0))

        # Hard-override emergency trigger (bypasses risk score)
        severe_shortfall = (gap_shortfall / max(risk_gap_ref, eps)) >= self.cfg.hybrid_emergency_gap_shortfall
        fast_closing = dv >= self.cfg.hybrid_emergency_closing_speed
        emergency_trigger = (
            (dv > 0.0 and ttc <= self.cfg.hybrid_emergency_ttc)
            or (gap <= self.cfg.hybrid_emergency_gap)
            or (severe_shortfall and fast_closing)
        )

        # Direct safe->emergency should require stricter severity than general emergency.
        direct_safe_emergency = (
            (dv > 0.0 and ttc <= self.cfg.hybrid_safe_direct_emergency_ttc)
            or (gap <= self.cfg.hybrid_safe_direct_emergency_gap)
        )

        comps = {
            "r_ttc": r_ttc,
            "r_gap": r_gap,
            "r_req_brake": r_req_brake,
            "r_headway": r_headway,
            "r_closing": r_closing,
        }
        return raw_risk, comps, emergency_trigger, direct_safe_emergency

    def _enter_mode(self, from_mode: str, to_mode: str) -> None:
        """Switch modes, record transition flags, and arm hold counters."""
        if from_mode != to_mode:
            self._track_transition(from_mode, to_mode)
        self._mode = to_mode
        if to_mode == "high_risk":
            self._high_hold_steps_remaining = max(0, int(self.cfg.hybrid_high_hold_steps))
        elif to_mode == "emergency":
            self._emergency_hold_steps_remaining = max(0, int(self.cfg.hybrid_emergency_hold_steps))

    def _track_transition(self, from_mode: str, to_mode: str) -> None:
        """Record a mode transition and update escalation / de-escalation flags."""
        _order = {"safe": 0, "caution": 1, "high_risk": 2, "emergency": 3}
        self._transition_this_step = True
        if _order.get(to_mode, 0) > _order.get(from_mode, 0):
            # Escalation
            if from_mode == "safe" and to_mode == "caution":
                self._escal_sc = True
            elif from_mode == "safe" and to_mode == "high_risk":
                self._escal_sh = True
            elif from_mode == "safe" and to_mode == "emergency":
                self._escal_se = True
            elif from_mode == "caution" and to_mode == "high_risk":
                self._escal_ch = True
            elif to_mode == "emergency":
                self._escal_he = True
        else:
            self._deescal = True

    def _update_mode(
        self,
        risk_raw: float,
        emergency_trigger: bool,
        direct_safe_emergency: bool,
    ) -> None:
        """
        Update supervisor mode with sharp escalation and hysteretic de-escalation.

        Escalation is driven by risk_raw (fast).
        De-escalation requires BOTH risk_raw AND risk_ema below the exit threshold
        (slow — EMA must recover before leaving a dangerous mode).

        Key design: direct safe → high_risk transition is allowed (skipping caution)
        when risk_raw spikes past hybrid_high_enter in a single step.
        """
        prev_mode = self._mode
        margin = max(0.0, self.cfg.hybrid_mode_hysteresis_margin)
        caution_exit_eff = min(self.cfg.hybrid_caution_exit, self.cfg.hybrid_caution_enter - margin)
        high_exit_eff = min(self.cfg.hybrid_high_exit, self.cfg.hybrid_high_enter - margin)
        emergency_exit_eff = min(self.cfg.hybrid_emergency_exit, self.cfg.hybrid_emergency_enter - margin)
        req_brake_risk = self._components["r_req_brake"]
        ttc_risk = self._components["r_ttc"]
        high_risk_trigger = (
            risk_raw >= self.cfg.hybrid_high_enter
            or req_brake_risk >= self.cfg.hybrid_high_enter_req_brake
            or ttc_risk >= self.cfg.hybrid_high_enter
        )

        if self._mode == "high_risk" and self._high_hold_steps_remaining > 0:
            self._high_hold_steps_remaining -= 1
        if self._mode == "emergency" and self._emergency_hold_steps_remaining > 0:
            self._emergency_hold_steps_remaining -= 1

        # --- Emergency (highest priority, instantaneous) ---
        # From safe, use stricter direct-emergency gate to reduce reactive jumps.
        if self._mode == "safe":
            if direct_safe_emergency:
                self._emergency_takeover = True
                self._enter_mode(prev_mode, "emergency")
                return
        elif emergency_trigger or risk_raw >= self.cfg.hybrid_emergency_enter:
            self._emergency_takeover = True
            if self._mode != "emergency":
                self._enter_mode(prev_mode, "emergency")
            return

        self._emergency_takeover = False

        # --- De-escalate from emergency ---
        if self._mode == "emergency":
            if (self._emergency_hold_steps_remaining <= 0
                    and risk_raw < emergency_exit_eff
                    and self._risk_ema < emergency_exit_eff):
                self._enter_mode("emergency", "high_risk")
            return

        # --- De-escalate from high_risk, or re-escalate ---
        if self._mode == "high_risk":
            if risk_raw >= self.cfg.hybrid_emergency_enter or emergency_trigger:
                # Should have been caught above, but guard against float edge cases.
                self._emergency_takeover = True
                self._enter_mode("high_risk", "emergency")
                return
            if (self._high_hold_steps_remaining <= 0
                    and risk_raw < high_exit_eff
                    and self._risk_ema < high_exit_eff):
                self._enter_mode("high_risk", "caution")
            return

        # --- caution mode ---
        if self._mode == "caution":
            if high_risk_trigger:
                self._enter_mode("caution", "high_risk")
                return
            if (risk_raw < caution_exit_eff
                    and self._risk_ema < caution_exit_eff):
                self._enter_mode("caution", "safe")
            return

        # --- safe mode ---
        if high_risk_trigger:
            # Danger spike: skip caution, go directly to high_risk
            self._enter_mode("safe", "high_risk")
        elif risk_raw >= self.cfg.hybrid_caution_enter:
            self._enter_mode("safe", "caution")

    def _compute_pid_weight(self, risk_raw: float) -> float:
        """
        Compute PID blend weight per mode.

        - safe:      0.0  — pure context_ppo, no PID interference
        - caution:   hybrid_pid_weight_caution (0.00 by default) — no contamination
        - high_risk: fixed hybrid_pid_weight_high (0.95 by default) for strong dominance
        - emergency: 1.0  — full PID takeover
        """
        if self._mode == "safe":
            return 0.0
        if self._mode == "emergency":
            return 1.0
        if self._mode == "caution":
            return float(self.cfg.hybrid_pid_weight_caution)
        # high_risk: fixed strong dominance to reduce variance and avoid weak intervention.
        return float(np.clip(self.cfg.hybrid_pid_weight_high, 0.0, 1.0))

    def _ppo_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute deterministic context_ppo action with stored obs normalisation."""
        obs_input = obs.copy().astype(np.float32)
        if self.obs_stats is not None:
            obs_input = (obs_input - self.obs_stats.mean) / (
                np.sqrt(self.obs_stats.var + 1e-8)
            )
            obs_input = np.clip(obs_input, -10.0, 10.0)
        action, _ = self.model.predict(obs_input, deterministic=True)
        return action.astype(np.float32)

    def compute_risk(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Update and return (risk_raw, risk_ema) from raw physical state."""
        # Reset per-step transition flags before each new step
        self._transition_this_step = False
        self._escal_sc = False
        self._escal_sh = False
        self._escal_se = False
        self._escal_ch = False
        self._escal_he = False
        self._deescal = False
        self._filter_active = False
        self._filter_clamped = False
        self._action_delta_context = 0.0

        gap, v_ego, _v_lead, dv, headway, ttc = self._extract_physical_state(obs, info)
        raw_risk, comps, emergency_trigger, direct_safe_emergency = self._compute_raw_risk(
            gap=gap,
            v_ego=v_ego,
            dv=dv,
            headway=headway,
            ttc=ttc,
        )

        alpha = float(np.clip(self.cfg.hybrid_ema_alpha, 0.0, 0.99))
        # Slight scale-up so high-risk samples land in a more interpretable range.
        raw_risk_scaled = float(np.clip(1.35 * raw_risk, 0.0, 1.0))
        self._risk_raw = raw_risk_scaled
        self._risk_ema = alpha * self._risk_ema + (1.0 - alpha) * raw_risk_scaled
        self._components = comps

        # risk_raw drives mode escalation; EMA stabilises mild-region behavior.
        self._update_mode(
            risk_raw=raw_risk_scaled,
            emergency_trigger=emergency_trigger,
            direct_safe_emergency=direct_safe_emergency,
        )
        self._pid_weight = self._compute_pid_weight(risk_raw=raw_risk_scaled)
        return self._risk_raw, self._risk_ema

    def act(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Select an action from context_ppo, protected by a hard safety filter in
        high_risk / emergency modes when physical danger is present.
        """
        self.compute_risk(obs, info)
        w = self._pid_weight
        ppo_action = self._ppo_action(obs)

        # Keep low-risk behavior almost identical to context_ppo.
        if w <= 1e-6:
            return ppo_action

        pid_action = self.pid.act(obs, info)
        action = (1.0 - w) * ppo_action + w * pid_action

        if self._mode == "high_risk" and self.cfg.hybrid_filter_enable_high:
            # Safety filter: final action cannot be less braking than PID.
            filtered = np.minimum(action, pid_action)
            self._filter_active = True
            self._filter_clamped = bool(np.any(filtered < action - 1e-6))
            action = filtered
        elif self._mode == "emergency" and self.cfg.hybrid_filter_enable_emergency:
            # Full safety override is clearer and safer in emergency.
            self._filter_active = True
            self._filter_clamped = True
            action = pid_action

        action = np.clip(action, self.pid.cfg.a_min, self.pid.cfg.a_max).astype(np.float32)
        self._action_delta_context = float(np.mean(np.abs(action - ppo_action)))
        return action

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def risk_ema(self) -> float:
        """Current EMA-smoothed risk value."""
        return self._risk_ema

    @property
    def risk_raw(self) -> float:
        """Current raw (unsmoothed) physical risk value."""
        return self._risk_raw

    @property
    def pid_weight(self) -> float:
        """Current PID blend weight in [0, 1]."""
        return self._pid_weight

    @property
    def mode(self) -> str:
        """Current supervisor mode: safe/caution/high_risk/emergency."""
        return self._mode

    @property
    def in_pid_mode(self) -> bool:
        """Backward-compatible boolean indicating any PID intervention."""
        return self._pid_weight > 1e-6

    def diagnostics(self) -> Dict[str, float]:
        """
        Return per-step hybrid diagnostics for evaluation logging.

        Mode flags (0/1) and risk values reflect end-of-step state.
        Transition flags (0/1) are set only on the step a transition occurs.
        """
        pid_active = 1.0 if self._pid_weight > 1e-6 else 0.0
        pid_dominant = 1.0 if self._pid_weight >= self.cfg.hybrid_pid_dominant_threshold else 0.0
        mode_safe = 1.0 if self._mode == "safe" else 0.0
        mode_caution = 1.0 if self._mode == "caution" else 0.0
        mode_high = 1.0 if self._mode == "high_risk" else 0.0
        mode_emergency = 1.0 if self._mode == "emergency" else 0.0
        return {
            # Risk scores
            "hybrid_risk_raw": self._risk_raw,
            "hybrid_risk_ema": self._risk_ema,
            # Action blending
            "hybrid_pid_weight": self._pid_weight,
            "hybrid_pid_active": pid_active,
            "hybrid_pid_dominant": pid_dominant,
            "hybrid_emergency_takeover": 1.0 if self._emergency_takeover else 0.0,
            # Mode flags
            "hybrid_mode_safe": mode_safe,
            "hybrid_mode_caution": mode_caution,
            "hybrid_mode_high_risk": mode_high,
            "hybrid_mode_emergency": mode_emergency,
            # Risk components (for interpretation)
            "hybrid_risk_ttc": self._components["r_ttc"],
            "hybrid_risk_gap": self._components["r_gap"],
            "hybrid_risk_req_brake": self._components["r_req_brake"],
            "hybrid_risk_headway": self._components["r_headway"],
            "hybrid_risk_closing": self._components["r_closing"],
            "hybrid_filter_active": 1.0 if self._filter_active else 0.0,
            "hybrid_filter_clamped": 1.0 if self._filter_clamped else 0.0,
            "hybrid_action_delta_context": self._action_delta_context,
            # Mode transition flags (1 only on the step a transition occurs)
            "hybrid_transition": 1.0 if self._transition_this_step else 0.0,
            "hybrid_escal_sc": 1.0 if self._escal_sc else 0.0,    # safe→caution
            "hybrid_escal_sh": 1.0 if self._escal_sh else 0.0,    # safe→high_risk
            "hybrid_escal_se": 1.0 if self._escal_se else 0.0,    # safe→emergency
            "hybrid_escal_ch": 1.0 if self._escal_ch else 0.0,    # caution→high_risk
            "hybrid_escal_he": 1.0 if self._escal_he else 0.0,    # *→emergency
            "hybrid_deescal": 1.0 if self._deescal else 0.0,      # any de-escalation
        }


# ---------------------------------------------------------------------------
# PID rollout helper
# ---------------------------------------------------------------------------

def run_pid_episode(
    env: Any,
    controller: PIDController,
    seed: Optional[int] = None,
) -> Tuple[float, list]:
    """
    Run a single episode with the PID controller.

    Parameters
    ----------
    env:
        A CarFollowingEnv instance (or compatible).
    controller:
        PIDController instance.
    seed:
        Optional seed for the episode reset.

    Returns
    -------
    episode_return:
        Cumulative reward over the episode.
    trajectory:
        List of info dicts, one per timestep.
    """
    reset_kwargs: Dict[str, Any] = {}
    if seed is not None:
        reset_kwargs["seed"] = seed

    obs, info = env.reset(**reset_kwargs)
    controller.reset()
    trajectory = [info]
    episode_return = 0.0
    done = False

    while not done:
        action = controller.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        info["reward"] = float(reward)
        trajectory.append(info)
        done = terminated or truncated

    return episode_return, trajectory
