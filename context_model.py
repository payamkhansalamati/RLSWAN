"""
context_model.py
================
Synthetic Human-State Simulator + Structured "LLM-like" Context Interpreter.

Architecture
------------
1. LatentHumanStateSimulator
   Maintains a hidden latent regime (calm / stressed / fatigued / both) that
   switches stochastically over time.  Each regime produces plausible ranges
   for physiological signals:
     heart_rate, HRV, respiration_rate, fatigue_index

2. PhysiologicalSignalGenerator
   Samples noisy physiological values from the current latent regime.

3. ContextInterpreter
   Deterministic rule-based logic (mimicking what an LLM would produce) that
   maps raw physiological signals → structured context dict:
     {stress_level, fatigue_level, preferred_headway,
      comfort_priority, efficiency_priority}

4. ContextModel  (façade)
   Combines the three components into a single step() call.
   Returns a numpy array c_t = [stress, fatigue, h_pref, comfort, efficiency].

The hidden human state is NOT visible to the RL agent; it only sees the
context vector output by the interpreter (with optional additive noise).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from config import HumanConfig


# ---------------------------------------------------------------------------
# Latent regime definitions
# ---------------------------------------------------------------------------

REGIMES = ("calm", "stressed", "fatigued", "both")

# Regime transition probability matrix (rows = from, cols = to)
# Designed so that regimes persist for many steps (inertia ≈ 1 - 1/hold_steps)
_BASE_TRANSITION = np.array([
    [0.97, 0.01, 0.01, 0.01],  # calm
    [0.02, 0.95, 0.01, 0.02],  # stressed
    [0.02, 0.01, 0.95, 0.02],  # fatigued
    [0.01, 0.02, 0.02, 0.95],  # both
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhysioReading:
    """Raw physiological measurement at one timestep."""
    heart_rate: float        # beats per minute
    hrv: float               # RMSSD proxy [ms]
    respiration_rate: float  # breaths per minute
    fatigue_index: float     # dimensionless [0, 1]


@dataclass
class ContextOutput:
    """Output of the structured context interpreter."""
    stress_level: float          # κ_stress  ∈ [0, 1]
    fatigue_level: float         # κ_fatigue ∈ [0, 1]
    preferred_headway: float     # h_pref    > 0  [s]
    comfort_priority: float      # κ_comfort ∈ [0, 1]
    efficiency_priority: float   # κ_eff     ∈ [0, 1]
    regime_label: str = "calm"   # human-readable latent regime label

    def to_array(self) -> np.ndarray:
        """Return a 5-element float32 array for use as state component."""
        return np.array([
            self.stress_level,
            self.fatigue_level,
            self.preferred_headway,
            self.comfort_priority,
            self.efficiency_priority,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {
            "stress_level":       self.stress_level,
            "fatigue_level":      self.fatigue_level,
            "preferred_headway":  self.preferred_headway,
            "comfort_priority":   self.comfort_priority,
            "efficiency_priority": self.efficiency_priority,
            "regime_label":       self.regime_label,
        }


# ---------------------------------------------------------------------------
# Latent human state simulator
# ---------------------------------------------------------------------------

class LatentHumanStateSimulator:
    """
    Markov-chain regime switcher with per-regime physiological signal ranges.

    The latent regime index is hidden from the RL agent.  It determines
    which physiological ranges are active at each timestep.
    """

    def __init__(self, cfg: HumanConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self._regime_idx: int = 0          # start in "calm"
        self._steps_in_regime: int = 0
        self._transition: np.ndarray = _BASE_TRANSITION.copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def regime(self) -> str:
        return REGIMES[self._regime_idx]

    def reset(self, initial_regime: Optional[str] = None) -> None:
        if initial_regime is not None and initial_regime in REGIMES:
            self._regime_idx = REGIMES.index(initial_regime)
        else:
            self._regime_idx = 0
        self._steps_in_regime = 0

    def step(self) -> str:
        """Advance latent state one timestep; may switch regime."""
        self._steps_in_regime += 1
        switch_prob = 1.0 / max(1, self.cfg.regime_hold_steps)
        if self.rng.random() < switch_prob:
            row = self._transition[self._regime_idx]
            self._regime_idx = int(self.rng.choice(len(REGIMES), p=row))
            self._steps_in_regime = 0
        return self.regime

    def get_physio_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the physiological value ranges for the current regime."""
        r = self.regime
        cfg = self.cfg
        suffix = "_both" if r == "both" else f"_{r}"
        return {
            "heart_rate":       getattr(cfg, f"hr{suffix}"),
            "hrv":              getattr(cfg, f"hrv{suffix}"),
            "respiration_rate": getattr(cfg, f"resp{suffix}"),
            "fatigue_index":    getattr(cfg, f"fatigue{suffix}"),
        }

    def force_regime(self, regime: str) -> None:
        """Force the latent regime deterministically (used in showcase schedules)."""
        if regime not in REGIMES:
            raise ValueError(f"Unknown regime: {regime!r}")
        self._regime_idx = REGIMES.index(regime)
        self._steps_in_regime = 0


# ---------------------------------------------------------------------------
# Physiological signal generator
# ---------------------------------------------------------------------------

class PhysiologicalSignalGenerator:
    """
    Draws noisy physiological readings from the current regime's ranges.

    Adds temporal smoothing via a first-order low-pass filter so that
    readings change gradually, which is more physiologically realistic.
    """

    def __init__(self, cfg: HumanConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self._smoothed: Optional[PhysioReading] = None
        self._alpha = 0.85  # low-pass coefficient

    def reset(self) -> None:
        self._smoothed = None

    def sample(self, ranges: Dict[str, Tuple[float, float]]) -> PhysioReading:
        """
        Return a smoothed physiological reading inside the provided ranges.
        If reset() was called, initialise from the range centre.
        """
        raw = PhysioReading(
            heart_rate=self._sample_range(ranges["heart_rate"]),
            hrv=self._sample_range(ranges["hrv"]),
            respiration_rate=self._sample_range(ranges["respiration_rate"]),
            fatigue_index=self._sample_range(ranges["fatigue_index"]),
        )

        if self._smoothed is None:
            self._smoothed = raw
            return raw

        # Exponential moving average smoothing
        alpha = self._alpha
        self._smoothed = PhysioReading(
            heart_rate=alpha * self._smoothed.heart_rate + (1 - alpha) * raw.heart_rate,
            hrv=alpha * self._smoothed.hrv + (1 - alpha) * raw.hrv,
            respiration_rate=alpha * self._smoothed.respiration_rate + (1 - alpha) * raw.respiration_rate,
            fatigue_index=alpha * self._smoothed.fatigue_index + (1 - alpha) * raw.fatigue_index,
        )
        return self._smoothed

    def _sample_range(self, bounds: Tuple[float, float]) -> float:
        lo, hi = bounds
        return float(self.rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# Structured "LLM-like" context interpreter
# ---------------------------------------------------------------------------

class ContextInterpreter:
    """
    Rule-based interpreter that maps physiological readings → ContextOutput.

    Rules are designed to be physiologically realistic:
      - High HR + low HRV  → high stress
      - High fatigue index → high fatigue
      - High stress/fatigue → larger preferred headway
      - High stress/fatigue → high comfort priority
      - Low  stress/fatigue → high efficiency priority

    Small Gaussian noise is added to mimic imperfect inference.
    """

    # Physiological normalisation anchors
    HR_MIN, HR_MAX = 55.0, 110.0
    HRV_MIN, HRV_MAX = 10.0, 80.0
    RESP_MIN, RESP_MAX = 10.0, 28.0

    # Headway preferences
    H_MIN, H_MAX = 1.4, 3.0   # seconds

    def __init__(self, cfg: HumanConfig, noise_std: float = 0.05) -> None:
        self.cfg = cfg
        self.noise_std = noise_std

    def interpret(
        self,
        reading: PhysioReading,
        rng: np.random.Generator,
    ) -> ContextOutput:
        """
        Map a PhysioReading to a ContextOutput.

        Parameters
        ----------
        reading:
            Raw physiological reading.
        rng:
            NumPy random generator for noise injection.
        """
        # --- Normalise physiological signals to [0, 1] ---
        hr_norm = self._norm(reading.heart_rate, self.HR_MIN, self.HR_MAX)
        hrv_norm = self._norm(reading.hrv, self.HRV_MIN, self.HRV_MAX)
        resp_norm = self._norm(reading.respiration_rate, self.RESP_MIN, self.RESP_MAX)
        fatigue_norm = float(np.clip(reading.fatigue_index, 0.0, 1.0))

        # --- Stress: strong, monotonic HR/HRV relationship ---
        stress_raw = 0.5 * hr_norm + 0.5 * (1.0 - hrv_norm)
        stress_raw = float(np.clip(stress_raw, 0.0, 1.0))

        # --- Fatigue: fatigue index + respiration instability proxy ---
        fatigue_raw = float(np.clip(0.7 * fatigue_norm + 0.3 * resp_norm, 0.0, 1.0))

        # --- Preferred headway: monotonic with stress + fatigue ---
        h_pref = 1.4 + 0.8 * stress_raw + 0.8 * fatigue_raw
        h_pref = float(np.clip(h_pref, self.H_MIN, self.H_MAX))

        # --- Comfort priority: higher when driver is impaired ---
        comfort_prio = float(np.clip(0.3 + 0.5 * stress_raw + 0.4 * fatigue_raw, 0.0, 1.0))

        # --- Efficiency priority: inversely related to impairment ---
        eff_prio = float(np.clip(1.0 - 0.6 * stress_raw - 0.5 * fatigue_raw, 0.0, 1.0))

        # --- Inject noise ---
        if self.noise_std > 0.0:
            noise = rng.normal(0.0, self.noise_std, size=5)
            stress_raw = float(np.clip(stress_raw + noise[0], 0.0, 1.0))
            fatigue_raw = float(np.clip(fatigue_raw + noise[1], 0.0, 1.0))
            h_pref = float(np.clip(h_pref + noise[2] * 0.5, self.H_MIN, self.H_MAX))
            comfort_prio = float(np.clip(comfort_prio + noise[3], 0.0, 1.0))
            eff_prio = float(np.clip(eff_prio + noise[4], 0.0, 1.0))

        return ContextOutput(
            stress_level=stress_raw,
            fatigue_level=fatigue_raw,
            preferred_headway=h_pref,
            comfort_priority=comfort_prio,
            efficiency_priority=eff_prio,
        )

    @staticmethod
    def _norm(x: float, lo: float, hi: float) -> float:
        """Linear normalise x ∈ [lo, hi] → [0, 1]."""
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Façade: ContextModel
# ---------------------------------------------------------------------------

class ContextModel:
    """
    Façade that wires together the human-state simulator, signal generator,
    and context interpreter into a single ``step()`` API.

    Usage
    -----
    model = ContextModel(cfg, rng)
    model.reset(initial_regime="calm")
    for t in range(T):
        ctx: ContextOutput = model.step()
        c_t = ctx.to_array()   # → shape (5,)
    """

    def __init__(
        self,
        cfg: HumanConfig,
        rng: np.random.Generator,
        noise_std: Optional[float] = None,
        fixed_regime: Optional[str] = None,
        regime_schedule: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        cfg:
            HumanConfig from config.py.
        rng:
            NumPy random generator (created externally for seed control).
        noise_std:
            Override for context-noise standard deviation (default: cfg.noise_std).
        fixed_regime:
            If set, the latent regime does NOT switch; useful for evaluation
            sub-scenarios ("calm", "stressed", "fatigued", "both").
        """
        self.cfg = cfg
        self.rng = rng
        self.fixed_regime = fixed_regime
        self.regime_schedule = regime_schedule or {}
        self._episode_step = 0

        effective_noise = noise_std if noise_std is not None else cfg.noise_std
        self._latent = LatentHumanStateSimulator(cfg, rng)
        self._physio = PhysiologicalSignalGenerator(cfg, rng)
        self._interpreter = ContextInterpreter(cfg, noise_std=effective_noise)

        self._last_output: Optional[ContextOutput] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, initial_regime: Optional[str] = None) -> ContextOutput:
        """
        Reset the context model.  Call at the start of every episode.

        Parameters
        ----------
        initial_regime:
            Force the starting latent regime (None → always starts as "calm"
            unless ``fixed_regime`` was set at construction time).
        """
        start = initial_regime or self.fixed_regime or "calm"
        self._latent.reset(initial_regime=start)
        self._episode_step = 0
        self._apply_schedule_if_needed()
        self._physio.reset()
        self._last_output = self._compute_output()
        return self._last_output

    def step(self) -> ContextOutput:
        """Advance the hidden human state and return updated context."""
        self._episode_step += 1
        self._apply_schedule_if_needed()
        if self.fixed_regime is None and not self.regime_schedule:
            self._latent.step()
        self._last_output = self._compute_output()
        return self._last_output

    @property
    def current_regime(self) -> str:
        return self._latent.regime if self.fixed_regime is None else self.fixed_regime

    @property
    def last_output(self) -> Optional[ContextOutput]:
        return self._last_output

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_output(self) -> ContextOutput:
        ranges = self._latent.get_physio_ranges()
        reading = self._physio.sample(ranges)
        out = self._interpreter.interpret(reading, self.rng)
        out.regime_label = self.current_regime
        return out

    def _apply_schedule_if_needed(self) -> None:
        """Apply deterministic regime transitions when a schedule is configured."""
        if not self.regime_schedule:
            return
        if self._episode_step in self.regime_schedule:
            self._latent.force_regime(self.regime_schedule[self._episode_step])


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------

def make_context_model(
    cfg: HumanConfig,
    seed: int,
    scenario: str = "mixed",
) -> ContextModel:
    """
    Convenience factory for common evaluation scenarios.

    Parameters
    ----------
    scenario:
        "calm"     – fixed calm regime throughout
        "stressed" – fixed stressed regime throughout
        "fatigued" – fixed fatigued regime throughout
        "both"     – fixed stressed+fatigued regime throughout
        "escalation" – deterministic transitions calm→stressed→fatigued→both
        "switching" – latent regime changes freely (most challenging)
        "mixed"    – same as "switching" (default for training)
    """
    rng = np.random.default_rng(seed)
    if scenario in ("calm", "stressed", "fatigued", "both"):
        return ContextModel(cfg, rng, fixed_regime=scenario)
    if scenario == "escalation":
        # Deterministic schedule with quarter-episode transitions for 500-step default.
        schedule = {
            0: "calm",
            125: "stressed",
            250: "fatigued",
            375: "both",
        }
        return ContextModel(cfg, rng, fixed_regime=None, regime_schedule=schedule)
    # "switching" / "mixed" → no fixed regime, starts calm
    return ContextModel(cfg, rng, fixed_regime=None)
