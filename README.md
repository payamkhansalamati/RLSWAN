# Context-Aware Autonomous Car-Following

A production-quality Python research project demonstrating measurable
improvement of a **context-aware PPO** controller over a standard PID
baseline and a vanilla PPO agent in a 1D longitudinal car-following task.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [How to Train](#4-how-to-train)
5. [How to Evaluate](#5-how-to-evaluate)
6. [How to Plot](#6-how-to-plot)
7. [Full Pipeline](#7-full-pipeline)
8. [Environment Description](#8-environment-description)
9. [Context Model](#9-context-model)
10. [Reward Function](#10-reward-function)
11. [Ablation Study Design](#11-ablation-study-design)
12. [Expected Outputs](#12-expected-outputs)
13. [Limitations and Future Extensions](#13-limitations-and-future-extensions)

---

## 1. Project Overview

| Controller | State input | Reward |
|---|---|---|
| **PID Baseline** | gap, v_ego, v_lead | Fixed desired-headway control |
| **Vanilla PPO** | 5-dim physical state | Static safety + efficiency + comfort weights |
| **Ablation: PPO+Context State** (`ablation_state`) | 10-dim (physical + context) | Static reward (no context terms) |
| **Ablation: PPO+Context Reward** (`ablation_reward`) | 5-dim physical state | Context-aware reward, no context in obs |
| **Context-Aware PPO** | 10-dim (physical + context) | Dynamic reward modulated by inferred driver state |

The context vector `c_t = [κ_stress, κ_fatigue, h_pref, κ_comfort, κ_eff]`
is derived in real-time from synthetic physiological signals (heart rate, HRV,
respiration rate, fatigue index) by a deterministic rule-based interpreter that
mimics an LLM-style structured output.

The gain of the context-aware PPO is designed to be **most visible in
non-stationary driver-state scenarios** (e.g. the "switching" evaluation
scenario) where the preferred headway and comfort priority change over time.

---

## 2. Installation

### Prerequisites

- Python 3.10 or later
- pip

### Steps

```bash
# 1. Clone / navigate to project root
cd RLSWAN

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU support**: The default `requirements.txt` installs the CPU-only
> build of PyTorch.  For GPU, replace the torch line with
> `torch>=2.2.0` and install CUDA-enabled wheels from
> https://pytorch.org/get-started/locally/.

---

## 3. Project Structure

```
RLSWAN/
├── config.py              # All hyper-parameters via Python dataclasses
├── context_model.py       # Synthetic driver signals + context interpreter
├── env_car_following.py   # Custom Gymnasium 1D car-following environment
├── controllers.py         # PID baseline controller
├── train.py               # SB3 PPO training for all agent variants
├── evaluate.py            # Metric computation and CSV export
├── plots.py               # All Matplotlib visualisations
├── utils.py               # Shared helpers (seeds, normalisation, I/O)
├── main.py                # CLI entry point
├── requirements.txt
├── README.md
└── artifacts/             # Created automatically at runtime
    ├── models/            # Saved PPO models and VecNormalize stats
    ├── results/           # CSV metric files
    └── plots/             # PNG figures
```

---

## 4. How to Train

Train **all four PPO variants** (vanilla, both ablations, context-aware) across
5 seeds (≈ 20–30 min on a modern CPU):

```bash
python main.py train
```

Train only specific agents:

```bash
python main.py train --agents vanilla_ppo context_ppo
```

Override seeds and timesteps for a quick smoke-test:

```bash
python main.py train --seeds 0 1 --timesteps 200000
```

Models are saved to `artifacts/models/<agent_type>_seed<N>/`.

---

## 5. How to Evaluate

Evaluate ALL agents (including PID which needs no training):

```bash
python main.py evaluate
```

Evaluate only selected agents:

```bash
python main.py evaluate --agents pid context_ppo
```

Results are saved to:
- `artifacts/results/episode_metrics_<agent>.csv`  – per-episode rows
- `artifacts/results/aggregate_metrics.csv`         – mean ± std
- `artifacts/results/radar_scores.csv`              – normalised radar dims

A summary table is also printed to the console.

### Showcase Evaluation (context-sensitive cases)

Run the 4 deterministic showcase cases that target when context-awareness
matters most (stressed braking, fatigued stop-and-go, combined impairment,
escalation sequence):

```bash
python main.py evaluate_showcase
```

With options:

```bash
# Custom seeds and episodes
python main.py evaluate_showcase --seeds 700 701 702 --n_episodes 4

# Skip the optional escalation case
python main.py evaluate_showcase --no_escalation
```

Showcase results are saved to:
- `artifacts/results/showcase_episode_metrics.csv`  – raw per-episode rows
- `artifacts/results/showcase_summary.csv`          – mean ± std + relative improvements vs PID/Vanilla
- `artifacts/results/showcase_win_counts.csv`       – metric win-count table (context PPO vs baselines)
- `artifacts/results/showcase_seed_consistency.csv` – how often context PPO is better across seeds

---

## 6. How to Plot

Generate all figures from saved CSVs and trained models:

```bash
python main.py plot
```

Skip trajectory plots (faster when only bar/radar charts are needed):

```bash
python main.py plot --no_traj
```

Figures are saved to `artifacts/plots/`:
- `training_curve_<agent>.png`
- `trajectory_<agent>.png`
- `bar_metrics.png`
- `radar_scores.png`
- `scenario_comparison.png`

### Showcase Plots

Generate context-sensitive showcase charts and trajectory comparisons:

```bash
python main.py plot_showcase
```

With options:

```bash
python main.py plot_showcase --traj_seed 700
python main.py plot_showcase --no_escalation
```

Showcase figures saved to `artifacts/plots/`:
- `showcase_grouped_metrics.png`              – grouped bars (collision, HW violation, jerk, headway error)
- `showcase_case1_stressed_braking.png`       – 6-panel trajectory: PID vs Vanilla vs Context PPO
- `showcase_case2_fatigued_stopgo.png`
- `showcase_case3_both.png`
- `showcase_escalation.png`
- `showcase_context_adaptation.png`           – context variables + acceleration response (escalation)

---

## 7. Full Pipeline

Run train → evaluate → plot in one command:

```bash
python main.py all
```

Run only the showcase evaluation and showcase plots (models must already be trained):

```bash
python main.py all_showcase
```

Quick full run with reduced settings (≈ 5 min):

```bash
python main.py all --seeds 0 1 --timesteps 200000 --n_episodes 3
```

Recommended final comparison run (stronger statistical confidence):

```bash
python main.py all --seeds 0 1 2 3 4 --timesteps 300000 --n_episodes 5
```

---

## 8. Environment Description

**File:** `env_car_following.py`

### Dynamics (1D longitudinal)

```
v_ego[t+1]  = clip(v_ego[t] + a_cmd * dt,  v_min, v_max)
gap[t+1]    = gap[t] + (v_lead[t+1] - v_ego[t+1]) * dt
jerk[t]     = (a_cmd[t] - a_cmd[t-1]) / dt
headway[t]  = gap[t] / (v_ego[t] + ε)
```

### Default parameters

| Parameter | Value |
|---|---|
| dt | 0.1 s |
| Max episode steps | 500 |
| Initial gap | ~20 m |
| Ego speed bounds | [0, 33] m/s |
| Accel command bounds | [-3, +2] m/s² |
| Target speed | 15 m/s |
| Safe headway threshold | 1.5 s |

### Observation spaces

| Mode | Dims | Contents |
|---|---|---|
| Vanilla | 5 | gap, v_ego, v_lead, accel, jerk |
| Context-aware | 10 | + κ_stress, κ_fatigue, h_pref, κ_comfort, κ_eff |

### Lead-vehicle profiles

| Profile | Description |
|---|---|
| `constant` | Fixed speed sampled at episode start |
| `sinusoidal` | Slow sine-wave oscillation |
| `braking` | Single abrupt deceleration event |
| `stop_and_go` | Cyclic stop-restart |
| `random_smooth` | AR(1) speed perturbations |

### Evaluation Scenarios

| Scenario | Context behaviour |
|---|---|
| `calm` | Fixed calm regime throughout |
| `stressed` | Fixed stressed regime throughout |
| `fatigued` | Fixed fatigued regime throughout |
| `both` | Fixed stressed+fatigued regime throughout |
| `switching` / `mixed` | Stochastic regime transitions (hardest) |
| `escalation` | Deterministic transitions: calm→stressed→fatigued→both |

Profiles are selected randomly at each training episode to ensure diversity.

---

## 9. Context Model

**File:** `context_model.py`

### Architecture

```
LatentHumanStateSimulator
    ↓  stochastic regime switching  (calm / stressed / fatigued / both)
PhysiologicalSignalGenerator
    ↓  noisy signal sampling
    heart_rate, HRV, respiration_rate, fatigue_index
ContextInterpreter  (rule-based "LLM-like")
    ↓  mapping rules
    {stress_level, fatigue_level, preferred_headway,
     comfort_priority, efficiency_priority}
```

### Interpretation rules (key)

```
stress  = 0.45 * hr_norm + 0.40 * (1 - hrv_norm) + 0.15 * resp_norm
fatigue = 0.70 * fatigue_idx + 0.30 * (1 - hrv_norm)
h_pref  = 1.4 + 0.8 * stress + 0.8 * fatigue   [1.4 .. 3.0 s]
comfort = clip(0.3 + 0.5 * stress + 0.4 * fatigue, 0, 1)
eff     = clip(1.0 - 0.6 * stress - 0.5 * fatigue, 0, 1)
```

Small Gaussian noise (`noise_std = 0.05`) is added to all five outputs to
prevent the agent from over-fitting to a perfectly noiseless signal.

The hidden latent regime is **never** exposed to the agent.

### Deterministic Escalation Schedule

When `scenario="escalation"` is used, regime transitions are fixed at:

| Step | Regime |
|---|---|
| 0 | calm |
| 125 | stressed |
| 250 | fatigued |
| 375 | both |

---

## 10. Reward Function

### Context-aware PPO reward

```
r_t = r_safe_t
    + κ_eff_t  · r_eff_t
    + κ_comfort_t · r_comfort_t
    + r_context_t
```

where:

```
r_safe_t    = -w_col · I[gap≤0]  - w_gap · max(0, h_pref - h_t)
r_comfort_t = -(|jerk| + α · |accel|)
r_eff_t     = -|v_ego - v_target|
r_context_t = -λ_stress · κ_stress · |jerk|
              -λ_fatigue · κ_fatigue · max(0, accel)
              -λ_headway · |h_t - h_pref|
```

### Default weights

All components are further scaled by top-level balance multipliers
(`safety_balance`, `comfort_balance`, `efficiency_balance`, `context_balance`),
all defaulting to `1.0`. Changing them lets you transparently retune emphasis
without touching term definitions.

| Weight | Value | Purpose |
|---|---|---|
| `w_collision` | 160.0 | Dominant terminal safety penalty |
| `w_gap` | 8.0 | Quadratic headway violation |
| `w_distance_buffer` | 1.8 | Extra penalty near very short distances |
| `α_accel` | 0.16 | Acceleration contribution inside comfort term |
| `jerk_weight` | 0.11 | Jerk contribution inside comfort term |
| `efficiency_weight` | 0.5 | Speed-tracking pressure |
| `λ_stress` | 2.8 | Jerk penalty scaling with stress |
| `λ_fatigue` | 2.2 | Accel penalty scaling with fatigue |
| `λ_headway` | 2.0 | Headway compliance penalty |
| `context_squash_scale` | 10.0 | Tanh denominator (stabilises context term) |
| `context_clip` | 6.0 | Hard cap on context reward magnitude |

### Vanilla PPO reward (no context)

```
r_vanilla_t = r_safe_t + β_eff · r_eff_t + β_comfort · r_comfort_t
```

with fixed `β_eff = 0.8`, `β_comfort = 0.5`, `h_safe = 1.5 s`.

---

## 11. Ablation Study Design

Four models isolate the source of benefit:

| Variant | Context in obs? | Context in reward? | Purpose |
|---|---|---|---|
| `vanilla_ppo` | ✗ | ✗ | Baseline RL without any context |
| `ablation_state` | ✓ | ✗ | Test if augmented obs alone helps |
| `ablation_reward` | ✗ | ✓ | Test if context-shaping reward alone helps |
| `context_ppo` | ✓ | ✓ | Full context-aware agent |

- Comparing `vanilla_ppo` vs `ablation_state`: does richer state alone matter?
- Comparing `vanilla_ppo` vs `ablation_reward`: does context in reward alone matter?
- Comparing `ablation_state` vs `context_ppo`: separates reward modulation from observation richness.

### Showcase Cases (new)

Four deterministic curated cases test when context-awareness should matter most:

| Case | Scenario | Lead Profile | Hypothesis |
|---|---|---|---|
| `case1_stressed_braking` | stressed | braking | Context PPO keeps larger safety buffer |
| `case2_fatigued_stopgo` | fatigued | stop_and_go | Context PPO avoids aggressive re-acceleration |
| `case3_both_impairment` | both | random_smooth | Context PPO is most conservative |
| `case4_escalation` | escalation | braking | Context PPO adapts as context transitions |

---

## 12. Expected Outputs

After a full run (`python main.py all`):

```
artifacts/
├── models/
│   ├── vanilla_ppo_seed0/final_model.zip
│   ├── vanilla_ppo_seed0/vecnorm_stats.pkl
│   ├── context_ppo_seed0/…
│   ├── ablation_state_seed0/…
│   └── ablation_reward_seed0/…
├── results/
│   ├── episode_metrics_pid.csv
│   ├── episode_metrics_vanilla_ppo.csv
│   ├── episode_metrics_context_ppo.csv
│   ├── episode_metrics_ablation_state.csv
│   ├── episode_metrics_ablation_reward.csv
│   ├── aggregate_metrics.csv
│   └── radar_scores.csv
└── plots/
    ├── training_curve_vanilla_ppo.png
    ├── training_curve_context_ppo.png
    ├── training_curve_ablation_state.png
    ├── training_curve_ablation_reward.png
    ├── trajectory_pid.png
    ├── trajectory_vanilla_ppo.png
    ├── trajectory_context_ppo.png
    ├── trajectory_ablation_state.png
    ├── bar_metrics.png
    ├── radar_scores.png
    └── scenario_comparison.png
```

After `python main.py all_showcase` (requires trained models):

```
artifacts/
├── results/
│   ├── showcase_episode_metrics.csv
│   ├── showcase_summary.csv
│   ├── showcase_win_counts.csv
│   └── showcase_seed_consistency.csv
└── plots/
    ├── showcase_grouped_metrics.png
    ├── showcase_case1_stressed_braking.png
    ├── showcase_case2_fatigued_stopgo.png
    ├── showcase_case3_both.png
    ├── showcase_escalation.png
    └── showcase_context_adaptation.png
```

A console summary similar to the following is printed at the end of
evaluation:

```
========================================================================
  EVALUATION SUMMARY
========================================================================
Metric                     pid                 vanilla_ppo         context_ppo
------------------------------------------------------------------------
Collision Rate              0.050 ± 0.218        0.030 ± 0.170       0.010 ± 0.100
HW Violation [%]           18.200 ± 8.400        9.500 ± 5.200       5.100 ± 3.100
Mean |Jerk| [m/s³]          1.920 ± 0.340        1.450 ± 0.290       0.980 ± 0.210
Speed Error [m/s]           2.100 ± 0.510        1.300 ± 0.380       1.050 ± 0.310
Episode Return           -142.0 ± 28.0         -88.0 ± 19.0         -55.0 ± 14.0
========================================================================
```

*(Exact values depend on training seed and machine; the ordering of methods
should be consistent.)*

---

## 13. Limitations and Future Extensions

### Current limitations

- **1D environment only**: no lateral dynamics, overtaking, or multi-lane traffic.
- **Synthetic physiology**: the driver signals are rule-based simulations,
  not real sensor recordings; the context interpreter does not call a real LLM.
- **No perception noise**: the ego vehicle receives ground-truth gap and speed.
- **Fixed vehicle model**: no actuator lag, tyre models, or fuel consumption.
- **Offline training**: the context model parameters are fixed; no online
  adaptation from human feedback.

### Pathways to extension

| Direction | Suggested change |
|---|---|
| **Real LLM context** | Replace `ContextInterpreter` with an API call to GPT-4 / Llama returning JSON |
| **CARLA integration** | Wrap CARLA's Python API as a Gymnasium env and inject `ContextModel` |
| **SMARTS integration** | Use SMARTS's traffic scenarios; plug in `CarFollowingEnv` reward logic |
| **Real physiology** | Replace synthetic signals with wearable sensor streams via LSL |
| **Multi-lane** | Extend state with lane offsets and add discrete lane-change action |
| **Online adaptation** | Fine-tune context weights in real-time with meta-RL (MAML, RL²) |
| **Imitation learning** | Add a behavioural-cloning pre-training phase using human trajectories |
| **Safety guarantees** | Add a CBF (Control Barrier Function) safety filter on top of the RL policy |
