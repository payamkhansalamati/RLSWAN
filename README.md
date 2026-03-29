# RLSWAN

Context-aware autonomous car-following with a safety-governed hybrid controller.

## Final System

Only these controllers are supported:
1. `pid`
2. `vanilla_ppo`
3. `context_ppo`
4. `hybrid_context_pid`

### Controller Roles

- `pid`: deterministic safety baseline with fixed-gap longitudinal control.
- `vanilla_ppo`: PPO on physical state only, no context in state or reward.
- `context_ppo`: PPO on augmented physical + context state with context-aware reward.
- `hybrid_context_pid`: context PPO nominal policy with risk-governed PID fallback.

## Architecture

### RL + Safety Stack

- Core environment: `env_car_following.py`
- RL training/inference: Stable-Baselines3 PPO (`train.py`)
- Safety fallback: `PIDController` (`controllers.py`)
- Hybrid governor: `HybridContextPIDController` (`controllers.py`)

Hybrid governance includes:
- Risk signals: TTC, gap shortfall, headway, required braking.
- Modes: `safe`, `caution`, `high_risk`, `emergency`.
- Safety filter in high/emergency that enforces conservative braking rather than soft blending.
- Diagnostics exported during evaluation for PID activity, mode occupancy, emergency usage, risk components, and filter activation/clamping.

## Project Structure

- `config.py`: centralized configuration.
- `registry.py`: single controller registry and validation helpers.
- `controllers.py`: PID and hybrid safety controller logic.
- `train.py`: PPO training for trainable controllers.
- `evaluate.py`: evaluation pipeline and aggregate metrics.
- `plots.py`: plotting pipeline (unchanged behavior).
- `main.py`: CLI entrypoint.
- `artifacts/models/`: trained models and checkpoints.
- `artifacts/results/`: evaluation CSVs and summaries.
- `artifacts/plots/`: generated figures.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Commands

### Train

Train supported PPO controllers:

```bash
python main.py train
```

Train a subset:

```bash
python main.py train --agents vanilla_ppo context_ppo --seeds 0 1
```

### Evaluate

Evaluate all final controllers:

```bash
python main.py evaluate
```

Evaluate a subset:

```bash
python main.py evaluate --agents pid context_ppo hybrid_context_pid --seeds 0 --n_episodes 10
```

### Plot

Generate plots from existing artifacts:

```bash
python main.py plot --traj_seed 0
```

### Full Pipeline

```bash
python main.py all
```

## Metrics

Evaluation exports per-episode and aggregate metrics under `artifacts/results/`.

Key categories:
- Safety: collision rate, minimum gap, headway/TTC violations.
- Comfort: jerk/acceleration magnitude and stability proxies.
- Performance: speed tracking and episode return.
- Hybrid diagnostics: PID active/dominant ratios, emergency ratio, mode distribution, risk statistics, filter statistics, transition counts.

## Key Findings (Current System Design)

- `context_ppo` is the nominal high-performance policy.
- `hybrid_context_pid` adds explicit safety governance without replacing context PPO globally.
- Hybrid action protection is mode-sensitive and physically grounded via TTC/gap/headway/required-braking risk.
- Unified registry and strict validation prevent unsupported controller usage in train/evaluate flows.

## Validation and Errors

The pipeline now fails early with clear messages when:
- an unknown controller name is passed to `train` or `evaluate`.
- a required trained model is missing for PPO/hybrid evaluation.
