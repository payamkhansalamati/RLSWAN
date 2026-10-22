# RLSWAN

**[▶ Watch the concept demo (60 s)](docs/media/context_aware_assistance_concept_demo.mp4)**

RLSWAN is a simulation for context-aware longitudinal car-following.
It compares a fixed-gap PID controller, two Stable-Baselines3 PPO policies, and
a hybrid context-PPO/PID controller in a synthetic Gymnasium environment.
The driver context is generated and interpreted by repository code; it is not
collected from a person or produced by a language model.

## Implemented approach

`CarFollowingEnv` simulates ego and lead vehicle speeds, their separation,
acceleration, and jerk. The lead vehicle follows a scripted profile
(`constant`, `sinusoidal`, `braking`, `stop_and_go`, or `random_smooth`). Episodes
end on collision or are truncated at the configured time horizon.

`ContextModel` creates a hidden synthetic driver regime (calm, stressed,
fatigued, or both), generates noisy physiological readings, and maps those
readings to a five-value context vector:

`[stress, fatigue, preferred_headway, comfort_priority, efficiency_priority]`

The latent regime and simulated physiological readings are not directly
observed by a policy. The interpreted context is available during both
training and evaluation; it is included in the `context_ppo` observation and
used by its context-shaped reward. `vanilla_ppo` receives physical state only
and does not use the context-shaped reward. Both policies use continuous
acceleration actions and PPO. The hybrid controller loads `context_ppo` as its
nominal policy and applies a PID fallback/filter according to physical risk
signals. In addition, when `context_ppo` is trained or evaluated, the
environment applies context-conditioned risk safeguards and action processing
(including smoothing, command slew limits, and recovery hold). Thus the
`vanilla_ppo` versus `context_ppo` comparison changes more than policy input
and reward alone.

### Controllers

| Name | Observation and behavior |
| --- | --- |
| `pid` | Fixed-gap PID baseline; no learned policy. |
| `vanilla_ppo` | PPO using five physical features, without context input or context reward. |
| `context_ppo` | PPO using the physical features plus five context values, with the context-shaped reward enabled. |
| `hybrid_context_pid` | Evaluation-only controller using `context_ppo` with risk-governed PID intervention. |

The physical observation is `[gap, ego_speed, lead_speed, acceleration, jerk]`.
The environment manually scales and clips observations before PPO's optional
`VecNormalize` observation normalization. The context policy's MLP is
`[256, 128]`; vanilla PPO uses the configured `[128, 128]` architecture by
default. PPO uses the activation and other hyperparameters in `config.py`.
Training also uses the configured curriculum and training-time safety shield.
These are properties of the implemented experiments, not guarantees of real
vehicle safety.

## Repository layout

- `config.py`: simulation, synthetic context, reward, controller, training,
  evaluation, and output-path settings.
- `context_model.py`: hidden regime simulator, physiological signal generator,
  and rule-based context interpreter.
- `env_car_following.py`: Gymnasium environment, lead profiles, reward, and
  observation construction.
- `controllers.py`: PID, PPO wrapper, and hybrid context-PPO/PID logic.
- `registry.py`: supported controllers and PPO environment settings.
- `train.py`: PPO setup, training callbacks, checkpoint selection, and model
  loading.
- `evaluate.py`: episode rollouts, metrics, aggregation, and trajectory
  recording.
- `plots.py`: plots from evaluation outputs and recorded trajectories.
- `main.py`: command-line entry point.
- `test_reward_validation.py`, `test_smoke_comprehensive.py`: existing
  environment and reward smoke tests.
- `Results01.docx`: a historical result summary. Its connection to committed
  logs, configurations, checkpoints, and evaluation outputs has not been
  verified; do not treat it as a reproducible result from the current code.

Generated models, logs, CSV results, and plots are written under `artifacts/`
by default. This output directory is ignored by Git. No trained model or
evaluation result is included in the tracked repository.

## Requirements and installation

The dependency manifest targets Python 3.8 through 3.12. It specifies
Gymnasium, Stable-Baselines3, NumPy, SciPy, PyTorch, pandas, and Matplotlib;
TensorBoard is optional for training curves.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependency ranges are version constraints, not a lockfile. GPU use requires
installing a PyTorch build compatible with the host's CUDA setup.

## Train

Train the two PPO controllers for the five configured seeds and 500,000
environment steps per seed:

```bash
python main.py train
```

Select controllers, seeds, training steps, parallel environments, or the
number of episodes used by each training-time checkpoint evaluation:

```bash
python main.py train --agents context_ppo --seeds 0 1 --timesteps 100000 \
  --n_envs 4 --checkpoint-eval-episodes 5
```

Training-time checkpoint selection uses
`config.ppo.checkpoint_eval_episodes` (default **5**). This is the number of
episodes in each Stable-Baselines3 `EvalCallback` evaluation, which chooses and
saves `best_model.zip`. It is independent of the final evaluation episode
count. Models and VecNormalize statistics are saved to
`artifacts/models/<controller>_seed<seed>/`; intermediate checkpoints are
saved below that directory. The final model is also saved as `final_model.zip`.
When loading, evaluation prefers `best_model.zip` and falls back to
`final_model.zip`.

## Final evaluation

Evaluate all four controllers over the configured seeds and four scenarios:

```bash
python main.py evaluate
```

The default final evaluation runs **50 episodes per scenario** for each seed.
The four scenarios are calm, stressed, fatigued, and switching. PPO and hybrid
evaluation require trained models for each requested seed. To run a smaller
evaluation or a subset:

```bash
python main.py evaluate --agents pid context_ppo --seeds 0 \
  --episodes-per-scenario 10
```

Final evaluation uses `config.eval.episodes_per_scenario` (default **50**),
independently of `config.ppo.checkpoint_eval_episodes`. The legacy CLI spelling
`--n_episodes` is also accepted for `evaluate`.

Per-episode metrics are saved as
`artifacts/results/episode_metrics_<controller>.csv`; aggregate metrics and
normalized radar scores are saved as `aggregate_metrics.csv` and
`radar_scores.csv`. Metrics include collision occurrence, minimum gap,
headway/TTC violations, jerk and acceleration, speed-tracking error, progress,
episode return, reward decomposition, and hybrid intervention diagnostics.
`collision_rate` is an episode indicator (1 if any collision occurred, else
0); aggregate means are averaged across evaluated episodes. The aggregate
standard deviation is computed from the episode metric values. Radar scores
are normalized using fixed bounds in `utils.py`; they are not probabilities
or statistical significance measures.

## Plot and combined commands

Generate plots from existing result files and record the default trajectory
examples:

```bash
python main.py plot --traj_seed 0
```

Use `--no_traj` to skip trajectory recording. Recording a PPO or hybrid
trajectory requires the corresponding checkpoint. Run the full configured
training, final evaluation, and plotting sequence with:

```bash
python main.py all
```

`all` uses the same independent checkpoint-selection and final-evaluation
settings described above.

## Presentation animations

The reproducible animation workflow, its generation commands, safety
definitions, and interpretation limits are documented in
[animations/README.md](./animations/README.md). It records measured trajectories
and renders two 1920×1080 MP4s under `animations/rendered/`; checkpoint
references, seed/config metadata, and per-step CSV data are stored under
`animations/data/`. The risk-governed controller shown is the implemented
`hybrid_context_pid` wrapper around `context_ppo`, not a separate PPO model.

## Reproducibility and limitations

Training and evaluation default to seeds `0` through `4`. The code seeds
Python, NumPy, and PyTorch and seeds environment instances, but exact
reproducibility is not guaranteed across hardware, dependency versions, or
nondeterministic backend operations. Reported evaluation outcomes depend on
the saved model, configuration, code revision, and evaluation seeds.

The context signals and driver regimes are synthetic proxies. They are not
validated physiological measurements, a model of an identified population,
or evidence for real-world driving safety. Historical result summaries are
not independently verifiable unless their source configurations, checkpoints,
logs, and evaluation data can be matched.
