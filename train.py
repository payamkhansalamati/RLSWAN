"""Training pipeline for final trainable PPO controllers.

Trainable controllers
---------------------
    vanilla_ppo
    context_ppo

Trained models and normalisation statistics are saved under:
    artifacts/models/<agent_type>_seed<N>/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from config import Config, PPOConfig
from env_car_following import CarFollowingEnv, make_env, make_single_env
from registry import PPO_AGENT_SPECS, TRAINABLE_CONTROLLERS, get_ppo_spec
from utils import get_logger, set_global_seed, Timer, agent_name

logger = get_logger(__name__)


# Backward-compatible exported name used by existing non-plot modules.
AGENT_CONFIGS: Dict[str, tuple] = dict(PPO_AGENT_SPECS)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ProgressLogCallback(BaseCallback):
    """Log training progress and reward components at regular intervals."""

    def __init__(self, log_interval: int = 50_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_logged = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_logged >= self.log_interval:
            ep_rews = self.model.ep_info_buffer
            if ep_rews:
                mean_rew = np.mean([ep["r"] for ep in ep_rews])
            else:
                mean_rew = float("nan")
            logger.info(
                f"  steps={self.num_timesteps:>8,d}  "
                f"mean_ep_return={mean_rew:>8.2f}"
            )
            
            # Log reward-balance diagnostics from each environment.
            try:
                diag_list = self.training_env.env_method("get_reward_diagnostics")
                if diag_list:
                    ratios = np.array([
                        float(d.get("context_base_ratio", 0.0)) for d in diag_list
                    ], dtype=float)
                    ratios_scaled = np.array([
                        float(d.get("context_base_ratio_scaled", 0.0)) for d in diag_list
                    ], dtype=float)
                    ratio_ema = np.array([
                        float(d.get("context_base_ratio_ema", 0.0)) for d in diag_list
                    ], dtype=float)
                    reward_scale = float(diag_list[0].get("effective_context_scale", float("nan")))

                    logger.info(
                        "    context_reward_scale=%.4f | ratio(mean)=%.3f ratio_scaled(mean)=%.3f ratio_ema(mean)=%.3f",
                        reward_scale,
                        float(np.mean(ratios)),
                        float(np.mean(ratios_scaled)),
                        float(np.mean(ratio_ema)),
                    )
                    if float(np.max(ratios_scaled)) > 0.3:
                        logger.warning(
                            "WARNING: context reward dominating (max scaled ratio=%.3f > 0.3)",
                            float(np.max(ratios_scaled)),
                        )
            except (AttributeError, IndexError):
                pass
            
            self._last_logged = self.num_timesteps
        return True


class CurriculumStageCallback(BaseCallback):
    """Update vectorised training environments based on curriculum progress."""

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_timesteps = max(1, total_timesteps)
        self._last_stage: Optional[int] = None

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_training_progress", 0.0)
        self._last_stage = 1

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        self.training_env.env_method("set_training_progress", progress)
        if progress < 0.30:
            stage = 1
        elif progress < 0.70:
            stage = 2
        else:
            stage = 3
        if stage != self._last_stage:
            logger.info(f"  curriculum_stage={stage} at progress={progress:.2%}")
            self._last_stage = stage
        return True


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _build_vec_env(
    config: Config,
    use_context: bool,
    use_context_reward: bool,
    seed: int,
    n_envs: int,
) -> VecNormalize:
    """
    Build a vectorised + normalised training environment.

    Parameters
    ----------
    config, use_context, use_context_reward, seed:
        Passed to CarFollowingEnv.
    n_envs:
        Number of parallel envs.

    Returns
    -------
    VecNormalize
        Wrapped vectorised environment with observation/reward normalisation.
    """
    def _make(rank: int):
        def _init():
            env = CarFollowingEnv(
                config=config,
                use_context=use_context,
                use_context_reward=use_context_reward,
                lead_profile=None,   # random per episode
                scenario="curriculum" if config.curriculum.enabled else "mixed",
                seed=seed + rank,
                normalize_observation=True,
                training_safety_shield=config.curriculum.training_safety_shield,
            )
            env = Monitor(env)
            return env
        return _init

    venv = DummyVecEnv([_make(i) for i in range(n_envs)])
    venv = VecNormalize(
        venv,
        norm_obs=config.ppo.normalize_obs,
        norm_reward=config.ppo.normalize_rew,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config.ppo.gamma,
    )
    return venv


def _build_eval_env(
    config: Config,
    use_context: bool,
    use_context_reward: bool,
    seed: int,
    vecnorm: Optional[VecNormalize] = None,
) -> VecNormalize:
    """
    Build a single-env (non-training) normalised eval environment.
    Shares normalisation statistics with the training env if provided.
    """
    def _init():
        env = CarFollowingEnv(
            config=config,
            use_context=use_context,
            use_context_reward=use_context_reward,
            lead_profile=None,
            scenario="mixed",
            seed=seed + 9999,
            normalize_observation=True,
            training_safety_shield=False,
        )
        return Monitor(env)

    venv = DummyVecEnv([_init])
    if vecnorm is not None:
        # Reuse statistics from training env; only update obs stats
        venv = VecNormalize(
            venv,
            norm_obs=config.ppo.normalize_obs,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
        venv.obs_rms = vecnorm.obs_rms
        venv.ret_rms = vecnorm.ret_rms
    else:
        venv = VecNormalize(
            venv,
            norm_obs=config.ppo.normalize_obs,
            norm_reward=False,
            clip_obs=10.0,
        )
    return venv


def train_single_agent(
    agent_type: str,
    seed: int,
    config: Config,
) -> Path:
    """
    Train one PPO agent variant for a given seed.

    Parameters
    ----------
    agent_type:
        One of ('vanilla_ppo', 'context_ppo').
    seed:
        Random seed for reproducibility.
    config:
        Master Config dataclass.

    Returns
    -------
    Path
        Directory where the model and VecNormalize stats are saved.
    """
    use_context, use_context_reward = get_ppo_spec(agent_type)
    ppo_cfg = config.ppo
    paths = config.paths
    if ppo_cfg.checkpoint_eval_episodes < 1:
        raise ValueError("ppo.checkpoint_eval_episodes must be at least 1.")

    set_global_seed(seed)

    model_dir = paths.models / agent_name(agent_type, seed)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training {agent_type} | seed={seed} | "
        f"use_context={use_context} | use_context_reward={use_context_reward}"
    )
    logger.info(
        "  PPO params | lr=%.1e gamma=%.3f gae_lambda=%.3f clip=%.2f ent=%.4f vf=%.2f",
        ppo_cfg.learning_rate,
        ppo_cfg.gamma,
        ppo_cfg.gae_lambda,
        ppo_cfg.clip_range,
        ppo_cfg.ent_coef,
        ppo_cfg.vf_coef,
    )

    # Build environments
    train_env = _build_vec_env(config, use_context, use_context_reward, seed, ppo_cfg.n_envs)
    eval_env  = _build_eval_env(config, use_context, use_context_reward, seed, train_env)

    # PPO model
    activation_fn = torch.nn.Tanh if ppo_cfg.activation_fn.lower() == "tanh" else torch.nn.ReLU
    net_arch = list(ppo_cfg.net_arch)
    if agent_type == "context_ppo":
        net_arch = [256, 128]
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=activation_fn,
    )
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        learning_rate=ppo_cfg.learning_rate,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=ppo_cfg.verbose,
        tensorboard_log=str(paths.results / "tb_logs"),
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, ppo_cfg.total_timesteps // 10),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ckpt",
        verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir),
        eval_freq=max(1, ppo_cfg.total_timesteps // (10 * ppo_cfg.n_envs)),
        n_eval_episodes=ppo_cfg.checkpoint_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )
    progress_cb = ProgressLogCallback(log_interval=max(2_000, ppo_cfg.total_timesteps // 4))
    callback_items = [checkpoint_cb, eval_cb, progress_cb]
    if config.curriculum.enabled:
        callback_items.append(CurriculumStageCallback(total_timesteps=ppo_cfg.total_timesteps))
    callbacks = CallbackList(callback_items)

    with Timer(f"{agent_type}/seed={seed}"):
        model.learn(
            total_timesteps=ppo_cfg.total_timesteps,
            callback=callbacks,
            tb_log_name=f"{agent_type}_seed{seed}",
            reset_num_timesteps=True,
            progress_bar=False,
        )

    # Save final model
    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    train_env.save(str(model_dir / "vecnorm_stats.pkl"))
    logger.info(f"  Saved model → {final_model_path}")

    train_env.close()
    eval_env.close()

    return model_dir


def train_agent_all_seeds(
    agent_type: str,
    config: Config,
    seeds: Optional[List[int]] = None,
) -> List[Path]:
    """
    Train an agent across all requested seeds.

    Parameters
    ----------
    agent_type:
        One of ('vanilla_ppo', 'context_ppo').
    config:
        Master Config dataclass.
    seeds:
        Override seed list; defaults to config.ppo.seeds.

    Returns
    -------
    List[Path]
        Saved model directories, one per seed.
    """
    if agent_type not in TRAINABLE_CONTROLLERS:
        allowed = ", ".join(TRAINABLE_CONTROLLERS)
        raise ValueError(
            f"Unknown controller for training: {agent_type}. Allowed: {allowed}."
        )

    seeds = seeds or config.ppo.seeds
    model_dirs: List[Path] = []
    for seed in seeds:
        model_dir = train_single_agent(agent_type, seed, config)
        model_dirs.append(model_dir)
    return model_dirs


def train_all(config: Config) -> None:
    """
    Train all final trainable controllers (vanilla_ppo, context_ppo)
    across all seeds specified in config.ppo.seeds.
    """
    logger.info("=" * 60)
    logger.info("Starting training of ALL agent variants")
    logger.info(f"  Seeds:           {config.ppo.seeds}")
    logger.info(f"  Total timesteps: {config.ppo.total_timesteps:,}")
    logger.info("=" * 60)

    for agent_type in TRAINABLE_CONTROLLERS:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Agent: {agent_type}")
        logger.info(f"{'─'*60}")
        with Timer(f"Total for {agent_type}"):
            train_agent_all_seeds(agent_type, config)

    logger.info("\nAll training runs complete.")


# ---------------------------------------------------------------------------
# Model loading helper (used by evaluate.py)
# ---------------------------------------------------------------------------

def load_trained_model(
    agent_type: str,
    seed: int,
    config: Config,
) -> tuple:
    """
    Load a previously saved PPO model and corresponding VecNormalize wrapper.

    Returns
    -------
    (model, vecnorm_stats_path)
    """
    use_context, use_context_reward = get_ppo_spec(agent_type)
    model_dir = config.paths.models / agent_name(agent_type, seed)
    best_model_path = model_dir / "best_model.zip"
    final_model_path = model_dir / "final_model.zip"
    vecnorm_path = model_dir / "vecnorm_stats.pkl"

    # Prefer best model; fall back to final
    if best_model_path.exists():
        model_path = best_model_path
    elif final_model_path.exists():
        model_path = final_model_path
    else:
        allowed = ", ".join(TRAINABLE_CONTROLLERS)
        raise FileNotFoundError(
            f"Missing trained model for '{agent_type}' (seed={seed}) in {model_dir}.\n"
            f"Train first with: python main.py train --agents {agent_type} --seeds {seed}\n"
            f"Allowed trainable controllers: {allowed}."
        )

    model = PPO.load(str(model_path))
    logger.debug(f"Loaded model from {model_path}")
    return model, vecnorm_path, use_context, use_context_reward
