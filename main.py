"""Command-line interface for the final RLSWAN pipeline.

Commands
--------
  train      Train PPO controllers.
  evaluate   Evaluate PID, PPO, and hybrid controllers.
  plot       Generate plots from saved artifacts.
  all        Run train -> evaluate -> plot.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from config import Config
from registry import (
    ALL_CONTROLLERS,
    TRAINABLE_CONTROLLERS,
    validate_controllers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RLSWAN final train/evaluate/plot pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train PPO controllers.")
    train_p.add_argument(
        "--agents",
        nargs="+",
        default=list(TRAINABLE_CONTROLLERS),
        help="Controllers to train.",
    )
    train_p.add_argument("--seeds", type=int, nargs="+", default=None)
    train_p.add_argument("--timesteps", type=int, default=None)
    train_p.add_argument("--n_envs", type=int, default=None)

    eval_p = subparsers.add_parser("evaluate", help="Evaluate controllers.")
    eval_p.add_argument(
        "--agents",
        nargs="+",
        default=list(ALL_CONTROLLERS),
        help="Controllers to evaluate.",
    )
    eval_p.add_argument("--seeds", type=int, nargs="+", default=None)
    eval_p.add_argument("--n_episodes", type=int, default=None)

    plot_p = subparsers.add_parser("plot", help="Generate plots.")
    plot_p.add_argument("--no_traj", action="store_true")
    plot_p.add_argument("--traj_seed", type=int, default=100)

    all_p = subparsers.add_parser("all", help="Train, evaluate, then plot.")
    all_p.add_argument("--seeds", type=int, nargs="+", default=None)
    all_p.add_argument("--timesteps", type=int, default=None)
    all_p.add_argument("--n_envs", type=int, default=None)
    all_p.add_argument("--n_episodes", type=int, default=None)

    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    seeds = getattr(args, "seeds", None)
    timesteps = getattr(args, "timesteps", None)
    n_envs = getattr(args, "n_envs", None)
    n_episodes = getattr(args, "n_episodes", None)

    if seeds is not None:
        config.ppo.seeds = list(seeds)
        config.eval.eval_seeds = list(seeds)
    if timesteps is not None:
        config.ppo.total_timesteps = int(timesteps)
    if n_envs is not None:
        config.ppo.n_envs = int(n_envs)
    if n_episodes is not None:
        config.eval.episodes_per_scenario = int(n_episodes)
    return config


def cmd_train(args: argparse.Namespace, config: Config) -> None:
    from train import train_agent_all_seeds

    agents = validate_controllers(args.agents, allow=TRAINABLE_CONTROLLERS, context="training")
    seeds = config.ppo.seeds

    print(f"\n{'=' * 60}")
    print("  TRAINING")
    print(f"  Agents:     {agents}")
    print(f"  Seeds:      {seeds}")
    print(f"  Timesteps:  {config.ppo.total_timesteps:,}")
    print(f"  n_envs:     {config.ppo.n_envs}")
    print(f"{'=' * 60}\n")

    for agent_type in agents:
        train_agent_all_seeds(agent_type, config, seeds=seeds)

    print("\nTraining complete.")


def cmd_evaluate(args: argparse.Namespace, config: Config) -> None:
    from evaluate import evaluate_agent, _save_aggregate

    agents = validate_controllers(args.agents, allow=ALL_CONTROLLERS, context="evaluation")
    seeds = config.eval.eval_seeds

    print(f"\n{'=' * 60}")
    print("  EVALUATION")
    print(f"  Agents:     {agents}")
    print(f"  Seeds:      {seeds}")
    print(f"  Episodes/scenario: {config.eval.episodes_per_scenario}")
    print(f"{'=' * 60}\n")

    all_dfs = {}
    for agent_type in agents:
        print(f"\nEvaluating: {agent_type}")
        df = evaluate_agent(agent_type, config, seeds=seeds)
        if df.empty:
            print(f"  [WARNING] No results for {agent_type}")
            continue
        all_dfs[agent_type] = df
        csv_path = config.paths.results / f"episode_metrics_{agent_type}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}")

    if all_dfs:
        _save_aggregate(all_dfs, config)

    print("\nEvaluation complete.")


def cmd_plot(args: argparse.Namespace, config: Config) -> None:
    from evaluate import record_trajectory
    from plots import generate_all_plots

    print(f"\n{'=' * 60}")
    print("  GENERATING PLOTS")
    print(f"{'=' * 60}\n")

    trajectories = {}
    if not args.no_traj:
        for agent_type in list(ALL_CONTROLLERS):
            print(f"  Recording trajectory: {agent_type}")
            try:
                traj = record_trajectory(
                    agent_type,
                    config,
                    seed=args.traj_seed,
                    scenario="switching",
                    lead_profile="sinusoidal",
                )
                trajectories[agent_type] = traj
            except FileNotFoundError as exc:
                print(f"  [WARNING] {exc} - skipping trajectory plot for {agent_type}")

    generate_all_plots(config, trajectories=trajectories if trajectories else None)
    print("\nPlot generation complete.")


def cmd_all(_args: argparse.Namespace, config: Config) -> None:
    class _TrainArgs:
        agents = list(TRAINABLE_CONTROLLERS)

    class _EvalArgs:
        agents = list(ALL_CONTROLLERS)

    class _PlotArgs:
        no_traj = False
        traj_seed = 100

    cmd_train(_TrainArgs(), config)
    cmd_evaluate(_EvalArgs(), config)
    cmd_plot(_PlotArgs(), config)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config()
    config = apply_overrides(config, args)

    cmd_map = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "plot": cmd_plot,
        "all": cmd_all,
    }
    handler = cmd_map.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args, config)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(2)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(3)


if __name__ == "__main__":
    main()
