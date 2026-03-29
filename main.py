"""
main.py
=======
Command-line interface for the Context-Aware Car-Following project.

Commands
--------
  train      – Train all PPO agent variants across all seeds
  evaluate   – Evaluate all agents and save metric CSVs
  diagnose   – Run context-sensitivity diagnostics on trained policies
  plot       – Generate all plots from saved results
  plot_diagnostics – Generate reward diagnostic plots only (base vs context)
  evaluate_showcase – Run deterministic context-sensitive showcase cases
  plot_showcase – Generate showcase-focused plots
  all_showcase – Evaluate showcase cases then plot showcase outputs
  all        – Run train → evaluate → plot in sequence

Usage examples
--------------
  python main.py train
  python main.py evaluate
  python main.py plot
  python main.py plot_diagnostics
  python main.py all

  # Quick ablation-only training with fewer timesteps and 2 seeds
  python main.py train --agents vanilla_ppo context_ppo --seeds 0 1 --timesteps 200000

  # Evaluate a single agent
  python main.py evaluate --agents pid context_ppo

  # Training with a custom config override (via monkey-patching config)
  python main.py all --timesteps 300000 --n_envs 4 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from config import Config


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Context-Aware Autonomous Car-Following – Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- TRAIN ----
    train_p = subparsers.add_parser("train", help="Train PPO agent variants.")
    train_p.add_argument(
        "--agents", nargs="+",
        default=["vanilla_ppo", "context_ppo", "ablation_state", "ablation_reward"],
        choices=["vanilla_ppo", "context_ppo", "ablation_state", "ablation_reward"],
        help="Which agent types to train.",
    )
    train_p.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Override random seeds (default: config.ppo.seeds)")
    train_p.add_argument("--timesteps", type=int, default=None,
                         help="Total training timesteps per run.")
    train_p.add_argument("--n_envs", type=int, default=None,
                         help="Number of parallel envs.")

    # ---- EVALUATE ----
    eval_p = subparsers.add_parser("evaluate", help="Evaluate trained agents.")
    eval_p.add_argument(
        "--agents", nargs="+",
        default=["pid", "vanilla_ppo", "context_ppo", "hybrid_context_pid"],
        help="Which agents to evaluate.",
    )
    eval_p.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Evaluation seeds.")
    eval_p.add_argument("--n_episodes", type=int, default=None,
                        help="Episodes per scenario per seed.")

    # ---- DIAGNOSE ----
    diag_p = subparsers.add_parser("diagnose", help="Run context sensitivity diagnostics.")
    diag_p.add_argument("--seed", type=int, default=0,
                        help="Model seed to load for diagnostics.")

    # ---- PLOT ----
    plot_p = subparsers.add_parser("plot", help="Generate all plots.")
    plot_p.add_argument("--no_traj", action="store_true",
                        help="Skip per-agent trajectory plots (faster).")
    plot_p.add_argument("--traj_seed", type=int, default=100,
                        help="Seed for demonstrative trajectory episode.")

    # ---- PLOT DIAGNOSTICS ----
    diag_plot_p = subparsers.add_parser(
        "plot_diagnostics",
        help="Generate reward diagnostic plots only (base vs context reward analysis).",
    )

    # ---- EVALUATE SHOWCASE ----
    eval_show_p = subparsers.add_parser(
        "evaluate_showcase",
        help="Run deterministic showcase evaluation cases.",
    )
    eval_show_p.add_argument(
        "--agents", nargs="+",
        default=["pid", "vanilla_ppo", "context_ppo", "hybrid_context_pid"],
        help="Agents for showcase evaluation.",
    )
    eval_show_p.add_argument("--seeds", type=int, nargs="+", default=None,
                             help="Showcase evaluation seeds.")
    eval_show_p.add_argument("--n_episodes", type=int, default=None,
                             help="Episodes per showcase case per seed.")
    eval_show_p.add_argument("--no_escalation", action="store_true",
                             help="Disable optional escalation showcase case.")

    # ---- PLOT SHOWCASE ----
    plot_show_p = subparsers.add_parser(
        "plot_showcase",
        help="Generate showcase-focused charts and trajectory comparisons.",
    )
    plot_show_p.add_argument("--traj_seed", type=int, default=700,
                             help="Seed for showcase trajectory recording.")
    plot_show_p.add_argument("--no_escalation", action="store_true",
                             help="Disable optional escalation showcase plots.")

    # ---- ALL ----
    all_p = subparsers.add_parser("all", help="Train, evaluate, then plot.")
    all_p.add_argument("--seeds", type=int, nargs="+", default=None)
    all_p.add_argument("--timesteps", type=int, default=None)
    all_p.add_argument("--n_envs", type=int, default=None)
    all_p.add_argument("--n_episodes", type=int, default=None)

    # ---- ALL SHOWCASE ----
    all_show_p = subparsers.add_parser(
        "all_showcase",
        help="Run showcase evaluation then showcase plotting.",
    )
    all_show_p.add_argument("--agents", nargs="+", default=["pid", "vanilla_ppo", "context_ppo", "hybrid_context_pid"])
    all_show_p.add_argument("--seeds", type=int, nargs="+", default=None)
    all_show_p.add_argument("--n_episodes", type=int, default=None)
    all_show_p.add_argument("--traj_seed", type=int, default=700)
    all_show_p.add_argument("--no_escalation", action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------

def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply any CLI overrides to the config in-place."""
    seeds        = getattr(args, "seeds", None)
    timesteps    = getattr(args, "timesteps", None)
    n_envs       = getattr(args, "n_envs", None)
    n_episodes   = getattr(args, "n_episodes", None)

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


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace, config: Config) -> None:
    from train import train_agent_all_seeds
    agents = args.agents
    seeds  = config.ppo.seeds

    print(f"\n{'=' * 60}")
    print(f"  TRAINING")
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
    import pandas as pd

    agents = args.agents
    seeds  = config.eval.eval_seeds

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION")
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
    from plots import generate_all_plots
    from evaluate import record_trajectory
    from train import AGENT_CONFIGS

    print(f"\n{'=' * 60}")
    print(f"  GENERATING PLOTS")
    print(f"{'=' * 60}\n")

    trajectories = {}
    if not args.no_traj:
        for agent_type in ["pid", "vanilla_ppo", "context_ppo", "hybrid_context_pid"]:
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
                print(f"  [WARNING] {exc} – skipping trajectory plot for {agent_type}")

    generate_all_plots(config, trajectories=trajectories if trajectories else None)
    print("\nPlot generation complete.")


def cmd_plot_diagnostics(args: argparse.Namespace, config: Config) -> None:
    """Generate reward diagnostic plots only (base vs context analysis)."""
    from plots import plot_reward_diagnostics

    print(f"\n{'=' * 60}")
    print(f"  GENERATING REWARD DIAGNOSTIC PLOTS")
    print(f"{'=' * 60}\n")

    plot_reward_diagnostics(config)
    print("\nReward diagnostic plots complete.")


def cmd_all(args: argparse.Namespace, config: Config) -> None:
    """Run train → evaluate → plot in full."""
    # Train
    class _TrainArgs:
        agents = ["vanilla_ppo", "context_ppo", "ablation_state", "ablation_reward"]
        seeds  = None
        timesteps = None
        n_envs = None

    train_args = _TrainArgs()
    cmd_train(train_args, config)

    # Evaluate
    class _EvalArgs:
        agents = ["pid", "vanilla_ppo", "context_ppo", "hybrid_context_pid"]
        seeds  = None
        n_episodes = None

    eval_args = _EvalArgs()
    cmd_evaluate(eval_args, config)

    # Plot
    class _PlotArgs:
        no_traj = False
        traj_seed = 100

    plot_args = _PlotArgs()
    cmd_plot(plot_args, config)

    class _DiagArgs:
        seed = 0

    diag_args = _DiagArgs()
    cmd_diagnose(diag_args, config)


def cmd_diagnose(args: argparse.Namespace, config: Config) -> None:
    from evaluate import run_context_sensitivity_diagnostics

    print(f"\n{'=' * 60}")
    print("  CONTEXT DIAGNOSTICS")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    df = run_context_sensitivity_diagnostics(config, seed=args.seed)
    if not df.empty:
        print(df.round(4).to_string(index=False))
    print("\nDiagnostics complete.")


def cmd_evaluate_showcase(args: argparse.Namespace, config: Config) -> None:
    from evaluate import evaluate_showcase

    print(f"\n{'=' * 60}")
    print("  SHOWCASE EVALUATION")
    print(f"  Agents: {args.agents}")
    print(f"  Seeds: {args.seeds if args.seeds is not None else config.eval.showcase_seeds}")
    print(f"  Episodes/case: {args.n_episodes if args.n_episodes is not None else config.eval.showcase_episodes_per_case}")
    print(f"  Include escalation: {not args.no_escalation}")
    print(f"{'=' * 60}\n")

    df = evaluate_showcase(
        config=config,
        agents=args.agents,
        seeds=args.seeds,
        n_episodes_per_case=args.n_episodes,
        include_optional_escalation=not args.no_escalation,
    )
    print(f"\nShowcase episodes evaluated: {len(df)}")


def cmd_plot_showcase(args: argparse.Namespace, config: Config) -> None:
    from plots import generate_showcase_plots

    print(f"\n{'=' * 60}")
    print("  SHOWCASE PLOTS")
    print(f"  traj_seed: {args.traj_seed}")
    print(f"  Include escalation: {not args.no_escalation}")
    print(f"{'=' * 60}\n")

    generate_showcase_plots(
        config=config,
        include_optional_escalation=not args.no_escalation,
        traj_seed=args.traj_seed,
    )
    print("\nShowcase plots complete.")


def cmd_all_showcase(args: argparse.Namespace, config: Config) -> None:
    """Run showcase evaluation then showcase plotting."""

    class _EvalShowArgs:
        agents = args.agents
        seeds = args.seeds
        n_episodes = args.n_episodes
        no_escalation = args.no_escalation

    class _PlotShowArgs:
        traj_seed = args.traj_seed
        no_escalation = args.no_escalation

    cmd_evaluate_showcase(_EvalShowArgs(), config)
    cmd_plot_showcase(_PlotShowArgs(), config)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Build master config and apply any CLI overrides
    config = Config()
    config = apply_overrides(config, args)

    # Dispatch command
    cmd_map = {
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "diagnose": cmd_diagnose,
        "plot":     cmd_plot,
        "plot_diagnostics": cmd_plot_diagnostics,
        "all":      cmd_all,
        "evaluate_showcase": cmd_evaluate_showcase,
        "plot_showcase": cmd_plot_showcase,
        "all_showcase": cmd_all_showcase,
    }

    handler = cmd_map.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args, config)


if __name__ == "__main__":
    main()
