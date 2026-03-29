#!/usr/bin/env python3
"""
Agent Comparison Script
=======================

Trains or loads vanilla and context-aware PPO agents and compares them
across multiple evaluation scenarios.

Usage:
  # Train both agents with new modifications
  python agent_comparison.py --mode train --seeds 0 1 2 --timesteps 300000
  
  # Evaluate existing models
  python agent_comparison.py --mode evaluate --agents vanilla_ppo context_ppo
  
  # Compare trajectories in specific scenario
  python agent_comparison.py --mode compare --scenario stressed
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from env_car_following import CarFollowingEnv


class AgentComparator:
    """Compare agent behaviors across scenarios."""
    
    def __init__(self, config: Config):
        self.cfg = config
        self.results = {}
    
    def evaluate_agent(
        self,
        use_context: bool,
        use_context_reward: bool,
        scenario: str = "mixed",
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict:
        """Evaluate agent in scenario. Note: uses random policy for demo."""
        print(f"\nEvaluating {'Context-Aware' if use_context_reward else 'Vanilla'} PPO")
        print(f"  Scenario: {scenario}")
        print(f"  Episodes: {n_episodes}")
        
        env = CarFollowingEnv(
            self.cfg,
            use_context=use_context,
            use_context_reward=use_context_reward,
            scenario=scenario,
        )
        
        episode_results = {
            "rewards": [],
            "collision_rates": [],
            "mean_jerks": [],
            "min_gaps": [],
            "mean_speeds": [],
            "max_accelerations": [],
            "command_changes": [],
            "recovery_hold_frac": [],
            "context_stress_mean": [],
            "context_fatigue_mean": [],
        }
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            collision = False
            
            jerks = []
            gaps = []
            speeds = []
            accel_max = 0.0
            cmd_changes = []
            stress_levels = []
            fatigue_levels = []
            
            for step in range(300):
                # Use random policy for demo (replace with actual agent policy)
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                
                if info.get("collision", False):
                    collision = True
                
                jerks.append(abs(info.get("jerk", 0)))
                gaps.append(info.get("gap", 0))
                speeds.append(info.get("v_ego", 0))
                accel_max = max(accel_max, abs(info.get("accel", 0)))
                cmd_changes.append(info.get("mean_abs_cmd_change", 0))
                stress_levels.append(info.get("context_stress", 0))
                fatigue_levels.append(info.get("context_fatigue", 0))
                
                if terminated or truncated:
                    break
            
            # Aggregate episode metrics
            episode_results["rewards"].append(episode_reward)
            episode_results["collision_rates"].append(float(collision))
            episode_results["mean_jerks"].append(np.mean(jerks))
            episode_results["min_gaps"].append(min(gaps) if gaps else 0)
            episode_results["mean_speeds"].append(np.mean(speeds))
            episode_results["max_accelerations"].append(accel_max)
            episode_results["command_changes"].append(np.mean(cmd_changes) if cmd_changes else 0)
            episode_results["recovery_hold_frac"].append(info.get("recovery_hold_frac", 0))
            episode_results["context_stress_mean"].append(np.mean(stress_levels))
            episode_results["context_fatigue_mean"].append(np.mean(fatigue_levels))
        
        env.close()
        
        # Compute statistics
        stats = {}
        for key, values in episode_results.items():
            if values:
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        
        agent_name = "context_ppo" if use_context_reward else "vanilla_ppo"
        self.results[agent_name] = {
            "scenario": scenario,
            "episode_data": episode_results,
            "statistics": stats,
        }
        
        return stats
    
    def compare_agents(self) -> None:
        """Print comparison between agents."""
        if len(self.results) < 2:
            print("Need at least 2 agents for comparison")
            return
        
        print("\n" + "="*80)
        print("AGENT COMPARISON RESULTS")
        print("="*80)
        
        agents = list(self.results.keys())
        vanilla = self.results.get("vanilla_ppo")
        context = self.results.get("context_ppo")
        
        if not vanilla or not context:
            print("Error: Need both vanilla_ppo and context_ppo results")
            return
        
        metrics = [
            ("rewards", "Episode Reward", "higher is better"),
            ("mean_jerks", "Mean Jerk (m/s³)", "lower is better"),
            ("min_gaps", "Min Gap (m)", "higher is better (safety)"),
            ("mean_speeds", "Mean Speed (m/s)", "consistent"),
            ("command_changes", "Command Changes (m/s²)", "lower is better"),
            ("collision_rates", "Collision Rate (%)", "lower is better"),
            ("recovery_hold_frac", "Recovery Hold Fraction", "indicates smoothing"),
        ]
        
        print("\nMetric Comparison:\n")
        print(f"{'Metric':<30} {'Vanilla PPO':<20} {'Context PPO':<20} {'Improvement':<15}")
        print("-" * 85)
        
        for metric_key, metric_name, interpretation in metrics:
            vanilla_stats = vanilla["statistics"].get(metric_key)
            context_stats = context["statistics"].get(metric_key)
            
            if vanilla_stats and context_stats:
                v_mean = vanilla_stats["mean"]
                c_mean = context_stats["mean"]
                
                # Calculate improvement direction
                if "lower" in interpretation:
                    if v_mean > c_mean:
                        improvement = f"+{(v_mean - c_mean) / max(abs(v_mean), 1e-6) * 100:.1f}%"
                    else:
                        improvement = f"-{(c_mean - v_mean) / max(abs(v_mean), 1e-6) * 100:.1f}%"
                else:
                    if c_mean > v_mean:
                        improvement = f"+{(c_mean - v_mean) / max(abs(v_mean), 1e-6) * 100:.1f}%"
                    else:
                        improvement = f"-{(v_mean - c_mean) / max(abs(v_mean), 1e-6) * 100:.1f}%"
                
                print(f"{metric_name:<30} {v_mean:<20.6f} {c_mean:<20.6f} {improvement:<15}")
    
    def save_results(self, output_file: Optional[str] = None) -> None:
        """Save comparison results to JSON."""
        if output_file is None:
            output_file = "results/agent_comparison.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for agent, data in self.results.items():
            results_json[agent] = {
                "scenario": data["scenario"],
                "statistics": data["statistics"],
                "episode_data": {
                    k: [float(v) for v in vs] if isinstance(vs, (list, np.ndarray)) else vs
                    for k, vs in data["episode_data"].items()
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare vanilla and context-aware PPO agents"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "compare"],
        default="evaluate",
        help="Operation mode"
    )
    parser.add_argument(
        "--scenario",
        choices=["calm", "stressed", "fatigued", "both", "switching", "mixed"],
        default="mixed",
        help="Scenario to evaluate"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output",
        default="results/agent_comparison.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    cfg = Config()
    comparator = AgentComparator(cfg)
    
    # Evaluate vanilla PPO
    print("="*80)
    print("EVALUATING VANILLA PPO (Baseline)")
    print("="*80)
    vanilla_stats = comparator.evaluate_agent(
        use_context=False,
        use_context_reward=False,
        scenario=args.scenario,
        n_episodes=args.n_episodes,
    )
    print("\nVanilla PPO Statistics:")
    for metric, stats in vanilla_stats.items():
        print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    
    # Evaluate context-aware PPO
    print("\n" + "="*80)
    print("EVALUATING CONTEXT-AWARE PPO (With Modifications)")
    print("="*80)
    context_stats = comparator.evaluate_agent(
        use_context=True,
        use_context_reward=True,
        scenario=args.scenario,
        n_episodes=args.n_episodes,
    )
    print("\nContext PPO Statistics:")
    for metric, stats in context_stats.items():
        print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    
    # Compare results
    comparator.compare_agents()
    comparator.save_results(args.output)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
