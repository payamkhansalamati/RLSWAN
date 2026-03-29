#!/usr/bin/env python3
"""
Training and Evaluation Plan for Context-Aware Car-Following Modifications
===========================================================================

This script provides:
1. Comprehensive training configuration examples
2. Evaluation and comparison utilities
3. Visualization helpers
4. Results aggregation

Run: python training_plan.py --help
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import Config


def print_training_config_vanilla():
    """Print recommended training config for vanilla PPO."""
    config = {
        "agent": "vanilla_ppo",
        "description": "Baseline PPO without context reward",
        "environment": {
            "use_context": False,
            "use_context_reward": False,
            "normalize_observation": True,
            "training_safety_shield": True,
        },
        "training": {
            "total_timesteps": 300000,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 1e-3,
        },
        "curriculum": {
            "enabled": True,
            "stage1_fraction": 0.33,
            "stage2_fraction": 0.66,
        },
        "notes": [
            "Uses fixed action smoothing (alpha=0.30)",
            "No context-aware reward modulation",
            "Standard safety shield protection",
            "Good for baseline comparison",
        ]
    }
    return config


def print_training_config_context():
    """Print recommended training config for context-aware PPO."""
    config = {
        "agent": "context_ppo",
        "description": "PPO with context-aware reward and control",
        "environment": {
            "use_context": True,
            "use_context_reward": True,
            "normalize_observation": True,
            "training_safety_shield": True,
        },
        "training": {
            "total_timesteps": 300000,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 1e-3,
        },
        "curriculum": {
            "enabled": True,
            "stage1_fraction": 0.33,
            "stage2_fraction": 0.66,
        },
        "key_features": {
            "risk_aware_jerk_penalty": "jerk_mult = 1.5 (safe) to 0.5 (emergency)",
            "recovery_hold": "Dampens control chatter after braking (6 steps)",
            "command_deadzone": "Suppresses micro-oscillations (0.01-0.02 m/s²)",
            "adaptive_smoothing": "alpha = 0.85 (safe) to 0.35 (emergency)",
            "context_reward_shaping": "Stress, fatigue, headway preference bonuses",
        },
        "notes": [
            "Uses adaptive context-dependent action smoothing",
            "Risk-aware jerk penalties scale with stress",
            "Recovery hold prevents control chatter",
            "Context-modulated reward shaping",
            "Expected to learn smoother, stress-aware behavior",
        ]
    }
    return config


def print_modifications_summary():
    """Print summary of all modifications."""
    print("\n" + "="*80)
    print("CONTEXT-AWARE CONTROL & REWARD MODIFICATIONS SUMMARY")
    print("="*80)
    
    modifications = [
        {
            "name": "Risk-Aware Jerk Penalty",
            "file": "env_car_following.py (lines 295-311)",
            "description": "Jerk penalty scales inversely with stress",
            "safe_driving": "1.5x multiplier (harsh)",
            "emergency": "0.5x multiplier (lenient)",
            "benefit": "Smooth in safe, aggressive when needed",
        },
        {
            "name": "Recovery Hold Mechanism",
            "file": "env_car_following.py (lines 791-806)",
            "description": "Dampens control after heavy braking",
            "duration": "6 steps",
            "effect": "Reduces upward accel slew by 50%",
            "benefit": "Prevents control chatter and oscillation",
        },
        {
            "name": "Command Deadzone",
            "file": "env_car_following.py (lines 815-820)",
            "low_risk": "0.02 m/s² threshold",
            "mid_risk": "0.01 m/s² threshold",
            "effect": "Suppresses micro-changes",
            "benefit": "Reduces unnecessary jerk, preserves responsiveness",
        },
        {
            "name": "Adaptive Action Smoothing",
            "file": "env_car_following.py (lines 825-835)",
            "vanilla": "alpha = 0.30 (fixed)",
            "low_risk": "alpha = 0.85 (heavy)",
            "emergency": "alpha = 0.35 (light)",
            "benefit": "Risk-sensitive filtering",
        },
        {
            "name": "Enhanced Jerk Diagnostics",
            "file": "env_car_following.py (lines 1017-1018)",
            "ema_abs_jerk": "EMA of |jerk| per step",
            "ema_pos_accel": "EMA of positive acceleration",
            "benefit": "Enables trajectory analysis and debugging",
        },
        {
            "name": "Headway Shortfall Penalty",
            "file": "env_car_following.py (lines 370-372)",
            "trigger": "When headway < (h_pref - deadband)",
            "gating": "Only when stress > 0.05",
            "benefit": "Prevents reward-hacking of short gaps",
        },
    ]
    
    for i, mod in enumerate(modifications, 1):
        print(f"\n{i}. {mod['name']}")
        print(f"   Location: {mod['file']}")
        print(f"   Description: {mod['description']}")
        for key, val in mod.items():
            if key not in ["name", "file", "description"]:
                print(f"   {key}: {val}")


def print_evaluation_metrics():
    """Print key metrics to track during evaluation."""
    print("\n" + "="*80)
    print("KEY EVALUATION METRICS")
    print("="*80)
    
    metrics = {
        "Safety": [
            "Collision rate (%)",
            "Min gap reached (m)",
            "TTC violations (<2s counts)",
            "Emergency braking frequency",
        ],
        "Comfort": [
            "Mean absolute jerk (m/s³)",
            "P95 jerk (m/s³)",
            "Mean absolute acceleration (m/s²)",
            "Action change frequency",
        ],
        "Efficiency": [
            "Mean speed (m/s)",
            "Speed target tracking error (m/s)",
            "Fuel economy proxy (smooth driving bonus)",
        ],
        "Control Quality": [
            "Command change smoothness (EMA ratio)",
            "Recovery hold activation rate",
            "Deadzone suppression rate",
            "Mean absolute jerk (EMA vs raw)",
        ],
        "Learning": [
            "Episode reward trend",
            "Context reward ratio (context/base)",
            "Safety shield engagement frequency",
            "Curriculum stage progression",
        ],
    }
    
    for category, metric_list in metrics.items():
        print(f"\n{category}:")
        for metric in metric_list:
            print(f"  - {metric}")


def print_training_steps():
    """Print recommended training steps."""
    print("\n" + "="*80)
    print("RECOMMENDED TRAINING STEPS")
    print("="*80)
    
    steps = [
        {
            "step": 1,
            "description": "Train Vanilla PPO Baseline",
            "command": "python train.py --agent vanilla_ppo --seed 0 --total_timesteps 300000",
            "duration": "~2-4 hours on GPU",
            "purpose": "Establish baseline behavior without context",
        },
        {
            "step": 2,
            "description": "Train Context-Aware PPO",
            "command": "python train.py --agent context_ppo --seed 0 --total_timesteps 300000",
            "duration": "~2-4 hours on GPU",
            "purpose": "Test context-aware control and reward",
        },
        {
            "step": 3,
            "description": "Evaluate Vanilla PPO",
            "command": "python evaluate.py --agent vanilla_ppo --seeds 0 --n_episodes 50 --render",
            "duration": "~30 minutes",
            "purpose": "Analyze vanilla behavior across scenarios",
        },
        {
            "step": 4,
            "description": "Evaluate Context-Aware PPO",
            "command": "python evaluate.py --agent context_ppo --seeds 0 --n_episodes 50 --render",
            "duration": "~30 minutes",
            "purpose": "Analyze context-aware behavior",
        },
        {
            "step": 5,
            "description": "Compare Trajectories",
            "command": "python compare_agents.py --agents vanilla_ppo context_ppo --scenario mixed",
            "duration": "~10 minutes",
            "purpose": "Side-by-side behavioral comparison",
        },
        {
            "step": 6,
            "description": "Generate Reports",
            "command": "python plots.py --compare vanilla_ppo context_ppo",
            "duration": "~5 minutes",
            "purpose": "Create comparison visualizations",
        },
    ]
    
    for step_info in steps:
        print(f"\nStep {step_info['step']}: {step_info['description']}")
        print(f"  Command:   {step_info['command']}")
        print(f"  Duration:  {step_info['duration']}")
        print(f"  Purpose:   {step_info['purpose']}")


def print_expected_outcomes():
    """Print expected outcomes from modifications."""
    print("\n" + "="*80)
    print("EXPECTED OUTCOMES")
    print("="*80)
    
    outcomes = {
        "Vanilla PPO (Baseline)": [
            "Smooth, predictable acceleration profiles",
            "Consistent behavior across scenarios",
            "Baseline jerk penalties (no context awareness)",
            "May show micro-oscillations in certain regimes",
        ],
        "Context-Aware PPO": [
            "Safe, smooth behavior in normal conditions",
            "More aggressive/responsive in emergency situations",
            "Stress-adaptive control (varies with context)",
            "Reduced micro-oscillations via deadzone/smoothing",
            "Better adaptation to driver state changes",
        ],
        "Improvement Areas": [
            "Jerk reduction in safe driving (+smoothness)",
            "Faster response in emergency scenarios (+safety)",
            "Reduced control chatter (+comfort)",
            "Better preference learning (+adaptation)",
        ],
    }
    
    for category, outcome_list in outcomes.items():
        print(f"\n{category}:")
        for outcome in outcome_list:
            print(f"  ✓ {outcome}")


def print_troubleshooting():
    """Print troubleshooting guide."""
    print("\n" + "="*80)
    print("TROUBLESHOOTING GUIDE")
    print("="*80)
    
    issues = [
        {
            "issue": "Training diverges or reward becomes NaN",
            "checks": [
                "Verify reward clipping is applied: total_reward_clip_min=-220, max=10",
                "Check learning rate not too high (1e-4 recommended)",
                "Ensure safety shield enabled during training",
                "Verify context stress level is in [0, 1] range",
            ],
        },
        {
            "issue": "Jerk penalty seems too strong/weak",
            "checks": [
                "Verify jerk_weight in config (default: 15.0)",
                "Check multiplier values (1.5 to 0.5 range)",
                "Monitor ema_abs_jerk in logs to see actual jerk levels",
                "Adjust comfort_guard_jerk_threshold if needed",
            ],
        },
        {
            "issue": "Recovery hold not activating",
            "checks": [
                "Verify recovery_hold_steps set (default: 6)",
                "Check context guard activation is working",
                "Monitor recovery_hold_frac in episode info",
                "Ensure action smoothing_enabled = True",
            ],
        },
        {
            "issue": "Context reward not modulating behavior",
            "checks": [
                "Verify context_reward_scale > 0 (default: varies)",
                "Check context model is providing valid stress levels",
                "Monitor context_stress in episode info",
                "Verify scenario triggers expected context changes",
            ],
        },
    ]
    
    for i, item in enumerate(issues, 1):
        print(f"\nIssue {i}: {item['issue']}")
        print("  Checks:")
        for check in item['checks']:
            print(f"    - {check}")


def save_configs_to_json():
    """Save training configs to JSON files."""
    configs = {
        "vanilla_ppo.json": print_training_config_vanilla(),
        "context_ppo.json": print_training_config_context(),
    }
    
    configs_dir = Path(__file__).parent / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    for filename, config in configs.items():
        filepath = configs_dir / filename
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Training and evaluation plan for context-aware car-following"
    )
    parser.add_argument(
        "--show",
        choices=["all", "configs", "modifications", "metrics", "steps", "outcomes", "troubleshooting"],
        default="all",
        help="What to display"
    )
    parser.add_argument(
        "--save-configs",
        action="store_true",
        help="Save training configs to JSON files"
    )
    
    args = parser.parse_args()
    
    if args.show in ["all", "configs"]:
        print("\n" + "="*80)
        print("VANILLA PPO TRAINING CONFIG")
        print("="*80)
        vanilla = print_training_config_vanilla()
        print(json.dumps(vanilla, indent=2))
        
        print("\n" + "="*80)
        print("CONTEXT-AWARE PPO TRAINING CONFIG")
        print("="*80)
        context = print_training_config_context()
        print(json.dumps(context, indent=2))
    
    if args.show in ["all", "modifications"]:
        print_modifications_summary()
    
    if args.show in ["all", "metrics"]:
        print_evaluation_metrics()
    
    if args.show in ["all", "steps"]:
        print_training_steps()
    
    if args.show in ["all", "outcomes"]:
        print_expected_outcomes()
    
    if args.show in ["all", "troubleshooting"]:
        print_troubleshooting()
    
    if args.save_configs:
        print("\n" + "="*80)
        print("SAVING CONFIGS")
        print("="*80)
        save_configs_to_json()


if __name__ == "__main__":
    main()
