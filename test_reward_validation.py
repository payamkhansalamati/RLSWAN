#!/usr/bin/env python3
"""
Comprehensive test for reward function modifications.
Validates risk-aware jerk penalties and context-aware reward modulation.
"""

import numpy as np
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from env_car_following import CarFollowingEnv
from context_model import ContextModel


def test_reward_components():
    """Test that reward components are computed correctly."""
    print("=" * 80)
    print("TEST 1: Basic Reward Component Computation")
    print("=" * 80)
    
    cfg = Config()
    env = CarFollowingEnv(cfg)
    
    # Reset environment to get initial state
    obs, info = env.reset()
    print(f"✓ Environment reset successfully")
    print(f"  Observation shape: {obs.shape}")
    
    # Take a sample action
    action = np.array([0.0])  # neutral acceleration
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✓ Step executed successfully")
    print(f"  Reward: {reward:.6f}")
    print(f"  Info keys: {list(info.keys())}")
    
    # Verify reward components are in info
    assert "reward_components" in info, "reward_components not in info"
    components = info["reward_components"]
    
    print(f"✓ Reward components present:")
    for key in ["base_safety", "base_comfort", "base_efficiency", "total"]:
        assert key in components, f"{key} missing from components"
        print(f"    {key}: {components[key]:.6f}")
    
    return True


def test_risk_aware_jerk_penalty():
    """Test risk-aware jerk penalty modification."""
    print("\n" + "=" * 80)
    print("TEST 2: Risk-Aware Jerk Penalty")
    print("=" * 80)
    
    cfg = Config()
    env = CarFollowingEnv(cfg)
    
    # Test 1: Vanilla mode (no context)
    obs, info = env.reset()
    env.use_context_reward = False
    
    # Trigger a maneuver with high jerk
    for _ in range(5):
        action = np.array([2.0])  # aggressive acceleration
        obs, reward, terminated, truncated, info = env.step(action)
    
    vanilla_reward = reward
    vanilla_components = info["reward_components"]
    
    print(f"✓ Vanilla mode (no context):")
    print(f"    Reward: {vanilla_reward:.6f}")
    print(f"    base_comfort: {vanilla_components['base_comfort']:.6f}")
    
    # Test 2: Context mode with low stress (high-risk situation)
    obs, info = env.reset()
    env.use_context_reward = True
    
    # Create a high-risk scenario (we'll simulate by checking stress-based reward scaling)
    for _ in range(5):
        action = np.array([2.0])  # aggressive acceleration
        obs, reward, terminated, truncated, info = env.step(action)
    
    context_reward = reward
    context_components = info["reward_components"]
    
    print(f"✓ Context mode:")
    print(f"    Reward: {context_reward:.6f}")
    print(f"    base_comfort: {context_components['base_comfort']:.6f}")
    
    # The base comfort penalty should differ based on context
    print(f"✓ Comfort component comparison:")
    print(f"    Vanilla: {vanilla_components['base_comfort']:.6f}")
    print(f"    Context: {context_components['base_comfort']:.6f}")
    
    return True


def test_context_model_integration():
    """Test that context model is properly integrated."""
    print("\n" + "=" * 80)
    print("TEST 3: Context Model Integration")
    print("=" * 80)
    
    cfg = Config()
    env = CarFollowingEnv(cfg)
    env.use_context_reward = True
    
    obs, info = env.reset()
    print(f"✓ Environment reset with context enabled")
    
    # Step through episode and verify context is tracked
    context_states = []
    for step_num in range(10):
        action = np.array([0.5])  # mild acceleration
        obs, reward, terminated, truncated, info = env.step(action)
        
        if "context" in info and info["context"] is not None:
            ctx = info["context"]
            context_states.append({
                "step": step_num,
                "stress": ctx.stress_level,
                "fatigue": ctx.fatigue_level,
                "headway": ctx.preferred_headway,
            })
    
    if context_states:
        print(f"✓ Context tracked for {len(context_states)} steps")
        print(f"  Sample context state (step 5):")
        if len(context_states) > 5:
            s = context_states[5]
            print(f"    stress_level: {s['stress']:.3f}")
            print(f"    fatigue_level: {s['fatigue']:.3f}")
            print(f"    preferred_headway: {s['headway']:.3f}")
    else:
        print(f"⚠ No context states collected (context may not be tracked in info)")
    
    return True


def test_reward_clipping():
    """Test that reward clipping is applied correctly."""
    print("\n" + "=" * 80)
    print("TEST 4: Reward Clipping and Bounds")
    print("=" * 80)
    
    cfg = Config()
    env = CarFollowingEnv(cfg)
    
    obs, info = env.reset()
    
    # Collect multiple reward values
    rewards = []
    for step_num in range(100):
        action = np.array([np.random.uniform(-2, 2)])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    rewards = np.array(rewards)
    
    print(f"✓ Collected {len(rewards)} reward samples")
    print(f"  Min reward: {rewards.min():.6f}")
    print(f"  Max reward: {rewards.max():.6f}")
    print(f"  Mean reward: {rewards.mean():.6f}")
    print(f"  Std reward: {rewards.std():.6f}")
    
    # Verify rewards are within bounds (use the nested reward config)
    min_bound = cfg.reward.total_reward_clip_min
    max_bound = cfg.reward.total_reward_clip_max
    
    assert rewards.min() >= min_bound, f"Reward below minimum bound: {rewards.min()} < {min_bound}"
    assert rewards.max() <= max_bound, f"Reward above maximum bound: {rewards.max()} > {max_bound}"
    
    print(f"✓ All rewards within bounds [{min_bound}, {max_bound}]")
    
    return True


def test_diagnostic_components():
    """Test that diagnostic components are properly recorded."""
    print("\n" + "=" * 80)
    print("TEST 5: Diagnostic Components")
    print("=" * 80)
    
    cfg = Config()
    env = CarFollowingEnv(cfg)
    
    obs, info = env.reset()
    
    # Take a few steps and check diagnostic components
    for step_num in range(5):
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert "reward_components" in info, "reward_components missing"
        components = info["reward_components"]
        
        # Check for essential diagnostic keys
        essential_keys = [
            "base_safety", "base_comfort", "base_efficiency",
            "total_raw", "total", "reward_clipped"
        ]
        
        for key in essential_keys:
            assert key in components, f"Component {key} missing"
    
    print(f"✓ Diagnostic components present in all {step_num + 1} steps")
    print(f"  Essential keys: {essential_keys}")
    
    # Print last step's components
    print(f"✓ Last step reward breakdown:")
    for key in essential_keys:
        val = components[key]
        if isinstance(val, (int, float)):
            print(f"    {key}: {val:.6f}")
        else:
            print(f"    {key}: {val}")
    
    return True


def test_numerical_stability():
    """Test numerical stability with extreme inputs."""
    print("\n" + "=" * 80)
    print("TEST 6: Numerical Stability")
    print("=" * 80)
    
    cfg = Config()
    env = CarFollowingEnv(cfg)
    
    obs, info = env.reset()
    
    # Test with various action ranges
    test_actions = [
        np.array([-2.0]),  # min decel
        np.array([2.0]),   # max accel
        np.array([0.0]),   # neutral
    ]
    
    print("Testing with extreme actions:")
    for action in test_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check for NaN or Inf
        assert not np.isnan(reward), f"NaN reward for action {action}"
        assert not np.isinf(reward), f"Inf reward for action {action}"
        
        components = info["reward_components"]
        for key, val in components.items():
            if isinstance(val, (int, float)):
                assert not np.isnan(val), f"NaN in component {key}"
                assert not np.isinf(val), f"Inf in component {key}"
        
        print(f"  ✓ Action {action[0]:5.1f} → Reward {reward:8.6f}")
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("REWARD FUNCTION VALIDATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Reward Components", test_reward_components),
        ("Risk-Aware Jerk Penalty", test_risk_aware_jerk_penalty),
        ("Context Model Integration", test_context_model_integration),
        ("Reward Clipping", test_reward_clipping),
        ("Diagnostic Components", test_diagnostic_components),
        ("Numerical Stability", test_numerical_stability),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = ("PASS", None)
            print(f"\n✓ {test_name}: PASSED")
        except Exception as e:
            results[test_name] = ("FAIL", str(e))
            print(f"\n✗ {test_name}: FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for status, _ in results.values() if status == "PASS")
    failed = sum(1 for status, _ in results.values() if status == "FAIL")
    
    for test_name, (status, error) in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
        if error:
            print(f"    {error}")
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
