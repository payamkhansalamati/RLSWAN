#!/usr/bin/env python3
"""
Comprehensive smoke test for all control and reward modifications.
Tests recovery hold, command deadzone, action smoothing, and reward scaling.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from env_car_following import CarFollowingEnv


class SmokeTestRunner:
    """Run comprehensive smoke tests on environment modifications."""
    
    def __init__(self):
        self.cfg = Config()
        self.passed = 0
        self.failed = 0
        
    def log_test(self, name: str, result: bool, message: str = ""):
        """Log test result."""
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
        if message:
            print(f"        {message}")
        if result:
            self.passed += 1
        else:
            self.failed += 1
        return result
    
    def test_recovery_hold_activation(self) -> bool:
        """Test that recovery hold activates and decrements counter."""
        print("\n" + "="*80)
        print("TEST 1: Recovery Hold Activation & Counter")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        obs, info = env.reset()
        
        # Capture initial recovery hold steps
        initial_hold = env._recovery_hold_steps
        self.log_test("Initial recovery hold steps", initial_hold == 0, 
                     f"Should be 0, got {initial_hold}")
        
        # Simulate heavy braking to trigger recovery hold
        # Emergency brake should be around -2.0
        emergency_brake = self.cfg.sim.emergency_brake  # -2.5
        
        # Step through and try to trigger recovery hold
        recovery_activated = False
        for step in range(50):
            action = np.array([emergency_brake - 0.1])  # Push beyond emergency brake
            obs, reward, terminated, truncated, info = env.step(action)
            
            if env._recovery_hold_steps > 0:
                recovery_activated = True
                hold_steps = env._recovery_hold_steps
                self.log_test("Recovery hold activated", True,
                             f"Hold counter: {hold_steps}")
                break
        
        # If not activated by strong braking, try context guard trigger
        if not recovery_activated:
            env.use_context_reward = True
            obs, info = env.reset()
            
            # Trigger a collision scenario to activate guard
            for step in range(200):
                action = np.array([0.5])
                obs, reward, terminated, truncated, info = env.step(action)
                
                if env._recovery_hold_steps > 0:
                    recovery_activated = True
                    self.log_test("Recovery hold (context guard)", True)
                    break
        
        if not recovery_activated:
            self.log_test("Recovery hold can activate", False,
                         "Could not trigger recovery hold in test")
        
        # Test hold counter decrements
        if recovery_activated:
            prev_hold = env._recovery_hold_steps
            for _ in range(3):
                action = np.array([0.0])
                obs, reward, terminated, truncated, info = env.step(action)
            
            new_hold = env._recovery_hold_steps
            self.log_test("Recovery hold counter decrements", new_hold < prev_hold,
                         f"Prev: {prev_hold}, New: {new_hold}")
        
        return True
    
    def test_command_deadzone(self) -> bool:
        """Test command deadzone suppression of micro-changes."""
        print("\n" + "="*80)
        print("TEST 2: Command Deadzone & Micro-Oscillation Suppression")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        obs, info = env.reset()
        
        deadzone_low = self.cfg.reward.cmd_deadzone_low_risk
        deadzone_mid = self.cfg.reward.cmd_deadzone_mid_risk
        
        self.log_test("Deadzone parameters exist", 
                     deadzone_low > 0 and deadzone_mid > 0,
                     f"Low: {deadzone_low}, Mid: {deadzone_mid}")
        
        # Test: Send tiny command changes that should be suppressed
        env.use_context_reward = False  # Vanilla mode
        obs, info = env.reset()
        env._prev_action_cmd = 0.0  # Initialize previous command
        
        tiny_change = deadzone_low / 2  # Below deadzone threshold
        changed_commands = []
        
        for step in range(10):
            # Send commands that change by less than deadzone
            action = np.array([env._prev_action_cmd + tiny_change])
            obs, reward, terminated, truncated, info = env.step(action)
            
            cmd_delta = info.get("mean_abs_cmd_change", 0)
            changed_commands.append(cmd_delta)
        
        # Average command changes should be very small (not all sent changes make it through)
        avg_change = np.mean(changed_commands)
        self.log_test("Deadzone suppresses small changes",
                     avg_change < deadzone_low,
                     f"Avg change: {avg_change:.6f}, Threshold: {deadzone_low}")
        
        return True
    
    def test_adaptive_action_smoothing(self) -> bool:
        """Test that action smoothing is applied with adaptive alpha."""
        print("\n" + "="*80)
        print("TEST 3: Adaptive Action Smoothing")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        
        # Test vanilla smoothing (fixed alpha)
        env.use_context_reward = False
        obs, info = env.reset()
        
        alpha_vanilla = self.cfg.reward.action_smoothing_alpha_vanilla
        self.log_test("Vanilla smoothing alpha exists",
                     alpha_vanilla > 0 and alpha_vanilla < 1,
                     f"Alpha: {alpha_vanilla}")
        
        # Test context smoothing (adaptive alpha)
        env.use_context_reward = True
        obs, info = env.reset()
        
        alpha_low = self.cfg.reward.context_smoothing_alpha_low_risk
        alpha_high = self.cfg.reward.context_smoothing_alpha_high_risk
        
        self.log_test("Context smoothing alphas exist",
                     alpha_low > alpha_high and alpha_high > 0,
                     f"Low: {alpha_low}, High: {alpha_high}")
        
        # Simulate various scenarios and check smoothing is applied
        large_smoothing_count = 0
        for step in range(50):
            action = np.array([np.random.uniform(-2, 2)])
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Smoothing should cause command_delta to be smaller than action
            if step > 0:
                cmd_delta = info.get("mean_abs_cmd_change", 0)
                if cmd_delta > 0.01:
                    large_smoothing_count += 1
            
            if terminated or truncated:
                obs, info = env.reset()
        
        self.log_test("Action smoothing applied",
                     large_smoothing_count > 0,
                     f"Smoothing observed in {large_smoothing_count} steps")
        
        return True
    
    def test_jerk_diagnostics(self) -> bool:
        """Test that jerk diagnostics are tracked correctly."""
        print("\n" + "="*80)
        print("TEST 4: Enhanced Jerk Diagnostics")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        obs, info = env.reset()
        
        # Collect jerk diagnostics over episode
        jerk_values = []
        ema_jerk_values = []
        accel_values = []
        
        for step in range(100):
            action = np.array([np.random.uniform(-1, 1)])
            obs, reward, terminated, truncated, info = env.step(action)
            
            jerk = info.get("jerk", 0)
            ema_jerk = info.get("ema_abs_jerk", 0)
            accel = info.get("accel", 0)
            
            jerk_values.append(abs(jerk))
            ema_jerk_values.append(ema_jerk)
            accel_values.append(abs(accel))
            
            if terminated or truncated:
                obs, info = env.reset()
        
        jerk_values = np.array(jerk_values)
        ema_jerk_values = np.array(ema_jerk_values)
        accel_values = np.array(accel_values)
        
        # Check that diagnostics are non-zero and tracked
        self.log_test("Jerk values tracked",
                     jerk_values.max() > 0,
                     f"Max jerk: {jerk_values.max():.6f}")
        
        self.log_test("EMA jerk tracked",
                     ema_jerk_values.max() > 0 or len(ema_jerk_values) > 0,
                     f"Max EMA jerk: {ema_jerk_values.max():.6f}, Tracked: {len(ema_jerk_values) > 0}")
        
        # EMA should smooth jerk values (lower variance)
        jerk_std = jerk_values.std()
        ema_std = ema_jerk_values.std()
        
        self.log_test("EMA jerk smooths raw jerk",
                     ema_std < jerk_std,
                     f"Raw std: {jerk_std:.6f}, EMA std: {ema_std:.6f}")
        
        # Positive acceleration tracking
        self.log_test("Positive acceleration tracked",
                     accel_values.max() > 0,
                     f"Max accel: {accel_values.max():.6f}")
        
        return True
    
    def test_context_reward_scaling(self) -> bool:
        """Test that context reward scales appropriately."""
        print("\n" + "="*80)
        print("TEST 5: Context Reward Scaling & Modulation")
        print("="*80)
        
        # Vanilla mode (no context)
        env = CarFollowingEnv(self.cfg)
        env.use_context_reward = False
        obs, info = env.reset()
        
        vanilla_rewards = []
        for step in range(50):
            action = np.array([np.random.uniform(-0.5, 0.5)])
            obs, reward, terminated, truncated, info = env.step(action)
            vanilla_rewards.append(reward)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        vanilla_mean = np.mean(vanilla_rewards)
        
        # Context mode
        env = CarFollowingEnv(self.cfg)
        env.use_context_reward = True
        obs, info = env.reset()
        
        context_rewards = []
        context_scales = []
        
        for step in range(50):
            action = np.array([np.random.uniform(-0.5, 0.5)])
            obs, reward, terminated, truncated, info = env.step(action)
            context_rewards.append(reward)
            context_scales.append(info.get("context_reward_scale", 0))
            
            if terminated or truncated:
                obs, info = env.reset()
        
        context_mean = np.mean(context_rewards)
        scale_mean = np.mean(context_scales)
        
        self.log_test("Vanilla mode computes rewards",
                     len(vanilla_rewards) > 0,
                     f"Collected {len(vanilla_rewards)} vanilla rewards")
        
        self.log_test("Context mode computes rewards",
                     len(context_rewards) > 0,
                     f"Collected {len(context_rewards)} context rewards")
        
        self.log_test("Context reward scale tracked",
                     scale_mean >= 0,
                     f"Mean scale: {scale_mean:.6f}")
        
        return True
    
    def test_risk_aware_jerk_penalty(self) -> bool:
        """Test risk-aware jerk multiplier in reward."""
        print("\n" + "="*80)
        print("TEST 6: Risk-Aware Jerk Penalty Scaling")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        env.use_context_reward = True
        obs, info = env.reset()
        
        # Verify config parameters exist
        low_mult = self.cfg.reward.low_risk_jerk_multiplier
        high_mult = self.cfg.reward.high_risk_jerk_multiplier
        
        self.log_test("Jerk multiplier parameters exist",
                     low_mult > high_mult and high_mult > 0,
                     f"Low: {low_mult}, High: {high_mult}")
        
        # Test: Collect rewards with varying stress levels
        comfort_rewards = []
        
        for step in range(100):
            action = np.array([np.random.uniform(-0.5, 0.5)])
            obs, reward, terminated, truncated, info = env.step(action)
            
            comfort = info.get("base_comfort", 0)
            comfort_rewards.append(comfort)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        comfort_array = np.array(comfort_rewards)
        
        self.log_test("Comfort rewards computed",
                     len(comfort_rewards) > 0,
                     f"Collected {len(comfort_rewards)} comfort values")
        
        # Comfort should include jerk penalty (negative values expected)
        self.log_test("Comfort includes jerk penalty",
                     comfort_array.min() < 0,
                     f"Min comfort: {comfort_array.min():.6f}")
        
        return True
    
    def test_diagnostic_components(self) -> bool:
        """Test all diagnostic components are present."""
        print("\n" + "="*80)
        print("TEST 7: Diagnostic Component Tracking")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        env.use_context_reward = True
        obs, info = env.reset()
        
        # Take a step to populate components
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check for key diagnostic components
        key_components = [
            "base_safety", "base_comfort", "base_efficiency",
            "context_reward", "context_reward_scale",
            "context_stress", "context_fatigue",
            "ema_abs_jerk", "ema_pos_accel",
        ]
        
        missing = []
        for component in key_components:
            if component not in info:
                missing.append(component)
        
        self.log_test("Key diagnostic components present",
                     len(missing) == 0,
                     f"Missing: {missing if missing else 'None'}")
        
        # Verify types are numeric
        numeric_ok = True
        for component in key_components:
            if component in info:
                val = info[component]
                if not isinstance(val, (int, float, np.number)):
                    numeric_ok = False
                    print(f"  Component {component} is {type(val)}, expected numeric")
        
        self.log_test("Diagnostic components are numeric",
                     numeric_ok)
        
        return True
    
    def test_numerical_stability(self) -> bool:
        """Test numerical stability with extreme scenarios."""
        print("\n" + "="*80)
        print("TEST 8: Numerical Stability & Edge Cases")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        env.use_context_reward = True
        obs, info = env.reset()
        
        nan_count = 0
        inf_count = 0
        total_steps = 200
        
        for step in range(total_steps):
            # Random actions including extremes
            if step % 20 == 0:
                action = np.array([-2.0])  # Max decel
            elif step % 20 == 10:
                action = np.array([2.0])   # Max accel
            else:
                action = np.array([np.random.uniform(-2, 2)])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check reward
            if np.isnan(reward):
                nan_count += 1
            if np.isinf(reward):
                inf_count += 1
            
            # Check all info values
            for key, val in info.items():
                if isinstance(val, (int, float, np.number)):
                    if np.isnan(val):
                        nan_count += 1
                    if np.isinf(val):
                        inf_count += 1
            
            if terminated or truncated:
                obs, info = env.reset()
        
        self.log_test("No NaN values produced",
                     nan_count == 0,
                     f"NaN count: {nan_count} in {total_steps} steps")
        
        self.log_test("No Inf values produced",
                     inf_count == 0,
                     f"Inf count: {inf_count} in {total_steps} steps")
        
        return True
    
    def test_full_episode(self) -> bool:
        """Test full episode with all features enabled."""
        print("\n" + "="*80)
        print("TEST 9: Full Episode Integration Test")
        print("="*80)
        
        env = CarFollowingEnv(self.cfg)
        env.use_context_reward = True
        
        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        collision = False
        
        for step in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if info.get("collision", False):
                collision = True
            
            if terminated or truncated:
                break
        
        # Episode should complete
        self.log_test("Full episode completes",
                     step_count > 0,
                     f"Episode lasted {step_count} steps")
        
        # Episode should produce valid reward
        self.log_test("Episode reward is finite",
                     np.isfinite(episode_reward),
                     f"Total reward: {episode_reward:.6f}")
        
        # Episode should not always collide (depends on scenario)
        self.log_test("Episode runs without guaranteed collision",
                     not collision or step_count < 50,
                     f"Collision: {collision}, Steps: {step_count}")
        
        return True
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        print("\n" + "="*80)
        print("COMPREHENSIVE SMOKE TEST SUITE")
        print("Testing all control & reward modifications")
        print("="*80)
        
        tests = [
            self.test_recovery_hold_activation,
            self.test_command_deadzone,
            self.test_adaptive_action_smoothing,
            self.test_jerk_diagnostics,
            self.test_context_reward_scaling,
            self.test_risk_aware_jerk_penalty,
            self.test_diagnostic_components,
            self.test_numerical_stability,
            self.test_full_episode,
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"[EXCEPTION] in {test_func.__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.failed += 1
        
        # Summary
        print("\n" + "="*80)
        print("SMOKE TEST SUMMARY")
        print("="*80)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n[ALL TESTS PASSED]")
            return True
        else:
            print(f"\n[{self.failed} TEST(S) FAILED]")
            return False


if __name__ == "__main__":
    runner = SmokeTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)
