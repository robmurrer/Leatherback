#!/usr/bin/env python3
"""
Quick test to check action and observation ranges in the Ackermann environment.
"""

import torch
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

def test_action_observation_ranges():
    print("Testing Ackermann Environment Action/Observation Ranges...")
    
    # Test our Ackermann implementation directly
    from test_ackermann import AckermannTester
    
    print("\n1. Testing Ackermann Math Implementation:")
    tester = AckermannTester()
    
    # Test with extreme actions that might be coming from the policy
    extreme_actions = torch.tensor([
        [1.0, 1.0],    # Max positive
        [-1.0, -1.0],  # Max negative  
        [1.0, -1.0],   # Mixed
        [0.0, 0.0],    # Zero
        [2.0, 2.0],    # Out of range (should be clipped)
    ])
    
    print("\nTesting extreme action values:")
    for i, action in enumerate(extreme_actions):
        forward_speed = action[0] * 5.0  # Our max_speed scaling
        steering_angle = action[1] * 0.5  # Our max_steering_angle scaling
        
        wheel_vels = tester.compute_wheel_velocities(forward_speed.unsqueeze(0), steering_angle.unsqueeze(0))
        left_angle, right_angle = tester.compute_ackermann_angles(steering_angle.unsqueeze(0))
        
        print(f"  Action {i}: [{action[0]:.1f}, {action[1]:.1f}]")
        print(f"    -> Speed: {forward_speed:.1f} m/s, Steering: {steering_angle:.3f} rad")
        print(f"    -> Wheel vels: {wheel_vels[0].tolist()}")
        print(f"    -> Steering angles: [{left_angle.item():.3f}, {right_angle.item():.3f}]")
        
        # Check for any extreme values
        if torch.any(torch.abs(wheel_vels) > 200):
            print(f"    ⚠️  WARNING: Extreme wheel velocities detected!")
        if torch.any(torch.abs(left_angle) > 1.0) or torch.any(torch.abs(right_angle) > 1.0):
            print(f"    ⚠️  WARNING: Extreme steering angles detected!")
        print()

def test_environment_config():
    print("\n2. Checking Environment Configuration:")
    
    try:
        # Try to import and check the environment config
        from Leatherback.tasks.direct.leatherback.leatherback_env import LeatherbackEnvCfg
        
        cfg = LeatherbackEnvCfg()
        print(f"  Action space: {cfg.action_space}")
        print(f"  Observation space: {cfg.observation_space}")
        print(f"  Max speed: {cfg.max_speed} m/s")
        print(f"  Max steering angle: {cfg.max_steering_angle} rad ({cfg.max_steering_angle * 57.3:.1f}°)")
        print(f"  Vehicle parameters:")
        print(f"    Wheelbase: {cfg.wheelbase}m")
        print(f"    Track width: {cfg.track_width}m") 
        print(f"    Wheel radius: {cfg.wheel_radius}m")
        
    except Exception as e:
        print(f"  Error importing environment config: {e}")

def test_action_scaling():
    print("\n3. Testing Action Scaling Logic:")
    
    # Simulate what happens in the environment
    actions = torch.tensor([
        [1.0, 1.0],
        [-1.0, -1.0],
        [0.5, -0.3],
        [10.0, 5.0],  # Extreme values that might come from untrained policy
    ])
    
    max_speed = 5.0
    max_steering_angle = 0.5
    
    print("  Input actions -> Scaled values:")
    for i, action in enumerate(actions):
        # Apply clamping like we do in the environment
        clamped_action = torch.clamp(action, -1.0, 1.0)
        
        forward_speed = clamped_action[0] * max_speed
        steering_angle = clamped_action[1] * max_steering_angle
        
        print(f"    [{action[0]:6.1f}, {action[1]:6.1f}] -> [{clamped_action[0]:4.1f}, {clamped_action[1]:4.1f}] -> [{forward_speed:5.1f} m/s, {steering_angle:5.3f} rad]")
        
        if torch.any(torch.abs(action) > 1.1):
            print(f"      ⚠️  Original action was out of range!")

if __name__ == "__main__":
    test_action_observation_ranges()
    test_environment_config()
    test_action_scaling()
    print("\n✅ Test completed!")
