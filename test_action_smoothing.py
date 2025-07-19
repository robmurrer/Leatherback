#!/usr/bin/env python3
"""
Test script to verify action smoothing effectiveness.
This script tests the action smoothing implementation and demonstrates oscillation reduction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def test_action_smoothing():
    """Test the action smoothing algorithm with oscillating inputs."""
    
    # Simulate the action smoothing parameters from the environment
    action_smoothing_factor = 0.3  # alpha for exponential moving average
    max_steering_rate = 0.5  # maximum change per timestep
    
    # Initialize state variables
    num_envs = 1
    device = torch.device('cpu')
    
    smoothed_actions = torch.zeros((num_envs, 2), device=device)  # [forward_speed, steering_angle]
    action_history = torch.zeros((num_envs, 5, 2), device=device)  # 5-step history
    
    # Create test data: alternating steering inputs (similar to what was observed in training)
    timesteps = 100
    raw_actions = []
    smoothed_results = []
    
    for t in range(timesteps):
        # Create oscillating steering input (like the problematic training behavior)
        if t % 2 == 0:
            steering = 1.0  # Full right
        else:
            steering = -1.0  # Full left
        
        forward_speed = 0.5  # Constant forward speed
        raw_action = torch.tensor([[forward_speed, steering]], device=device)
        
        # Apply action smoothing (replicated from environment)
        # Exponential moving average
        alpha = action_smoothing_factor
        smoothed_action = alpha * raw_action + (1 - alpha) * smoothed_actions
        
        # Rate limiting for steering angle (index 1)
        steering_diff = smoothed_action[:, 1] - smoothed_actions[:, 1]
        steering_diff = torch.clamp(steering_diff, -max_steering_rate, max_steering_rate)
        smoothed_action[:, 1] = smoothed_actions[:, 1] + steering_diff
        
        # Update history (rolling buffer)
        action_history[:, :-1] = action_history[:, 1:]
        action_history[:, -1] = smoothed_action
        
        # Apply additional smoothing based on recent history
        history_weight = 0.1
        recent_mean = torch.mean(action_history, dim=1)
        smoothed_action = (1 - history_weight) * smoothed_action + history_weight * recent_mean
        
        # Update state
        smoothed_actions = smoothed_action
        
        # Store results
        raw_actions.append([forward_speed, steering])
        smoothed_results.append(smoothed_action[0].tolist())
    
    # Convert to numpy for plotting
    raw_actions = np.array(raw_actions)
    smoothed_results = np.array(smoothed_results)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    timestep_array = np.arange(timesteps)
    
    # Plot steering angles
    ax1.plot(timestep_array, raw_actions[:, 1], 'r-', linewidth=2, label='Raw Steering (Oscillating)', alpha=0.7)
    ax1.plot(timestep_array, smoothed_results[:, 1], 'b-', linewidth=2, label='Smoothed Steering')
    ax1.set_ylabel('Steering Angle')
    ax1.set_title('Action Smoothing: Oscillation Reduction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.2, 1.2)
    
    # Plot forward speed (should be relatively stable)
    ax2.plot(timestep_array, raw_actions[:, 0], 'r-', linewidth=2, label='Raw Forward Speed', alpha=0.7)
    ax2.plot(timestep_array, smoothed_results[:, 0], 'b-', linewidth=2, label='Smoothed Forward Speed')
    ax2.set_ylabel('Forward Speed')
    ax2.set_xlabel('Timestep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/home/goat/Documents/GitHub/Leatherback/action_smoothing_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate oscillation metrics
    raw_steering_var = np.var(raw_actions[:, 1])
    smoothed_steering_var = np.var(smoothed_results[:, 1])
    oscillation_reduction = (raw_steering_var - smoothed_steering_var) / raw_steering_var * 100
    
    # Calculate maximum rate of change
    raw_steering_changes = np.abs(np.diff(raw_actions[:, 1]))
    smoothed_steering_changes = np.abs(np.diff(smoothed_results[:, 1]))
    max_raw_change = np.max(raw_steering_changes)
    max_smoothed_change = np.max(smoothed_steering_changes)
    
    print("🔧 Action Smoothing Test Results")
    print("=" * 50)
    print(f"Raw steering variance: {raw_steering_var:.4f}")
    print(f"Smoothed steering variance: {smoothed_steering_var:.4f}")
    print(f"Oscillation reduction: {oscillation_reduction:.1f}%")
    print()
    print(f"Max raw steering change: {max_raw_change:.4f}")
    print(f"Max smoothed steering change: {max_smoothed_change:.4f}")
    print(f"Rate limiting effectiveness: {(1 - max_smoothed_change/max_raw_change)*100:.1f}%")
    print()
    print("✅ Action smoothing test completed successfully!")
    print("📊 Plot saved to: action_smoothing_test.png")
    
    return smoothed_results

if __name__ == "__main__":
    test_action_smoothing()
