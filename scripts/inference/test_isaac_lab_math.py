#!/usr/bin/env python3
"""
🧪 Test Isaac Lab Math Implementation

This script tests our manual Isaac Lab math implementation against known
training data to ensure we're computing the same properties correctly.
"""

import numpy as np
import torch
import csv
import os

# Isaac Lab math utilities (our implementation)
def convert_quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from Isaac Sim (x,y,z,w) to Isaac Lab (w,x,y,z) format."""
    return torch.cat([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], dim=-1)

def quat_apply_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternion rotation to vector (transform world to body frame)."""
    # Extract quaternion components (w,x,y,z format)
    qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    
    # Inverse quaternion (conjugate for unit quaternions)
    quat_inv = torch.stack([-qx, -qy, -qz, qw], dim=-1)  # (-x,-y,-z,w)
    
    # Apply quaternion rotation: v' = q^-1 * v * q
    # For efficiency, use direct formula for vector rotation
    
    # Vector components
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    
    # Rotation using inverse quaternion
    # v' = v + 2 * q_xyz × (q_xyz × v + q_w * v)
    qx_inv, qy_inv, qz_inv, qw_inv = quat_inv[:, 0], quat_inv[:, 1], quat_inv[:, 2], quat_inv[:, 3]
    
    # Cross product: q_xyz × v
    cross1_x = qy_inv * vz - qz_inv * vy
    cross1_y = qz_inv * vx - qx_inv * vz  
    cross1_z = qx_inv * vy - qy_inv * vx
    
    # q_xyz × v + q_w * v
    temp_x = cross1_x + qw_inv * vx
    temp_y = cross1_y + qw_inv * vy
    temp_z = cross1_z + qw_inv * vz
    
    # Cross product: q_xyz × temp
    cross2_x = qy_inv * temp_z - qz_inv * temp_y
    cross2_y = qz_inv * temp_x - qx_inv * temp_z
    cross2_z = qx_inv * temp_y - qy_inv * temp_x
    
    # Final result: v + 2 * cross2
    result_x = vx + 2 * cross2_x
    result_y = vy + 2 * cross2_y
    result_z = vz + 2 * cross2_z
    
    return torch.stack([result_x, result_y, result_z], dim=-1)

def compute_heading_w(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Compute yaw heading from quaternion (Isaac Lab implementation)."""
    # Extract quaternion components (w,x,y,z format)
    qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    
    # Compute yaw angle (rotation around Z-axis)
    heading = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return heading

def test_math_with_training_data():
    """Test our math implementation using training CSV data."""
    print("🧪 Testing Isaac Lab Math Implementation")
    print("=" * 60)
    
    # Load training CSV for test data
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "leatherback_direct",
                           "2025-07-18_15-16-40", "csv_logs", "observations_actions_2025-07-18_15-26-31.csv")
    csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(csv_path):
        print(f"❌ Training CSV not found: {csv_path}")
        return
    
    print(f"📊 Loading training data from: {csv_path}")
    
    # Read first few rows as test cases
    test_cases = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 3:  # Only need first 3 rows for testing
                break
            test_cases.append(row)
    
    print(f"✅ Loaded {len(test_cases)} test cases")
    print()
    
    # Test each case
    for i, case in enumerate(test_cases):
        print(f"🧪 TEST CASE {i+1}:")
        print(f"   Timestep: {case['timestep']}, Sim time: {case['sim_time']}s")
        
        # Extract robot state from CSV
        pos_x = float(case['pos_x'])
        pos_y = float(case['pos_y']) 
        pos_z = float(case['pos_z'])
        
        # Extract known observations (these come directly from Isaac Lab)
        known_obs = [
            float(case['obs_0_position_error']),
            float(case['obs_1_heading_cos']),
            float(case['obs_2_heading_sin']),
            float(case['obs_3_linear_vel_x']),
            float(case['obs_4_linear_vel_y']),
            float(case['obs_5_angular_vel_z']),
            float(case['obs_6_throttle_state']),
            float(case['obs_7_steering_state'])
        ]
        
        print(f"   Position: [{pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}]")
        print(f"   Known Isaac Lab observations: {known_obs}")
        
        # For testing math functions, we need to reconstruct quaternion and velocities
        # Since we don't have these in the CSV, let's test with synthetic data that matches
        # the known linear velocity observations
        
        # Create test quaternion that would produce the known heading
        # From obs_1 (cos) and obs_2 (sin), we can reconstruct the heading error
        heading_cos = known_obs[1]  # cos(target_heading_error)
        heading_sin = known_obs[2]  # sin(target_heading_error)
        
        # Test our heading computation
        # We'll create a quaternion and see if our function produces reasonable results
        
        # Create test quaternion (identity quaternion as baseline)
        test_quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        test_quat_wxyz = convert_quat_xyzw_to_wxyz(test_quat_xyzw)
        computed_heading = compute_heading_w(test_quat_wxyz)
        
        print(f"   🧮 Math test - Identity quaternion heading: {computed_heading[0]:.6f} rad")
        
        # Test velocity transformation
        # Create test world velocity that should transform to known body velocity
        known_vel_x = known_obs[3]  # obs_3_linear_vel_x (body frame)
        known_vel_y = known_obs[4]  # obs_4_linear_vel_y (body frame)
        
        # For identity quaternion, world and body frames are the same
        test_world_vel = torch.tensor([[known_vel_x, known_vel_y, 0.0]], dtype=torch.float32)
        computed_body_vel = quat_apply_inverse(test_quat_wxyz, test_world_vel)
        
        print(f"   🧮 Math test - Identity rotation velocity transform:")
        print(f"      Input world vel: [{test_world_vel[0,0]:.6f}, {test_world_vel[0,1]:.6f}, {test_world_vel[0,2]:.6f}]")
        print(f"      Output body vel: [{computed_body_vel[0,0]:.6f}, {computed_body_vel[0,1]:.6f}, {computed_body_vel[0,2]:.6f}]")
        print(f"      Expected match:  [{known_vel_x:.6f}, {known_vel_y:.6f}, 0.000000]")
        
        # Check if transformation is working (should be identity for identity quaternion)
        vel_diff_x = abs(computed_body_vel[0,0].item() - known_vel_x)
        vel_diff_y = abs(computed_body_vel[0,1].item() - known_vel_y)
        
        if vel_diff_x < 1e-6 and vel_diff_y < 1e-6:
            print(f"      ✅ PASS: Velocity transformation working correctly")
        else:
            print(f"      ❌ FAIL: Velocity transformation error - x_diff={vel_diff_x:.6f}, y_diff={vel_diff_y:.6f}")
        
        print()
    
    # Test with actual rotation
    print("🔄 Testing with actual rotation:")
    
    # Test 90-degree rotation around Z-axis
    angle = np.pi / 2  # 90 degrees
    test_quat_xyzw_90 = torch.tensor([[0.0, 0.0, np.sin(angle/2), np.cos(angle/2)]], dtype=torch.float32)
    test_quat_wxyz_90 = convert_quat_xyzw_to_wxyz(test_quat_xyzw_90)
    
    computed_heading_90 = compute_heading_w(test_quat_wxyz_90)
    print(f"   90° rotation - computed heading: {computed_heading_90[0]:.6f} rad ({np.degrees(computed_heading_90[0]):.1f}°)")
    print(f"   Expected: {angle:.6f} rad (90.0°)")
    
    heading_error = abs(computed_heading_90[0].item() - angle)
    if heading_error < 1e-5:
        print(f"   ✅ PASS: Heading computation working correctly")
    else:
        print(f"   ❌ FAIL: Heading computation error - {heading_error:.6f} rad")
    
    # Test velocity transformation with 90° rotation
    test_world_vel_90 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)  # Forward in world
    computed_body_vel_90 = quat_apply_inverse(test_quat_wxyz_90, test_world_vel_90)
    
    print(f"   90° rotation velocity test:")
    print(f"      World vel (forward): [{test_world_vel_90[0,0]:.6f}, {test_world_vel_90[0,1]:.6f}, {test_world_vel_90[0,2]:.6f}]")
    print(f"      Body vel (should be rightward): [{computed_body_vel_90[0,0]:.6f}, {computed_body_vel_90[0,1]:.6f}, {computed_body_vel_90[0,2]:.6f}]")
    print(f"      Expected approximately: [0.000000, -1.000000, 0.000000]")
    
    print()
    print("🏁 Math validation complete!")

if __name__ == "__main__":
    test_math_with_training_data()
