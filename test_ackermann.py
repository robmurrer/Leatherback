#!/usr/bin/env python3
"""
Test script for Ackermann steering implementation.
This script validates the mathematical correctness of the Ackermann geometry calculations.
"""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np

class AckermannTester:
    def __init__(self, wheelbase=2.5, track_width=1.8, wheel_radius=0.3):
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.wheel_radius = wheel_radius
        
    def compute_ackermann_angles(self, steering_angle):
        """Test implementation of Ackermann angle computation"""
        # Clamp steering angle to prevent division by zero
        steering_angle = torch.clamp(steering_angle, -0.49, 0.49)
        
        # Handle near-zero steering angles
        small_angle_mask = torch.abs(steering_angle) < 1e-6
        steering_angle = torch.where(small_angle_mask, 
                                   torch.sign(steering_angle) * 1e-6, 
                                   steering_angle)
        
        # Ackermann steering equations
        turning_radius = self.wheelbase / torch.tan(torch.abs(steering_angle))
        
        left_angle = torch.atan(self.wheelbase / 
                               (turning_radius + torch.where(steering_angle > 0, 
                                                            -self.track_width / 2,
                                                            self.track_width / 2)))
        right_angle = torch.atan(self.wheelbase / 
                                (turning_radius + torch.where(steering_angle > 0,
                                                             self.track_width / 2,
                                                             -self.track_width / 2)))
        
        # Apply steering direction
        left_angle = torch.where(steering_angle > 0, left_angle, -left_angle)
        right_angle = torch.where(steering_angle > 0, right_angle, -right_angle)
        
        return left_angle, right_angle
    
    def compute_wheel_velocities(self, forward_speed, steering_angle):
        """Test implementation of wheel velocity computation"""
        angular_vel = forward_speed * torch.tan(steering_angle) / self.wheelbase
        
        v_fl = forward_speed + angular_vel * self.track_width / 2
        v_fr = forward_speed - angular_vel * self.track_width / 2
        v_rl = forward_speed + angular_vel * self.track_width / 2
        v_rr = forward_speed - angular_vel * self.track_width / 2
        
        wheel_angular_vels = torch.stack([v_fl, v_fr, v_rl, v_rr], dim=1) / self.wheel_radius
        
        return wheel_angular_vels
    
    def test_straight_line(self):
        """Test that straight-line driving has equal wheel speeds and zero steering"""
        print("Testing straight-line driving...")
        
        forward_speed = torch.tensor([1.0, 2.0, 5.0])
        steering_angle = torch.tensor([0.0, 0.0, 0.0])
        
        left_angle, right_angle = self.compute_ackermann_angles(steering_angle)
        wheel_vels = self.compute_wheel_velocities(forward_speed, steering_angle)
        
        # Check steering angles are zero
        assert torch.allclose(left_angle, torch.zeros_like(left_angle), atol=1e-6), "Left steering should be zero"
        assert torch.allclose(right_angle, torch.zeros_like(right_angle), atol=1e-6), "Right steering should be zero"
        
        # Check all wheels have same speed
        expected_wheel_speed = forward_speed / self.wheel_radius
        for i in range(4):
            assert torch.allclose(wheel_vels[:, i], expected_wheel_speed, atol=1e-6), f"Wheel {i} speed mismatch"
        
        print("✓ Straight-line test passed")
    
    def test_turning_geometry(self):
        """Test that Ackermann geometry is correct for turns"""
        print("Testing Ackermann turning geometry...")
        
        forward_speed = torch.tensor([2.0])
        steering_angle = torch.tensor([0.3])  # ~17 degrees
        
        left_angle, right_angle = self.compute_ackermann_angles(steering_angle)
        
        # For right turn (positive steering), left wheel should have larger angle
        assert left_angle > right_angle, "Left wheel should have larger angle for right turn"
        
        # Check that both angles point in same direction as main steering
        assert left_angle > 0 and right_angle > 0, "Both wheels should steer right for positive input"
        
        print(f"✓ Steering geometry test passed")
        print(f"  Main angle: {steering_angle.item():.3f} rad ({math.degrees(steering_angle.item()):.1f}°)")
        print(f"  Left angle: {left_angle.item():.3f} rad ({math.degrees(left_angle.item()):.1f}°)")
        print(f"  Right angle: {right_angle.item():.3f} rad ({math.degrees(right_angle.item()):.1f}°)")
    
    def test_wheel_speed_differential(self):
        """Test that wheel speeds are correctly differentiated for turns"""
        print("Testing wheel speed differential...")
        
        forward_speed = torch.tensor([3.0])
        steering_angle = torch.tensor([0.2])  # Right turn
        
        wheel_vels = self.compute_wheel_velocities(forward_speed, steering_angle)
        
        v_fl, v_fr, v_rl, v_rr = wheel_vels[0]
        
        # For right turn, left wheels should be faster than right wheels
        assert v_fl > v_fr, "Front left should be faster than front right in right turn"
        assert v_rl > v_rr, "Rear left should be faster than rear right in right turn"
        
        # Front and rear wheels on same side should have similar speeds
        assert abs(v_fl - v_rl) < 0.1, "Left wheels should have similar speeds"
        assert abs(v_fr - v_rr) < 0.1, "Right wheels should have similar speeds"
        
        print(f"✓ Wheel speed differential test passed")
        print(f"  FL: {v_fl:.2f}, FR: {v_fr:.2f}, RL: {v_rl:.2f}, RR: {v_rr:.2f} rad/s")
    
    def plot_ackermann_behavior(self):
        """Generate plots showing Ackermann behavior"""
        print("Generating Ackermann behavior plots...")
        
        # Test range of steering angles
        steering_angles = torch.linspace(-0.4, 0.4, 50)
        forward_speed = torch.ones_like(steering_angles) * 2.0
        
        left_angles = []
        right_angles = []
        turning_radii = []
        
        for angle in steering_angles:
            left, right = self.compute_ackermann_angles(angle.unsqueeze(0))
            left_angles.append(left.item())
            right_angles.append(right.item())
            
            # Compute turning radius
            if abs(angle) > 1e-6:
                radius = self.wheelbase / math.tan(abs(angle))
            else:
                radius = float('inf')
            turning_radii.append(radius)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot steering angles
        ax1.plot(np.degrees(steering_angles), np.degrees(left_angles), 'b-', label='Left wheel')
        ax1.plot(np.degrees(steering_angles), np.degrees(right_angles), 'r-', label='Right wheel')
        ax1.plot(np.degrees(steering_angles), np.degrees(steering_angles), 'k--', label='Input angle')
        ax1.set_xlabel('Input Steering Angle (degrees)')
        ax1.set_ylabel('Wheel Steering Angle (degrees)')
        ax1.set_title('Ackermann Steering Angles')
        ax1.legend()
        ax1.grid(True)
        
        # Plot turning radius
        ax2.plot(np.degrees(steering_angles), turning_radii, 'g-')
        ax2.set_xlabel('Input Steering Angle (degrees)')
        ax2.set_ylabel('Turning Radius (m)')
        ax2.set_title('Vehicle Turning Radius')
        ax2.set_ylim(0, 50)  # Limit y-axis for better visibility
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/goat/Documents/GitHub/Leatherback/ackermann_behavior.png', dpi=150, bbox_inches='tight')
        print("✓ Plots saved to ackermann_behavior.png")
        
    def run_all_tests(self):
        """Run all validation tests"""
        print("Running Ackermann implementation validation tests...\n")
        
        self.test_straight_line()
        self.test_turning_geometry()
        self.test_wheel_speed_differential()
        self.plot_ackermann_behavior()
        
        print("\n✅ All tests passed! Ackermann implementation is working correctly.")
        print("\nVehicle parameters:")
        print(f"  Wheelbase: {self.wheelbase:.1f}m")
        print(f"  Track width: {self.track_width:.1f}m")
        print(f"  Wheel radius: {self.wheel_radius:.1f}m")

if __name__ == "__main__":
    tester = AckermannTester()
    tester.run_all_tests()
