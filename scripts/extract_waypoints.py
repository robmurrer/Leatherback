#!/usr/bin/env python3
"""
Standalone script to extract and display waypoint coordinates from the Leatherback environment.
This can be used to get the target positions without running a full simulation.
"""

import torch
import numpy as np

def generate_waypoints(num_goals=10, env_spacing=32.0, course_length_coefficient=2.5):
    """
    Generate waypoints using the same logic as the LeatherbackEnv.
    
    Args:
        num_goals: Number of waypoints to generate
        env_spacing: Environment spacing parameter
        course_length_coefficient: Course length scaling factor
        
    Returns:
        numpy array of waypoint positions [x, y]
    """
    device = "cpu"  # Use CPU for this utility
    
    # Replicate the waypoint generation logic from LeatherbackEnv._reset_idx
    spacing = 2 / num_goals
    target_positions = torch.arange(-0.8, 1.1, spacing, device=device) * env_spacing / course_length_coefficient
    
    # Create 2D positions (y-coordinates are 0 in the current implementation)
    waypoints = torch.zeros((num_goals, 2), device=device)
    waypoints[:len(target_positions), 0] = target_positions
    waypoints[:, 1] = 0  # Y-coordinates are currently set to 0
    
    return waypoints.numpy()

def print_waypoints_for_copy_paste(waypoints, env_spacing=32.0, course_length_coefficient=2.5, num_goals=10):
    """Print waypoints in various copy-pasteable formats."""
    
    print("\n" + "="*80)
    print("LEATHERBACK WAYPOINTS - Copy-pasteable formats:")
    print("="*80)
    
    # Format as Python list
    print("# Target positions as Python list:")
    target_list = waypoints.tolist()
    print(f"target_positions = {target_list}")
    
    print("\n# Target positions as NumPy array:")
    print(f"import numpy as np")
    print(f"target_positions = np.array({target_list})")
    
    print("\n# Individual waypoints:")
    for i, pos in enumerate(waypoints):
        print(f"waypoint_{i} = [{pos[0]:.6f}, {pos[1]:.6f}]")
    
    print("\n# X and Y coordinates separately:")
    x_coords = [pos[0] for pos in waypoints]
    y_coords = [pos[1] for pos in waypoints]
    print(f"x_coordinates = {x_coords}")
    print(f"y_coordinates = {y_coords}")
    
    print("\n# Environment parameters used:")
    print(f"env_spacing = {env_spacing}")
    print(f"course_length_coefficient = {course_length_coefficient}")
    print(f"num_goals = {num_goals}")
    
    print("\n# Matplotlib plotting code:")
    print("import matplotlib.pyplot as plt")
    print("import numpy as np")
    print(f"waypoints = np.array({target_list})")
    print("plt.figure(figsize=(12, 8))")
    print("plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', markersize=8, linewidth=2)")
    print("plt.xlabel('X Position')")
    print("plt.ylabel('Y Position')")
    print("plt.title('Leatherback Waypoints')")
    print("plt.grid(True)")
    print("plt.axis('equal')")
    print("for i, (x, y) in enumerate(waypoints):")
    print("    plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')")
    print("plt.show()")
    
    print("\n# ROS2 geometry_msgs/PoseStamped format (if needed):")
    print("waypoint_poses = [")
    for i, pos in enumerate(waypoints):
        print(f"    # Waypoint {i}")
        print("    {")
        print("        'position': {'x': " + f"{pos[0]:.6f}, 'y': {pos[1]:.6f}, 'z': 0.0" + "},")
        print("        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}")
        print("    }," if i < len(waypoints) - 1 else "    }")
    print("]")
    
    print("="*80)

def main():
    """Main function to generate and display waypoints."""
    
    # Default parameters from LeatherbackEnv
    num_goals = 10
    env_spacing = 32.0
    course_length_coefficient = 2.5
    
    print("Generating Leatherback waypoints with default parameters...")
    print(f"  - Number of goals: {num_goals}")
    print(f"  - Environment spacing: {env_spacing}")
    print(f"  - Course length coefficient: {course_length_coefficient}")
    
    # Generate waypoints
    waypoints = generate_waypoints(num_goals, env_spacing, course_length_coefficient)
    
    # Display in copy-pasteable formats
    print_waypoints_for_copy_paste(waypoints, env_spacing, course_length_coefficient, num_goals)
    
    # Optional: Save to file
    output_file = "leatherback_waypoints.txt"
    with open(output_file, 'w') as f:
        f.write("# Leatherback Waypoints\n")
        f.write(f"# Generated with: num_goals={num_goals}, env_spacing={env_spacing}, course_length_coefficient={course_length_coefficient}\n\n")
        f.write(f"target_positions = {waypoints.tolist()}\n")
    
    print(f"\nWaypoints also saved to: {output_file}")

if __name__ == "__main__":
    main()
