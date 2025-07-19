# 🧪 Design of Experiments: Observation Calculation Debug

## Objective
Fix the observation calculations in the Leatherback ONNX demo by systematically testing with simple scenarios.

## Current Problem
- Robot shows high angular velocity (6.58 rad/s) vs training (±1.8 rad/s)
- Large heading errors (66°) vs training (6-12°)  
- Negative forward velocity vs training positive velocity
- Saturated steering actions (0.75) vs training varied steering

## Experiment Design

### Experiment 1: Single Waypoint Directly Ahead ✅ IMPLEMENTED
**Setup:**
- Robot starts at (0, 0, 0.05) facing east (heading ≈ 0°)
- Single waypoint at (0, 3) - directly north
- Expected observations:
  - position_error = 3.0
  - heading_error = π/2 (90°) if robot faces east, target is north
  - cos(heading_error) = 0.0
  - sin(heading_error) = 1.0
  - velocities should start at 0.0

**What to check:**
1. Is robot_heading calculated correctly from quaternion?
2. Is target_heading calculated correctly from position difference?
3. Is heading_error calculation correct?
4. Are body-frame velocities transformed correctly?

### Experiment 2: Waypoint to the Right (if Exp 1 works)
**Setup:**
- Robot at (0, 0), waypoint at (3, 0) - directly east
- Expected: heading_error = 0° (robot already faces east)

### Experiment 3: Waypoint to the Left (if Exp 2 works)  
**Setup:**
- Robot at (0, 0), waypoint at (-3, 0) - directly west
- Expected: heading_error = π (180°)

### Experiment 4: Return to Complex Path (if all above work)

## Key Debug Points

### 1. Quaternion to Heading Conversion
Current code:
```python
robot_heading = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
```
- This gives Z-axis rotation (yaw)
- Need to verify this matches training environment

### 2. Body Frame Velocity Transformation
Current code:
```python
R_world_to_body = np.array([...])  # 3x3 rotation matrix
robot_lin_vel_body = R_world_to_body @ robot_lin_vel_world
```
- Need to verify rotation matrix is correct
- Check if training uses same transformation

### 3. Heading Error Calculation
Current code:
```python
target_heading = np.arctan2(position_error_vec[1], position_error_vec[0])
heading_error = np.arctan2(np.sin(target_heading - robot_heading), np.cos(target_heading - robot_heading))
```
- This should give angle from robot heading to target direction
- Range: [-π, π]

## Expected Training-like Observations
For a well-behaved case (robot moving toward nearby target):
- position_error: 2-6 meters
- cos(heading_error): 0.95-0.99 (small heading errors)
- sin(heading_error): -0.2 to 0.2 (small heading errors)  
- linear_vel_x: 0-2.5 m/s (forward motion)
- linear_vel_y: -0.3 to 0.3 m/s (minimal side motion)
- angular_vel_z: -2 to 2 rad/s (reasonable turning)

## Success Criteria
1. Heading calculations are consistent and reasonable
2. Velocities show forward motion toward target
3. Angular velocity is not extreme (< 3 rad/s)
4. Policy outputs reasonable actions (throttle ±25, steering ±7.5)
