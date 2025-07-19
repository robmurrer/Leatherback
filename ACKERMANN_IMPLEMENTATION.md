# Ackermann Steering Implementation for Leatherback

## Overview

This document describes the implementation of Ackermann steering control for the Leatherback vehicle simulation, replacing the previous manual wheel control approach.

## Key Changes

### 1. Action Space Simplification
- **Before**: 6 DOF control (4 wheels + 2 steering joints)
- **After**: 2 DOF Ackermann control `[forward_speed, steering_angle]`

### 2. Realistic Vehicle Dynamics
- Proper Ackermann steering geometry
- Differential wheel speeds for turning
- Physically accurate vehicle behavior

### 3. Improved Joint Damping Configuration

#### Wheel Drive Joints
```python
"wheel_drive": ImplicitActuatorCfg(
    joint_names_expr=["Wheel.*"],
    effort_limit=400.0,      # Reduced from 40000.0 for realistic control
    velocity_limit=50.0,     # Reduced from 100.0 for stability
    stiffness=0.0,          # For velocity control
    damping=50.0,           # Reduced from 100000.0 for responsiveness
)
```

#### Steering Joints
```python
"steering": ImplicitActuatorCfg(
    joint_names_expr=["Knuckle__Upright__Front.*"],
    effort_limit=200.0,      # Reduced from 40000.0
    velocity_limit=10.0,     # Reduced from 100.0 for smooth steering
    stiffness=2000.0,       # Increased from 1000.0 for precision
    damping=100.0,          # Increased from 0.0 for stability
)
```

## Vehicle Parameters

### Physical Configuration
- **Wheelbase**: 2.5m (distance between front and rear axles)
- **Track Width**: 1.8m (distance between left and right wheels)
- **Wheel Radius**: 0.3m

### Action Scaling
- **Max Speed**: 5.0 m/s
- **Max Steering Angle**: 0.5 radians (~30 degrees)

## Ackermann Geometry Implementation

### Steering Angle Calculation
```python
def _compute_ackermann_angles(self, steering_angle):
    turning_radius = wheelbase / tan(abs(steering_angle))
    
    left_angle = atan(wheelbase / (turning_radius ± track_width/2))
    right_angle = atan(wheelbase / (turning_radius ∓ track_width/2))
```

### Wheel Velocity Calculation
```python
def _compute_wheel_velocities(self, forward_speed, steering_angle):
    angular_vel = forward_speed * tan(steering_angle) / wheelbase
    
    v_wheel = forward_speed ± angular_vel * track_width/2
    wheel_angular_vel = v_wheel / wheel_radius
```

## Benefits of Ackermann Implementation

### 1. **Simplified Training**
- 2D action space instead of 6D
- More intuitive control semantics
- Faster convergence expected

### 2. **Realistic Physics**
- Proper vehicle dynamics
- No tire scrubbing
- Correct turning behavior

### 3. **Better Stability**
- Improved damping prevents oscillations
- Reasonable force limits prevent instability
- Smooth steering response

### 4. **Enhanced Observations**
- Added current speed magnitude
- Direct action feedback in observations
- Better state representation

## Damping Configuration Rationale

### Why Damping Matters
1. **Prevents Oscillations**: High-frequency vibrations in joints
2. **Improves Stability**: Reduces numerical instabilities
3. **Realistic Behavior**: Mimics real vehicle damping systems
4. **Training Stability**: More consistent learning environment

### Steering Damping (100.0)
- **Previous**: 0.0 (caused instability)
- **Current**: 100.0 (provides smooth control)
- **Effect**: Eliminates steering oscillations while maintaining responsiveness

### Wheel Damping (50.0)
- **Previous**: 100000.0 (over-damped)
- **Current**: 50.0 (balanced response)
- **Effect**: Quick response to velocity commands without overshoot

## Migration Guide

### For Existing Models
1. **Action Space**: Models trained on 6D actions need retraining
2. **Observation Space**: Increased from 8 to 9 dimensions
3. **Scaling**: New action scaling factors

### Training Considerations
1. **Initial Policy**: May need different initialization
2. **Hyperparameters**: Exploration noise may need adjustment
3. **Reward Tuning**: Vehicle behavior changes may affect rewards

## Testing and Validation

### Instability Monitoring
- Enhanced CSV logging with Ackermann-specific metrics
- Real-time NaN detection and reporting
- Detailed vehicle state tracking

### Key Metrics to Monitor
- Steering angle stability
- Wheel velocity consistency
- Vehicle trajectory smoothness
- Training convergence rate

## Future Enhancements

### Potential Improvements
1. **Tire Model**: Add slip and friction modeling
2. **Suspension**: Include suspension dynamics
3. **Differential**: Model limited-slip differential
4. **Motor Curves**: RPM-dependent torque characteristics

### Advanced Features
1. **Adaptive Damping**: Dynamic damping based on speed
2. **Terrain Adaptation**: Damping adjustment for different surfaces
3. **Performance Modes**: Sport/comfort damping profiles

## Troubleshooting

### Common Issues

#### High Oscillation
- **Cause**: Insufficient damping
- **Solution**: Increase damping values gradually

#### Sluggish Response
- **Cause**: Over-damping
- **Solution**: Reduce damping or increase effort limits

#### Training Instability
- **Cause**: Action scaling mismatch
- **Solution**: Adjust max_speed and max_steering_angle

### Debug Tools
- CSV instability logging
- Real-time vehicle state monitoring
- Action-response visualization
