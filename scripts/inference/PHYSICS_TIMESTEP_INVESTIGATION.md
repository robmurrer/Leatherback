"""
🔬 Physics Timestep Investigation Report

Based on analysis of the Forest Gump simulation failures and codebase research,
this document summarizes findings about optimal physics timestep settings.

## Current Settings Analysis

### Training Environment (leatherback_env.py):
- physics_dt: 1/60 seconds (60Hz)
- decimation: 4
- **Effective control frequency: 60Hz / 4 = 15Hz**
- episode_length_s: 20.0

### Reference Cartpole Environment:
- physics_dt: 1/120 seconds (120Hz) 
- decimation: 2
- **Effective control frequency: 120Hz / 2 = 60Hz**

### Current Inference Settings:
- physics_dt: 1/60 seconds (60Hz)
- rendering_dt: 1/50 seconds (50Hz)
- **Control frequency: 60Hz (no decimation)**

## Problem Identification

### Root Cause: Control Frequency Mismatch
The training environment runs at **15Hz effective control frequency** due to decimation,
but inference runs at **60Hz**, creating a 4x frequency mismatch.

This means:
- **Training**: Model receives new observations every 4 physics steps (0.0667s intervals)
- **Inference**: Model receives new observations every physics step (0.0167s intervals)

### Physics Integration Issues
- **Too High Frequency**: Can cause numerical instability in physics solver
- **Too Low Frequency**: Can cause integration errors and control lag
- **Frequency Mismatch**: Trained model expects different temporal dynamics

## Recommended Solutions

### Immediate Fix: Match Training Frequency
```python
# Option 1: Reduce physics frequency to match effective training rate
PHYSICS_DT = 1/15  # 15Hz to match training decimation
RENDERING_DT = 1/30  # 30Hz for smooth visualization

# Option 2: Keep physics high but decimate control (RECOMMENDED)
PHYSICS_DT = 1/60   # 60Hz physics
CONTROL_DECIMATION = 4  # Apply same decimation as training
EFFECTIVE_CONTROL_HZ = 60 / 4  # 15Hz control updates
```

### Physics Timestep Testing Priority Order:
1. **1/15 Hz (0.0667s)** - Match training effective frequency
2. **1/60 Hz (0.0167s) with decimation=4** - Match training exactly  
3. **1/120 Hz (0.0083s)** - Match cartpole (known stable)
4. **1/240 Hz (0.0042s)** - Very high frequency for stability
5. **1/30 Hz (0.0333s)** - Lower frequency fallback

### Control Decimation Implementation:
```python
# In physics step callback:
if step_counter % CONTROL_DECIMATION == 0:
    # Only run ONNX inference every N steps
    obs = create_observations()
    actions = onnx_model.predict(obs)
    # Apply actions for next N physics steps

# For other steps, maintain previous actions
```

## Expected Improvements

### Stability Metrics:
- **Reduce position deviation**: < 1.0m throughout simulation
- **Eliminate joint velocity oscillations**: < 25 units steering
- **Consistent control response**: Predictable action magnitudes
- **Temporal alignment**: Match training time dynamics

### Testing Validation:
1. **Timing verification**: Measure actual vs expected step rates
2. **Action consistency**: Compare with training action ranges
3. **Trajectory tracking**: Monitor position error progression
4. **Joint stability**: Verify normal velocity ranges

## Implementation Notes

### Physics Solver Considerations:
- Isaac Lab uses PhysX solver with specific solver iteration settings
- Higher frequencies may require adjusted solver parameters
- Balance between stability and computational efficiency

### Model Adaptation:
- If changing frequency significantly, may need model retraining
- Temporal features in observations may need rescaling
- Action scaling might need adjustment for different frequencies

## Conclusion

The physics timestep mismatch is the **primary cause** of Forest Gump instabilities.
The 4x control frequency difference between training (15Hz) and inference (60Hz)
creates temporal dynamics the model was never trained to handle.

**Immediate Action**: Implement control decimation to match 15Hz training frequency.
**Medium-term**: Test various physics frequencies for optimal stability.
**Long-term**: Retrain model with robust temporal features.
"""
