# Forest Gump Demo Collection

This directory contains multiple demo scripts to address the Forest Gump simulation instability issues.

## 🔧 The Critical Fix: Control Frequency Mismatch

**Problem Identified**: The training environment uses `decimation=4` (effective 15Hz control frequency), but inference was running at 60Hz control frequency - a **4x mismatch** causing instabilities.

**Solution**: Implement proper control decimation to match training frequency.

## 📁 Demo Scripts

### 1. `forest_gump_minimal_demo.py` - 🧪 Minimal Test
- **Purpose**: Test control decimation timing without requiring trained models
- **Features**: Uses static dummy actions, validates timing accuracy
- **Best for**: Quick validation that the timing fix works
- **Dependencies**: Only Isaac Sim (no ONNX model needed)

### 2. `forest_gump_fixed_demo.py` - 🎯 Full Fixed Demo  
- **Purpose**: Complete demo with ONNX inference and proper control decimation
- **Features**: Real ONNX policy, dummy observations, comprehensive timing analysis
- **Best for**: Testing real policy behavior with timing fix
- **Dependencies**: Isaac Sim + latest ONNX policy file

### 3. `forest_gump_simple_demo.py` - 🔬 Physics Investigation
- **Purpose**: Original physics timestep investigation script
- **Features**: Tests different physics frequencies, comprehensive logging
- **Best for**: Understanding physics timing effects
- **Dependencies**: Isaac Sim + latest ONNX policy file

## 🔍 Analysis Files

### `PHYSICS_TIMESTEP_INVESTIGATION.md`
- Comprehensive analysis of physics timing issues
- Root cause identification (control frequency mismatch)
- Recommended solutions and implementation details
- Physics timestep testing methodology

## 🚀 Quick Start

### Option 1: Minimal Test (Recommended First)
```bash
cd /home/goat/Documents/GitHub/Leatherback
python scripts/inference/forest_gump_minimal_demo.py
```
**Expected Results:**
- 60Hz physics rate
- 15Hz control update rate  
- 4x decimation ratio
- Stable timing validation

### Option 2: Full Fixed Demo
```bash
cd /home/goat/Documents/GitHub/Leatherback
python scripts/inference/forest_gump_fixed_demo.py
```
**Expected Results:**
- Real ONNX policy predictions
- Stable control behavior
- No extreme action oscillations
- Proper temporal dynamics

### Option 3: Physics Investigation
```bash
cd /home/goat/Documents/GitHub/Leatherback  
python scripts/inference/forest_gump_simple_demo.py
```

## 📊 Expected Improvements

With the control frequency fix, you should see:

### ✅ Stability Metrics
- **Position deviation**: < 1.0m throughout simulation
- **Joint velocities**: Normal ranges (no ±250 unit oscillations)
- **Action magnitudes**: Reasonable values (no saturation)
- **Timing consistency**: Stable control intervals

### ⚠️ Before Fix (Problems)
- Control frequency: 60Hz (4x too fast)
- Position deviation: >9m after 400 steps
- Joint velocity oscillations: ±250+ units
- Action saturation: Constant [50,50,50,50] throttle

### ✅ After Fix (Expected)
- Control frequency: 15Hz (matches training)
- Position deviation: <1m throughout
- Joint velocities: <25 units (stable)
- Action ranges: Normal distribution

## 🔬 Technical Details

### Training Environment Settings
```python
# From leatherback_env.py
decimation = 4                    # Control decimation
sim: SimulationCfg = SimulationCfg(dt=1/60, render_interval=decimation)
# Effective control frequency: 60Hz / 4 = 15Hz
```

### Fixed Inference Settings  
```python
PHYSICS_DT = 1/60              # 60Hz physics (matches training)
CONTROL_DECIMATION = 4         # Match training decimation  
EFFECTIVE_CONTROL_HZ = 15      # 60Hz / 4 = 15Hz control

# In physics callback:
if step_counter % CONTROL_DECIMATION == 0:
    # Only run ONNX inference every 4th physics step
    actions = onnx_model.predict(observations)
# Apply same actions for 4 physics steps
```

## 📝 Logging

All demos create detailed logs in `/logs/forest_gump/`:
- `minimal_demo_TIMESTAMP.log` - Timing validation results
- `fixed_demo_TIMESTAMP.log` - Full demo with ONNX inference  
- `simple_demo_TIMESTAMP.log` - Physics investigation data

## 🎯 Success Criteria

The fix is working if you see:
1. **Timing validation passed** in minimal demo
2. **No extreme action warnings** in fixed demo
3. **Stable position tracking** (deviation <1m)
4. **Normal joint velocities** (<25 units)
5. **Consistent control intervals** (every 4 physics steps)

## 🔄 Next Steps

After validating the timing fix:
1. Test with full CSV playback (`forest_gump_playback_broken.py` + timing fix)
2. Implement hybrid control (ONNX + PID backup)
3. Add joint velocity limits as additional safety
4. Consider model retraining with robust temporal features

## 📚 References

- `analysis_report_forest_gump_simulation.md` - Complete failure analysis
- Training environment: `source/Leatherback/Leatherback/tasks/direct/leatherback/leatherback_env.py`
- Original broken script: `forest_gump_playback_broken.py`
