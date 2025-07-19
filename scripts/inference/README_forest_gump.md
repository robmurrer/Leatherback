# Forest Gump Scripts

This directory contains two related but different scripts for ONNX policy inference:

## 🚀 forest_gump_demo.py (NEW - WORKING)
**Clean, simple demonstration script**

- ✅ **Working and ready to use**
- 🎯 Auto-detects latest ONNX policy file
- 🗺️ Spawns fresh environment with generated waypoints
- 🤖 Real-time ONNX inference with live robot navigation
- 📊 Simple progress logging
- 🎨 Perfect for demonstrations and showcasing trained policies

**Usage:**
```bash
python scripts/inference/forest_gump_demo.py
```

**What it does:**
1. Finds the latest trained policy (ONNX file)
2. Creates a fresh simulation environment
3. Generates waypoints in a rectangular pattern
4. Runs the robot using ONNX inference to navigate through waypoints
5. Shows real-time progress and waypoint completion

---

## ⚠️ forest_gump_playback_broken.py (BROKEN - FOR REFERENCE)
**CSV validation script with known issues**

- ❌ **Not working properly - has significant issues**
- 📋 Attempts to validate ONNX predictions against logged CSV data
- 🐛 Encounters position deviations and synchronization problems
- 📊 Complex validation logic that fails due to simulation drift
- 📝 Kept for reference and future debugging

**Known Issues:**
- Large position deviations between expected and actual robot positions
- Simulation timing synchronization problems
- Action/observation validation failures
- See `analysis_report_forest_gump_simulation.md` for detailed analysis

**Status:** Use `forest_gump_demo.py` instead for live demonstrations.

---

## Key Differences

| Feature | forest_gump_demo.py | forest_gump_playback_broken.py |
|---------|-------------------|---------------------------|
| **Purpose** | Live demonstration | CSV validation |
| **Status** | ✅ Working | ❌ Broken |
| **Environment** | Fresh spawn with waypoints | Attempts to replay logged positions |
| **Complexity** | Simple & clean | Complex validation logic |
| **Use Case** | Showcasing policies | Debugging/validation (broken) |
| **Waypoints** | Generated at runtime | From CSV data |
| **Code Size** | ~150 lines | ~425 lines |

---

## Recommendation

**Use `forest_gump_demo.py` for:**
- Demonstrating trained policies
- Testing ONNX inference
- Showcasing robot navigation
- Clean, reliable demos

**Avoid `forest_gump_playback_broken.py` unless:**
- You're debugging the validation approach
- You're working on fixing the synchronization issues
- You need to reference the CSV validation logic
