# Forest Gump Logging

This document describes the logging functionality added to `forest_gump.py` for better debugging and analysis.

## Logging Configuration

The script now automatically creates timestamped log files in the `/home/goat/Documents/GitHub/Leatherback/logs/forest_gump/` directory.

### Log File Format
- **File naming**: `forest_gump_YYYYMMDD_HHMMSS.log`
- **Format**: `TIMESTAMP - LEVEL - MESSAGE`
- **Example**: `2025-01-17 14:30:45,123 - INFO - Starting simulation...`

## Log Levels

### Console Output (INFO and above)
- **INFO**: Important status messages, simulation start/end, configuration
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors and simulation deviations

### File Output (DEBUG and above)
- **DEBUG**: Detailed step-by-step information, observations, actions, positions
- **INFO**: Same as console
- **WARNING**: Same as console  
- **ERROR**: Same as console

## Key Logged Information

### Simulation Start
- Log file location
- Configuration parameters (thresholds, epsilons, intervals)
- CSV file validation
- Robot starting position
- Joint mapping information

### During Simulation
- **Every step (DEBUG)**: 
  - Step number, simulation time, CSV timestep
  - Expected vs actual robot position
  - Observation values
  - Action predictions
- **Periodic checks (INFO)**:
  - Position deviation checks
  - Reset detection
- **On deviation (ERROR)**:
  - Detailed position mismatch information
  - Timing comparisons
  - Deviation statistics

### Simulation End
- Total simulation steps
- Position deviation statistics:
  - Average deviation
  - Maximum deviation
  - Total checks performed

## Enhanced Features

### Position Tracking
- **Tolerance**: 1e-3m (configurable via `POSITION_EPSILON`)
- **Check interval**: Every 10 steps (configurable via `POSITION_CHECK_INTERVAL`)
- **Deviation threshold**: 0.5m (stops simulation if exceeded)

### Action Validation
- **Tolerance**: 1e-4 (configurable via `ACTION_EPSILON`)
- **Comparison**: Predicted vs logged actions with floating-point precision handling

### Reset Detection
- **Threshold**: 5.0m position jump between consecutive CSV rows
- **Action**: Resets world state, robot position, and joint states

## Usage

1. **Run the script**: The logging is automatic
2. **Find logs**: Check the console output for the log file location
3. **Monitor progress**: Use `tail -f` to follow the log in real-time:
   ```bash
   tail -f /home/goat/Documents/GitHub/Leatherback/logs/forest_gump/forest_gump_YYYYMMDD_HHMMSS.log
   ```

## Debug Analysis

When simulation deviations occur, the log file contains:
- Exact step where deviation occurred
- Expected vs actual positions
- Timing information
- Historical deviation statistics

This information helps identify:
- Simulation drift patterns
- Timing synchronization issues
- Action/observation mismatches
- Reset detection accuracy

## Configuration

Key constants can be adjusted at the top of the script:
```python
OBSERVATION_EPSILON = 1e-6
ACTION_EPSILON = 1e-4
POSITION_EPSILON = 1e-3
POSITION_CHECK_INTERVAL = 10
```
