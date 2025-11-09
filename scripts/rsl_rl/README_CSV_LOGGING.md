# CSV Logging for RSL-RL Play Script

The `play.py` script has been enhanced with CSV logging functionality to record observations and actions during gameplay sessions.

## Usage

To enable CSV logging, use the `--log-csv` flag when running the play script:

```bash
python scripts/rsl_rl/play.py --task <TASK_NAME> --log-csv
```

## CSV Output Format

When CSV logging is enabled, a timestamped CSV file will be created in the log directory with the following columns:

### Metadata
- `timestep`: Current simulation timestep
- `env_id`: Environment ID (useful for multi-environment setups)
- `timestamp`: ISO format timestamp of the observation

### Actions (2D)
- `action_throttle`: Throttle action value
- `action_steering`: Steering action value

### Observations (8D)
Based on the LeatherbackEnv observation structure:
- `obs_position_error`: Distance error to target position
- `obs_cos_heading_error`: Cosine of heading error to target
- `obs_sin_heading_error`: Sine of heading error to target
- `obs_linear_vel_x`: Linear velocity in X direction (body frame)
- `obs_linear_vel_y`: Linear velocity in Y direction (body frame)
- `obs_angular_vel_z`: Angular velocity around Z axis (world frame)
- `obs_throttle_state`: Current throttle state
- `obs_steering_state`: Current steering state

## Example Usage

```bash
# Play with CSV logging enabled
python scripts/rsl_rl/play.py --task Isaac-Leatherback-Direct-v0 --log-csv --num_envs 1

# Play without CSV logging (default behavior)
python scripts/rsl_rl/play.py --task Isaac-Leatherback-Direct-v0
```

## Output Location

CSV files are saved in the same directory as the checkpoint logs with the filename format:
`play_session_YYYY-MM-DD_HH-MM-SS.csv`

## Data Analysis

The CSV files can be easily loaded into pandas for analysis:

```python
import pandas as pd

# Load the CSV data
df = pd.read_csv('path/to/play_session_YYYY-MM-DD_HH-MM-SS.csv')

# Analyze action distributions
print(df[['action_throttle', 'action_steering']].describe())

# Plot observations over time
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(df['timestep'], df['obs_position_error'])
plt.title('Position Error Over Time')
plt.xlabel('Timestep')
plt.ylabel('Position Error')

plt.subplot(2, 2, 2)
plt.plot(df['timestep'], df['obs_linear_vel_x'])
plt.title('Linear Velocity X Over Time')
plt.xlabel('Timestep')
plt.ylabel('Linear Velocity X')

# ... more plots as needed
plt.tight_layout()
plt.show()
```

## Performance Considerations

- CSV logging adds some computational overhead due to file I/O operations
- For long gameplay sessions or many environments, CSV files can become quite large
- Use `--log-csv` only when data collection is needed
- CSV data is flushed after each timestep to ensure data integrity even if the simulation is interrupted