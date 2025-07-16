# Observation and Action Logging

This directory contains scripts for logging and plotting observations and actions from Isaac Lab RL training runs.

## Files

- `play.py` - Modified play script that logs observations and actions to CSV
- `plot_observations_actions.py` - Script to plot the logged CSV data
- `plot_latest.sh` - Convenience script to install dependencies and plot the latest CSV
- `requirements_plotting.txt` - Python dependencies for plotting

## Usage

### 1. Running play.py with CSV logging

The CSV logging is enabled by default. To run with logging:

```bash
python scripts/rsl_rl/play.py --task=Isaac-Cartpole-Direct-v0 --num_envs=1
```

To disable CSV logging:

```bash
python scripts/rsl_rl/play.py --task=Isaac-Cartpole-Direct-v0 --num_envs=1 --no_log_csv
```

### 2. CSV Output

The CSV files are saved to: `logs/rsl_rl/<experiment_name>/<run_id>/csv_logs/`

Each CSV file contains:
- `timestep` - Step number
- `sim_time` - Simulation time in seconds
- `obs_0`, `obs_1`, ... - Flattened observation values
- `action_0`, `action_1`, ... - Action values

The filename format is: `observations_actions_YYYY-MM-DD_HH-MM-SS.csv`

### 3. Plotting the data

#### Option A: Use the convenience script
```bash
cd scripts
./plot_latest.sh
```

This will:
1. Install required dependencies (matplotlib, pandas, seaborn, numpy)
2. Find and plot the most recent CSV file
3. Show interactive plots and save them to a `plots/` directory

#### Option B: Use the plotting script directly

Install dependencies:
```bash
pip install -r scripts/requirements_plotting.txt
```

Plot a specific CSV file:
```bash
python scripts/plot_observations_actions.py --csv_path path/to/your/file.csv
```

Plot the latest CSV file:
```bash
python scripts/plot_observations_actions.py --logs_dir logs
```

Save plots without showing them:
```bash
python scripts/plot_observations_actions.py --no_show --output_dir my_plots
```

### 4. Generated Plots

The plotting script generates several visualizations:

1. **Observations Time Series** - All observation dimensions over time
2. **Actions Time Series** - All action dimensions over time  
3. **Action Distributions** - Histograms showing action value distributions
4. **Observation Statistics** - Rolling statistics and correlation matrix

## Command Line Options

### play.py additional options:
- `--log_csv` - Enable CSV logging (default: True)
- `--no_log_csv` - Disable CSV logging

### plot_observations_actions.py options:
- `--csv_path PATH` - Specific CSV file to plot
- `--logs_dir DIR` - Directory to search for CSV files (default: logs)
- `--output_dir DIR` - Directory to save plots
- `--no_show` - Don't display plots interactively

## Examples

1. Run a short episode and plot the results:
```bash
# Run simulation with video recording and CSV logging
python scripts/rsl_rl/play.py --task=Isaac-Cartpole-Direct-v0 --num_envs=1 --video --video_length=100

# Plot the results
cd scripts && ./plot_latest.sh
```

2. Analyze a long simulation run:
```bash
# Run without video but with CSV logging
python scripts/rsl_rl/play.py --task=Isaac-Cartpole-Direct-v0 --num_envs=1

# Plot with custom output directory
python scripts/plot_observations_actions.py --output_dir analysis_plots
```

3. Compare multiple runs:
```bash
# Plot specific files
python scripts/plot_observations_actions.py --csv_path logs/.../run1/csv_logs/observations_actions_*.csv --output_dir run1_plots
python scripts/plot_observations_actions.py --csv_path logs/.../run2/csv_logs/observations_actions_*.csv --output_dir run2_plots
```
