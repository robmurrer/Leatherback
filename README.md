# Leatherback Driving Environment for Isaac Lab

## Overview

This repository contains a reinforcement learning environment for controlling a Leatherback wheeled robot within the NVIDIA Isaac Lab simulation platform.
The environment uses the "direct" workflow from Isaac Lab.

**Key Features:**

- Leatherback robot model with throttle and steering actuators.
- Basic driving task (details to be added based on your specific task goals).
- Integrates with Isaac Lab's simulation and RL capabilities.

## Installation

1.  **Install Isaac Lab:** Follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation.

2.  **Clone this repository:**
    ```bash
    git clone https://github.com/MuammerBay/isaac-leatherback.git
    cd isaac-leatherback
    ```

3.  **Install the package:** Using a python interpreter where Isaac Lab is installed (e.g., your `env_isaaclab` conda environment), install this package in editable mode:
    ```bash
    # Ensure your Isaac Lab conda environment is activated
    # conda activate env_isaaclab

    # Install the package
    pip install -e .
    ```
    *Note: The `-e` flag installs the package in "editable" mode, linking directly to your source code, which is convenient for development.* 

## Usage

1.  **Verify Installation:** List the available tasks to ensure the environment is registered correctly:
    ```bash
    # Make sure your Isaac Lab conda environment is activated
    python scripts/list_envs.py
    ```
    You should see `Template-Leatherback-Direct-v0` in the output.

2.  **Run Training:** Start an RL training session (replace `<RL_LIBRARY>` with the specific library configured by the template, e.g., `skrl`):
    ```bash
    # Make sure your Isaac Lab conda environment is activated
    python scripts/<RL_LIBRARY>/train.py --task Template-Leatherback-Direct-v0
    ```

## Dependencies

- NVIDIA Isaac Lab (check Isaac Lab documentation for specific version compatibility)
- Python >= 3.8
- PyTorch
- Gymnasium

## Custom Assets

The Leatherback robot model (`leatherback_simple_better.usd`) is included in `source/Leatherback/tasks/direct/leatherback/custom_assets/`.

## Development Setup (Optional)

### IDE Setup (VSCode)

Follow the instructions in the original template README for setting up VSCode Python environment indexing.

### Code Formatting

This project uses `pre-commit` for code formatting. Install and run it:
```bash
pip install pre-commit
pre-commit run --all-files
```

## License

This project uses the BSD-3-Clause license, consistent with Isaac Lab.