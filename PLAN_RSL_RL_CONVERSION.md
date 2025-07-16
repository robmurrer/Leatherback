# Leatherback RSL-RL Conversion Plan

## Overview
This document outlines the plan to convert the Leatherback project from SKRL to RSL-RL. The current project uses SKRL for reinforcement learning training, but we want to add RSL-RL support while maintaining the existing SKRL functionality.

## Current State Analysis

### Existing Structure
- **Environment**: `LeatherbackEnv` inherits from `DirectRLEnv` (Isaac Lab)
- **Configuration**: `LeatherbackEnvCfg` with proper action/observation spaces
- **RL Library**: Currently uses SKRL with PPO configuration
- **Scripts**: Separate train/play scripts for SKRL
- **Task Registration**: Registered as `Template-Leatherback-Direct-v0`

### Key Environment Details
- **Action Space**: 2 (throttle, steering)
- **Observation Space**: 8 (position error, heading, velocities, previous actions)
- **Environment**: 4-wheeled vehicle navigating through 10 waypoints
- **Reward**: Based on position progress, heading alignment, and goal completion

## Implementation Steps

### Step 1: Create RSL-RL Configuration File
**File**: `source/Leatherback/Leatherback/tasks/direct/leatherback/agents/rsl_rl_ppo_cfg.py`

**Content**:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from rsl_rl.algorithms.ppo import PPOCfg

class LeatherbackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for training Leatherback with RSL-RL PPO."""
    
    def __post_init__(self):
        # Set experiment details
        self.experiment_name = "leatherback_direct"
        self.seed = 42
        
        # Training parameters (matching SKRL config)
        self.num_steps_per_env = 32  # rollouts in SKRL
        self.max_iterations = 300    # 9600 timesteps / 32 rollouts = 300 iterations
        self.empirical_normalization = False
        self.policy = RslRlOnPolicyRunnerCfg.PolicyCfg(
            # Network architecture (matching SKRL: [32, 32])
            init_noise_std = 1.0,
            actor_hidden_dims = [32, 32],
            critic_hidden_dims = [32, 32],
            activation = "elu",
        )
        
        self.algorithm = PPOCfg(
            # Match SKRL hyperparameters
            value_loss_coef = 2.0,     # value_loss_scale in SKRL
            use_clipped_value_loss = True,  # clip_predicted_values in SKRL
            clip_param = 0.2,          # ratio_clip in SKRL
            entropy_coef = 0.0,        # entropy_loss_scale in SKRL
            num_learning_epochs = 8,   # learning_epochs in SKRL
            num_mini_batches = 8,      # mini_batches in SKRL
            learning_rate = 5e-4,      # learning_rate in SKRL
            schedule = "adaptive",     # learning_rate_scheduler: KLAdaptiveLR
            gamma = 0.99,              # discount_factor in SKRL
            lam = 0.95,                # lambda in SKRL
            desired_kl = 0.008,        # kl_threshold in SKRL
            max_grad_norm = 1.0,       # grad_norm_clip in SKRL
        )
```

### Step 2: Update Task Registration
**File**: `source/Leatherback/Leatherback/tasks/direct/leatherback/__init__.py`

**Action**: Add RSL-RL configuration entry point to existing registration

**Current Code**:
```python
gym.register(
    id="Template-Leatherback-Direct-v0",
    entry_point=f"{__name__}.leatherback_env:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_env:LeatherbackEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

**Updated Code**:
```python
gym.register(
    id="Template-Leatherback-Direct-v0",
    entry_point=f"{__name__}.leatherback_env:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leatherback_env:LeatherbackEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeatherbackPPORunnerCfg",
    },
)
```

## Usage Commands

### Training with RSL-RL
```bash
# Basic training
python scripts/rsl_rl/train.py --task Template-Leatherback-Direct-v0 --num_envs 32

# Training with more environments (faster)
python scripts/rsl_rl/train.py --task Template-Leatherback-Direct-v0 --num_envs 4096 --headless

# Training with custom iterations
python scripts/rsl_rl/train.py --task Template-Leatherback-Direct-v0 --num_envs 32 --max_iterations 500
```

### Playing with RSL-RL
```bash
# Play with latest checkpoint
python scripts/rsl_rl/play.py --task Template-Leatherback-Direct-v0 --num_envs 32

# Play with specific checkpoint
python scripts/rsl_rl/play.py --task Template-Leatherback-Direct-v0 --num_envs 32 --checkpoint logs/rsl_rl/leatherback_direct/*/model_*.pt
```

## File Structure After Implementation
```
source/Leatherback/Leatherback/tasks/direct/leatherback/agents/
├── __init__.py
├── skrl_ppo_cfg.yaml          # Existing SKRL config
└── rsl_rl_ppo_cfg.py          # New RSL-RL config
```
