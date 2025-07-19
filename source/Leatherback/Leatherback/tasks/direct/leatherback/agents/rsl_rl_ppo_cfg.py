# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LeatherbackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for training Leatherback with RSL-RL PPO."""
    
    # Training parameters (matching SKRL config exactly)
    num_steps_per_env = 32  # rollouts in SKRL
    max_iterations = 300    # 9600 timesteps / 32 rollouts = 300 iterations
    save_interval = 50
    experiment_name = "leatherback_direct"
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(
        # Network architecture (matching SKRL: [32, 32])
        init_noise_std=1.0,  # Back to original value
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        # Match SKRL hyperparameters exactly
        value_loss_coef=2.0,     # value_loss_scale in SKRL
        use_clipped_value_loss=True,  # clip_predicted_values in SKRL
        clip_param=0.2,          # ratio_clip in SKRL
        entropy_coef=0.2,        # entropy_loss_scale in SKRL
        num_learning_epochs=8,   # learning_epochs in SKRL
        num_mini_batches=8,      # mini_batches in SKRL
        learning_rate=5e-4,      # learning_rate in SKRL
        schedule="adaptive",     # learning_rate_scheduler: KLAdaptiveLR
        #schedule="linear",     # learning_rate_scheduler: KLAdaptiveLR
        gamma=0.99,              # discount_factor in SKRL
        lam=0.95,                # lambda in SKRL
        desired_kl=0.008,        # kl_threshold in SKRL
        max_grad_norm=1.0,       # grad_norm_clip in SKRL
    )
