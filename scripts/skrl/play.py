# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# CSV logging is always enabled - removed CLI options

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import csv
import datetime
import numpy as np

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import Leatherback.tasks  # noqa: F401
import Robpole.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # setup CSV logging for observations and actions (always enabled)
    csv_file = None
    csv_writer = None
    csv_initialized = False
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_dir = os.path.join(log_dir, "csv_logs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = f"observations_actions_{timestamp}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    csv_file = open(csv_path, 'w', newline='')
    print(f"[INFO] Logging observations and actions to: {csv_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                # - multi-agent (deterministic) actions
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                # - single-agent (deterministic) actions
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                # env stepping
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # log observations and actions to CSV (always enabled)
                if not csv_initialized:
                        # initialize CSV headers based on actual data shapes
                        obs_flat = obs.cpu().numpy().flatten() if isinstance(obs, torch.Tensor) else np.array(obs).flatten()
                        actions_flat = actions.cpu().numpy().flatten() if isinstance(actions, torch.Tensor) else np.array(actions).flatten()
                        
                        # create headers with meaningful names
                        obs_headers = [
                            "obs_0_position_error",
                            "obs_1_heading_cos", 
                            "obs_2_heading_sin",
                            "obs_3_linear_vel_x",
                            "obs_4_linear_vel_y", 
                            "obs_5_angular_vel_z",
                            "obs_6_throttle_state",
                            "obs_7_steering_state"
                        ]
                        action_headers = [
                            "action_0_throttle",
                            "action_1_steering"
                        ]
                        reset_headers = ["terminated", "truncated", "episode_reset"]
                        # Add position headers for compatibility with forest_gump and plotting scripts
                        position_headers = ["pos_x", "pos_y", "pos_z"] 
                        headers = ["timestep", "sim_time"] + obs_headers + action_headers + reset_headers + position_headers
                        
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(headers)
                        csv_initialized = True
                        print(f"[INFO] CSV initialized with {len(obs_flat)} observations and {len(actions_flat)} actions")
                
                # write current step data
                obs_flat = obs.cpu().numpy().flatten() if isinstance(obs, torch.Tensor) else np.array(obs).flatten()
                actions_flat = actions.cpu().numpy().flatten() if isinstance(actions, torch.Tensor) else np.array(actions).flatten()
                
                # Check for any resets (terminated or truncated)
                any_terminated = terminated.any().item() if isinstance(terminated, torch.Tensor) else any(terminated) if hasattr(terminated, '__iter__') else terminated
                any_truncated = truncated.any().item() if isinstance(truncated, torch.Tensor) else any(truncated) if hasattr(truncated, '__iter__') else truncated
                episode_reset = any_terminated or any_truncated
                
                # Get robot position safely through environment interface  
                # Use environment state instead of direct robot access
                try:
                    # Try to get position from info dict first (most reliable)
                    if 'robot_position' in info:
                        pos = info['robot_position'][0].cpu().numpy() if isinstance(info['robot_position'], torch.Tensor) else info['robot_position'][0]
                    # Fallback to environment unwrapped access (only if available)
                    elif hasattr(env.unwrapped, 'leatherback') and hasattr(env.unwrapped.leatherback, 'data'):
                        pos = env.unwrapped.leatherback.data.root_pos_w[0, :3].cpu().numpy()
                    else:
                        # Last resort: use zeros (but log the issue)
                        pos = [0.0, 0.0, 0.0]
                        if timestep == 0:  # Only warn once
                            print(f"[WARNING] Could not access robot position - using fallback values")
                except Exception as e:
                    pos = [0.0, 0.0, 0.0]
                    if timestep == 0:  # Only warn once  
                        print(f"[WARNING] Error accessing robot position: {e} - using fallback values")
                
                sim_time = timestep * dt
                row_data = [timestep, sim_time] + obs_flat.tolist() + actions_flat.tolist() + [any_terminated, any_truncated, episode_reset] + pos.tolist()
                csv_writer.writerow(row_data)
                
                # Log reset information when it happens
                if episode_reset:
                    print(f"[INFO] Episode reset detected at step {timestep}: terminated={any_terminated}, truncated={any_truncated}")
                
            # Increment timestep counter
            timestep += 1
            
            # Check exit conditions
            if args_cli.video:
                # exit the play loop after recording one video
                if timestep >= args_cli.video_length:
                    break
            else:
                # Exit after logging a reasonable number of steps (one episode worth)
                if timestep >= 1000:  # Adjust this number based on typical episode length
                    print(f"[INFO] CSV logging completed after {timestep} steps.")
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    
    finally:
        # ensure CSV file is properly closed
        if csv_file:
            csv_file.close()
            print(f"[INFO] CSV logging completed. {timestep} steps logged to {csv_path}")

    # Generate plots automatically after play session
    print("🎨 Automatically generating plots from collected data...")
    try:
        # Import and run the plotting script
        import sys
        script_dir = os.path.dirname(__file__)
        plot_script_path = os.path.join(script_dir, "..", "plot_observations_actions.py")
        if os.path.exists(plot_script_path):
            # Add the script directory to Python path so we can import the plotting functions
            sys.path.insert(0, os.path.dirname(plot_script_path))
            
            # Import the plotting function
            from plot_observations_actions import plot_csv_data
            
            print(f"📊 Generating plots for: {csv_path}")
            plot_csv_data(csv_path, show_plots=False)  # Don't show plots interactively
            print("✅ Plots generated successfully!")
        else:
            print(f"⚠️ Plot script not found at: {plot_script_path}")
    except Exception as e:
        print(f"❌ Failed to generate plots: {e}")
        print("You can manually run: python scripts/plot_observations_actions.py")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
