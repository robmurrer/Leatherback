#!/usr/bin/env python3
"""
Quick test to load the latest trained model and check action ranges.
"""

import torch
import os
import sys
import numpy as np

def test_trained_model():
    """Test the latest trained model to see what action ranges it produces."""
    
    print("🔍 Testing trained model action ranges...")
    
    # Find the latest training run
    log_dir = "/home/goat/Documents/GitHub/Leatherback/logs/rsl_rl/leatherback_direct"
    
    # Get the latest run directory
    runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("2025-")]
    if not runs:
        print("❌ No training runs found!")
        return
    
    latest_run = sorted(runs)[-1]
    run_dir = os.path.join(log_dir, latest_run)
    print(f"📁 Latest run: {latest_run}")
    
    # Check what model files exist
    model_files = [f for f in os.listdir(run_dir) if f.endswith('.pt')]
    if not model_files:
        print("❌ No model files found!")
        return
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(run_dir, latest_model)
    print(f"🤖 Latest model: {latest_model}")
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        
        # Check what's in the checkpoint
        print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"🧠 Model state keys: {list(model_state.keys())}")
            
            # Look for actor/policy parameters
            actor_keys = [k for k in model_state.keys() if 'actor' in k.lower()]
            if actor_keys:
                print(f"🎭 Actor parameters found: {len(actor_keys)} keys")
                # Check the final layer (should output 2 actions)
                final_layer_keys = [k for k in actor_keys if 'weight' in k]
                if final_layer_keys:
                    final_key = final_layer_keys[-1]
                    final_weights = model_state[final_key]
                    print(f"🔚 Final layer shape: {final_weights.shape}")
                    print(f"📈 Final layer weight range: [{final_weights.min():.3f}, {final_weights.max():.3f}]")
        
        # Generate some test observations and see what actions the model would produce
        print(f"\n🧪 Testing model with sample observations...")
        
        # Create some sample observations (9D for our Ackermann setup)
        test_obs = torch.tensor([
            [2.0, 0.5, 0.8, 1.0, 0.0, 0.1, 2.0, 0.0, 0.0],      # Moving forward
            [5.0, -0.2, 0.9, 0.5, 0.2, -0.3, 1.5, 0.5, -0.2],   # Turning
            [1.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Stationary
        ], dtype=torch.float32)
        
        # This is just a rough test - would need full environment setup for real testing
        print(f"🔢 Sample observations shape: {test_obs.shape}")
        print(f"📏 Observation ranges:")
        for i in range(test_obs.shape[1]):
            obs_col = test_obs[:, i]
            print(f"  obs_{i}: [{obs_col.min():.3f}, {obs_col.max():.3f}]")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def check_training_progress():
    """Check the training progress from logs."""
    
    print(f"\n📈 Checking training progress...")
    
    log_dir = "/home/goat/Documents/GitHub/Leatherback/logs/rsl_rl/leatherback_direct"
    runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("2025-")]
    
    if not runs:
        print("❌ No training runs found!")
        return
    
    latest_run = sorted(runs)[-1]
    run_dir = os.path.join(log_dir, latest_run)
    
    # Check for progress logs
    progress_files = [f for f in os.listdir(run_dir) if 'progress' in f.lower() or f.endswith('.csv')]
    print(f"📄 Progress files: {progress_files}")
    
    # Check for model files to see training iterations
    model_files = [f for f in os.listdir(run_dir) if f.endswith('.pt')]
    model_iterations = []
    for f in model_files:
        try:
            # Extract iteration number from filename like "model_150.pt"
            iter_num = int(f.replace('model_', '').replace('.pt', ''))
            model_iterations.append(iter_num)
        except:
            pass
    
    if model_iterations:
        model_iterations.sort()
        print(f"🏁 Training iterations with saved models: {model_iterations}")
        print(f"🎯 Latest saved iteration: {max(model_iterations)}")
    
    # Check CSV logs
    csv_dir = os.path.join(run_dir, "csv_logs")
    if os.path.exists(csv_dir):
        csv_files = os.listdir(csv_dir)
        print(f"📊 CSV log files: {csv_files}")

if __name__ == "__main__":
    test_trained_model()
    check_training_progress()
    print(f"\n✅ Diagnostic completed!")
