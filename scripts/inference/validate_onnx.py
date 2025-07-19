#!/usr/bin/env python3
"""
🧪 ONNX Model Validation Test

Test the ONNX model with exact observations from training data
to verify it produces expected actions.
"""

import numpy as np
import onnxruntime as rt
import os

def test_onnx_model():
    # Load the ONNX model
    script_dir = os.path.dirname(__file__)
    policy_path = os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "leatherback_direct", 
                              "2025-07-18_15-16-40", "exported", "policy.onnx")
    policy_path = os.path.abspath(policy_path)
    
    print(f"🎯 Loading ONNX model: {policy_path}")
    sess = rt.InferenceSession(policy_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"📊 Model input: {input_name}, output: {output_name}")
    
    # Test cases from training CSV (row 1, 2, 3)
    # Format: [obs_0_position_error, obs_1_heading_cos, obs_2_heading_sin, obs_3_linear_vel_x, 
    #          obs_4_linear_vel_y, obs_5_angular_vel_z, obs_6_throttle_state, obs_7_steering_state]
    # Expected actions: [action_0_throttle, action_1_steering]
    
    test_cases = [
        {
            "name": "Training Row 1",
            "observation": np.array([5.891197204589844, 0.977267324924469, -0.21201078593730927, 
                                   0.0507037453353405, -0.022322842851281166, 0.7193582057952881, 
                                   50.0, 0.5525519847869873], dtype=np.float32),
            "expected_actions": [19.28487777709961, -3.9794719219207764],
            "description": "Standard case with forward motion"
        },
        {
            "name": "Training Row 2", 
            "observation": np.array([5.863720893859863, 0.9809550046920776, -0.1942351907491684,
                                   0.7818400859832764, -0.29400378465652466, -1.833878755569458,
                                   50.0, -0.39794719219207764], dtype=np.float32),
            "expected_actions": [21.791423797607422, 0.9485114216804504],
            "description": "Case with lateral velocity and turning"
        },
        {
            "name": "Training Row 3",
            "observation": np.array([5.7953901290893555, 0.9923713207244873, -0.1232849657535553,
                                   1.2103641033172607, -0.06260162591934204, -0.26878416538238525,
                                   50.0, 0.09485114365816116], dtype=np.float32),
            "expected_actions": [20.79751968383789, 0.9354293942451477],
            "description": "Case with higher forward velocity"
        }
    ]
    
    print("\n🧪 TESTING ONNX MODEL WITH TRAINING DATA:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📝 Test {i+1}: {test_case['name']}")
        print(f"📄 Description: {test_case['description']}")
        
        # Prepare input (reshape to batch dimension)
        obs_input = test_case["observation"].reshape(1, -1)
        print(f"📊 Input observation: {obs_input.flatten()}")
        
        # Run ONNX inference
        onnx_actions = sess.run([output_name], {input_name: obs_input})[0]
        onnx_throttle = onnx_actions[0, 0]
        onnx_steering = onnx_actions[0, 1]
        
        # Expected actions from training
        expected_throttle = test_case["expected_actions"][0]
        expected_steering = test_case["expected_actions"][1]
        
        print(f"🎮 ONNX outputs:      throttle={onnx_throttle:.6f}, steering={onnx_steering:.6f}")
        print(f"🎯 Expected (training): throttle={expected_throttle:.6f}, steering={expected_steering:.6f}")
        
        # Calculate differences
        throttle_diff = abs(onnx_throttle - expected_throttle)
        steering_diff = abs(onnx_steering - expected_steering)
        throttle_error = throttle_diff / abs(expected_throttle) * 100 if expected_throttle != 0 else 0
        steering_error = steering_diff / abs(expected_steering) * 100 if expected_steering != 0 else 0
        
        print(f"📈 Differences:       throttle_diff={throttle_diff:.6f} ({throttle_error:.2f}%), steering_diff={steering_diff:.6f} ({steering_error:.2f}%)")
        
        # Check if within reasonable tolerance (±5%)
        if throttle_error < 5.0 and steering_error < 5.0:
            print(f"✅ PASS: Actions match training data within 5% tolerance")
        else:
            print(f"❌ FAIL: Actions differ significantly from training data")
            if throttle_error >= 5.0:
                print(f"   - Throttle error too high: {throttle_error:.2f}%")
            if steering_error >= 5.0:
                print(f"   - Steering error too high: {steering_error:.2f}%")
        
        print("-" * 60)
    
    print(f"\n🏁 ONNX Model Validation Complete!")

if __name__ == "__main__":
    test_onnx_model()
