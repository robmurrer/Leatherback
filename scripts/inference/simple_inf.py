import os
import numpy
import onnxruntime as rt

script_path = os.path.dirname(__file__)
policy_path_rel = "../../logs/rsl_rl/cartpole_direct/2025-07-16_10-00-54/exported/policy.onnx"
policy_path = os.path.join(script_path, policy_path_rel)
sess = rt.InferenceSession(policy_path, providers=rt.get_available_providers())

print(sess)
for inp in sess.get_inputs():
    print(f"Input Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")


input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

import numpy as np

X_test = np.array([[-0.7193706035614014,2.2654635906219482,0.027544492855668068,2.144970417022705]])
expected_action = 2.1009788513183594
print(input_name, label_name)
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
print(f"Expected: {expected_action}, Predicted: {pred_onx[0][0]}")


# second confirmation

#101,1.6833333333333333,-0.012251214124262333,0.014838214963674545,0.6669488549232483,-0.25134098529815674,-0.02319391444325447
X_test = np.array([[-0.012251214124262333,0.014838214963674545,0.6669488549232483,-0.25134098529815674]])

expected_action = -0.02319391444325447
print(input_name, label_name)
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
print(f"Expected: {expected_action}, Predicted: {pred_onx[0][0]}")