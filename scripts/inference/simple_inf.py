import os
import numpy
import onnxruntime as rt

script_path = os.path.dirname(__file__)
#policy_path_rel = "../../logs/rsl_rl/cartpole_direct/2025-07-16_10-00-54/exported/policy.onnx"
policy_path = "/home/goat/Documents/GitHub/Leatherback/logs/rsl_rl/leatherback_direct/2025-07-16_15-20-50/exported/policy.onnx"
#policy_path = os.path.join(script_path, policy_path_rel)
sess = rt.InferenceSession(policy_path, providers=rt.get_available_providers())

print(sess)
for inp in sess.get_inputs():
    print(f"Input Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")


input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

import numpy as np

#X_test = np.array([[-0.7193706035614014,2.2654635906219482,0.027544492855668068,2.144970417022705]])
#expected_action = 2.1009788513183594

X_test = np.array([[5.812362,0.940476,-0.339861,-0.043386,0.010108,-0.050856,38.587463,-0.208948]])
expected_action = (6.928863, -1.420094)

print(input_name, label_name)
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
print(f"Expected: {expected_action}, Predicted: {pred_onx}")

