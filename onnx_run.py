import onnxruntime
import numpy as np
import time

sess0 = onnxruntime.InferenceSession('./weights/rgbt_yolov5_m3fd_debug_op13.onnx', providers=['CPUExecutionProvider'])
sess1 = onnxruntime.InferenceSession('./weights/rgbt_yolov5_m3fd_debug_op13_one_input.onnx', providers=['CPUExecutionProvider'])


low_res = 640
x0 = np.ones((1, 3, low_res, low_res),dtype=np.float32)
x1 = np.ones((2, 3, low_res, low_res),dtype=np.float32)
for i in range(10):
    time_start = time.time()  
    output0 = sess0.run(['output0'], {'input0': x0, 'input1': x0})[0]
    output1 = sess1.run(['output0'], {'input': x1})[0]
    # if output0 == output1:
    print(np.mean((output1 - output1) ** 2))
    time_end = time.time()  
    print("inference_time:{}".format(time_end-time_start))
# assert np.allclose(output, a * x + b)