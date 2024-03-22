import onnxruntime
import numpy as np
import time

sess = onnxruntime.InferenceSession('./weights/mmgsn_gen60.onnx', providers=['CUDAExecutionProvider'])

low_res = 640
x = np.ones((1, 3, low_res, low_res),dtype=np.float32)
x = np.ones((1, 3, low_res, low_res),dtype=np.float32)
for i in range(10):
    time_start = time.time()  
    output = sess.run(['output0'], {'input0': x, 'input1': x})[0]
    time_end = time.time()  
    print("inference_time:{}".format(time_end-time_start))
# assert np.allclose(output, a * x + b)