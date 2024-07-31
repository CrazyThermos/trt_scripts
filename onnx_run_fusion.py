from PIL import Image
import onnxruntime
import numpy as np
import time

sess = onnxruntime.InferenceSession('./weights/fusion_one_input.onnx', providers=['CPUExecutionProvider'])
sess1 = onnxruntime.InferenceSession('./weights/fusion.onnx', providers=['CPUExecutionProvider'])


low_res = 640
imrgb_arr =  np.transpose(np.expand_dims(np.array(Image.open('./datasets/m3fd/00000_rgb.png'), dtype=np.float32), 0), (0,3,1,2)) / 255.0
imt_arr =  np.transpose(np.expand_dims(np.array(Image.open('./datasets/m3fd/00000_t.png'), dtype=np.float32), 0), (0,3,1,2)) / 255.0

inp = np.concatenate((imrgb_arr, imt_arr), 0)
# inp = np.transpose(inp, (0,3,1,2))
# x0 = np.ones((1, 3, low_res, low_res),dtype=np.float32)
# x1 = np.ones((2, 3, low_res, low_res),dtype=np.float32)
for i in range(1):
    time_start = time.time()  
    output0 = sess.run(['fusion'], {'input': inp})[0]
    # output1 = sess1.run(['fusion'], {'visible': imrgb_arr, 'infrared': imt_arr})[0]
    # if output0 == output1:
    # print(np.mean((output1 - output1) ** 2))
    time_end = time.time()  
    print("inference_time:{}".format(time_end-time_start))

# assert np.allclose(output, a * x + b)
image_array = np.squeeze(np.transpose((output0 * 255).astype(np.uint8), (0,2,3,1)))
image_pil = Image.fromarray(image_array, mode="RGB")
image_pil.save('output_image.png', 'PNG')
