import os
import numpy as np
from PIL import Image 

def load_data():
    root = './datasets/MOT17_FRCNN'
    img_paths = os.listdir(root)
    for path in img_paths:
        img = Image.open(root+'/'+path).convert('RGB').resize((1088 ,608))
        # print(img.shape)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0).transpose(0,3,1,2)
        img_array = img_array.astype(np.float32)/ 255.0
        # print(img_array.shape)
        yield {"images": img_array}  # Still totally real data

load_data()