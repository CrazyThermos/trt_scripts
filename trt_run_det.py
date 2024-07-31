#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import time
import numpy as np
import tensorrt as trt
import argparse
from PIL import ImageDraw
from PIL import Image

# from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
from downloader import getFilePath

TRT_LOGGER = trt.Logger()

# def parse_args():
#     """ parse args """
#     parser = argparse.ArgumentParser(__doc__)
#     parser.add_argument('--dataset_name', type=str, default='advertisegen',
#                         help='The name of the dataset to load.')
#     parser.add_argument('--model_name_or_path', type=str, default='./model/',
#                         help='The path or shortcut name of the pre-trained model.')
#     parser.add_argument("--eval_file", type=str, required=False, default="./data/eval.json",
#                         help="Predict data path.")
#     parser.add_argument('--batch_size', type=int, default=16,
#                         help='Batch size per GPU/CPU for training.')
#     parser.add_argument('--output_path', type=str, default='./predict.txt',
#                         help='The file path where the infer result will be saved.')
#     parser.add_argument("--use_fp16_decoding", action="store_true",
#                         help="Whether to use fp16 decoding to predict. ")
#     parser.add_argument("--use_ft", action="store_true",
#                         help="Whether to use FasterUNIMOText model. ")
#     parser.add_argument('--decode_strategy', type=str, default='beam_search', choices=["beam_search", "greedy_search"],
#                         help='The decode strategy in generation.')
#     parser.add_argument(
#         "--decoding_lib", default="lib/libdecoding_op.so", type=str, help="Path of libdecoding_op.so."
#     )
#     args = parser.parse_args()
#     return args

def get_engine(onnx_file_path, engine_file_path="", shape=None, build_model=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    if not build_model:
        # 设置TensorRT日志级别
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 创建TensorRT Runtime对象
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_file_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 30  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = shape
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main(onnx_file_path, int8_engine_file_path="", fp16_engine_file_path="", shape=None, build_model=False):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    # Do inference with TensorRT
    int8_trt_outputs = []
    with get_engine(onnx_file_path, int8_engine_file_path, shape, build_model) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # print("Running inference on image {}...".format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        low_res_h = shape[2]
        low_res_w = shape[3]
        # x = np.ones((1, 3, low_res_h, low_res_w),dtype=np.float32)
        imrgb_arr =  np.transpose(np.expand_dims(np.array(np.resize(Image.open('./datasets/m3fd/00000_rgb.png'),(shape[1],low_res_h, low_res_w)), dtype=np.float32), 0), (0,3,1,2)) / 255.0
        imt_arr =  np.transpose(np.expand_dims(np.array(np.resize(Image.open('./datasets/m3fd/00000_t.png'),(shape[1],low_res_h, low_res_w)), dtype=np.float32), 0), (0,3,1,2)) / 255.0
        inp = np.concatenate((imrgb_arr, imt_arr), 0)
        # inputs[0].host = x
        inputs[0].host = inp
        # inputs[1].host = x
        time_start = time.time()  
        int8_trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        time_end = time.time()  
        print("inference_time:{}",time_end-time_start)
        int8_trt_outputs = int8_trt_outputs[0]
        # image_array = (trt_outputs * 255).astype(np.uint8)
        # image_pil = Image.fromarray(image_array, mode="RGB")
        # image_pil.save('output_image_trt.png', 'PNG')
    
    fp16_trt_outputs = []
    with get_engine(onnx_file_path, fp16_engine_file_path, shape, build_model) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        # print("Running inference on image {}...".format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        low_res_h = shape[2]
        low_res_w = shape[3]
        # x = np.ones((1, 3, low_res_h, low_res_w),dtype=np.float32)
        imrgb_arr =  np.transpose(np.expand_dims(np.array(np.resize(Image.open('./datasets/m3fd/00000_rgb.png'),(shape[1],low_res_h, low_res_w)), dtype=np.float32), 0), (0,3,1,2)) / 255.0
        imt_arr =  np.transpose(np.expand_dims(np.array(np.resize(Image.open('./datasets/m3fd/00000_t.png'),(shape[1],low_res_h, low_res_w)), dtype=np.float32), 0), (0,3,1,2)) / 255.0
        inp = np.concatenate((imrgb_arr, imt_arr), 0)
        # inputs[0].host = x
        inputs[0].host = inp
        # inputs[1].host = x
        time_start = time.time()  
        fp16_trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        time_end = time.time()  
        print("inference_time:{}",time_end-time_start)
        fp16_trt_outputs = fp16_trt_outputs[0]
        # image_array = (trt_outputs * 255).astype(np.uint8)
        # image_pil = Image.fromarray(image_array, mode="RGB")
        # image_pil.save('output_image_trt.png', 'PNG')
    pass


if __name__ == "__main__":
    onnx_file_path = "./weights/fusion.onnx"
    # onnx_file_path = "./weights/t1ven.onnx"
    int8_engine_file_path = "./engine/rgbt_yolov5_m3fd_op13_one_input_int8_SYMM_LINEAR_PERCHANNEL_dynamic_quantized.engine"
    fp16_engine_file_path = "./engine/rgbt_yolov5_m3fd_debug_b2_op13_one_input.engine"

    # engine_file_path = "t1ven.trt"
    shape = [1,3,640,640]
    main(onnx_file_path, int8_engine_file_path, fp16_engine_file_path, shape, build_model=False)