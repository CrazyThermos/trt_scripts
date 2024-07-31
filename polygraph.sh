# polygraphy convert weights/yolox_s_cprelu_608x1088.onnx \
#                     --int8 \
#                     --data-loader-script dataloader.py \
#                     --calibration-cache MOT17_FRCNN_cprelu.cache \
#                     --convert-to trt \
#                     -o engine/yolox_s_cprelu_608x1088_quantized.plan \
#                     --trt-min-shapes images:[1,3,608,1088] \
#                     --trt-opt-shapes images:[1,3,608,1088] \
#                     --trt-max-shapes images:[1,3,608,1088] 

# polygraphy convert weights/rgbt_rgbt_m3fd_b1_op13.onnx \
#                     --fp16 \
#                     --convert-to trt \
#                     -o engine/rgbt_rgbt_m3fd_b1_op13_fp16.engine \
#                     --trt-min-shapes input0:[1,3,640,640];input1:[1,3,640,640] \
#                     --trt-opt-shapes input0:[1,3,640,640];input1:[1,3,640,640] \
#                     --trt-max-shapes input0:[1,3,640,640];input1:[1,3,640,640] 


# polygraphy convert weights/rgbt_yolov5_m3fd_debug_op13_one_input.onnx \
#                     --fp16 \
#                     --convert-to trt \
#                     -o engine/rgbt_yolov5_m3fd_debug_b2_op13_one_input.engine \
#                     --trt-min-shapes input:[2,3,640,640] \
#                     --trt-opt-shapes input:[2,3,640,640] \
#                     --trt-max-shapes input:[2,3,640,640] 
                    
                    # --fp32 \

# polygraphy convert weights/fusion_y_one_input.onnx \
#                     --fp16 \
#                     --convert-to trt \
#                     -o engine/fusion_y_one_input_fp16_768x1024.engine \
#                     --trt-min-shapes input:[2,3,768,1024] \
#                     --trt-opt-shapes input:[2,3,768,1024] \
#                     --trt-max-shapes input:[2,3,768,1024] 

    
# polygraphy convert weights/fusion_one_input.onnx \
#                     --fp16 \
#                     --convert-to trt \
#                     -o engine/fusion_one_input_fp16_768x1024_static_output.engine \
#                     --trt-min-shapes input:[2,3,768,1024] \
#                     --trt-opt-shapes input:[2,3,768,1024] \
#                     --trt-max-shapes input:[2,3,768,1024] 

polygraphy convert weights/fusion_low_y_dynamic_shape_one_input.onnx \
                    --fp16 \
                    --convert-to trt \
                    -o engine/fusion_low_static_shape_one_input_fp16_768x1024_opt.engine \
                    --trt-min-shapes input:[2,3,384,512] \
                    --trt-opt-shapes input:[2,3,384,512] \
                    --trt-max-shapes input:[2,3,384,512] 

# polygraphy convert weights/fusion_low_y_dynamic_shape.onnx \
#                     --fp16 \
#                     --convert-to trt \
#                     -o engine/fusion_low_y_dynamic_shape.engine \
#                     --direct-io 
                    # --trt-min-shapes visible:[1,1,768,1024]infrared:[1,1,768,1024] \
                    # --trt-opt-shapes visible:[1,1,768,1024]infrared:[1,1,768,1024] \
                    # --trt-max-shapes visible:[1,1,768,1024]infrared:[1,1,768,1024] 