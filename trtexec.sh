# trtexec --onnx=weights/mmgsn_gen60.onnx --saveEngine=./engine/mmgsn_gen60.trt  --explicitBatch --useCudaGraph --fp16

# trtexec  --loadEngine=./engine/yolox_s_cprelu_608x1088_quantized_0.plan \
#         --calib=MOT17_FRCNN_cprelu.cache \
#         --shapes=images:1x3x608x1088 \
#         --workspace=4096 \
#         --useSpinWait \
#         --int8		

# trtexec  --loadEngine=./engine/yolov5s_nc3_bs32_op13_quantized.engine \
#         --shapes=images:32x3x640x640 \
#         --workspace=4096 \
#         --useSpinWait \
#         --int8	

# trtexec  --loadEngine=./engine/rgbt_yolov5_m3fd_op13_dynamic_quantized.engine \
#         --shapes=input0:1x3x640x640,input1:1x3x640x640 \
#         --workspace=4096 \
#         --useSpinWait \
#         --useCudaGraph \
#         --fp16 \
#         --int8		

# trtexec  --loadEngine=./engine/fusion_y_static_shape_one_input_quantized.engine \
#         --shapes=input:2x3x768x1024 \
#         --workspace=40960 \
#         --fp16 \
#         --int8 \
        # --useSpinWait \
        # --useCudaGraph 
        # --fp16 \
        # --int8	

# trtexec  --loadEngine=./engine/fusion_low_static_shape_one_input_fp16_768x1024_opt.engine \
#         --shapes=input:2x3x384x512 \
#         --workspace=40960 \
#         --fp16 \

# trtexec  --loadEngine=./engine/fusion_low_ycbcr_static_shape.engine \
#         --shapes=visible:1x3x640x640,infrared:1x3x640x640 \
#         --workspace=40960 \
#         --useSpinWait \
#         --fp16 \
        # --int8	
        
# trtexec --loadEngine=./engine/yolox_s_cprelu_608x1088.trt --shapes=images:1x3x608x1088 --fp16

trtexec  --loadEngine=./engine/rgbt_ca_rtdetrv2_589_m3fd_one_input_SYMM_LINEAR_PERCHANNEL.engine \
        --calib=rgbt.cache \
        --shapes=input:2x3x640x640 \
        --workspace=4096 \
        --useSpinWait \
        --useCudaGraph \
        --int8		