# trtexec --onnx=weights/mmgsn_gen60.onnx --saveEngine=./engine/mmgsn_gen60.trt  --explicitBatch --useCudaGraph --fp16

# trtexec  --loadEngine=./engine/yolox_s_cprelu_608x1088_quantized_0.plan \
#         --calib=MOT17_FRCNN_cprelu.cache \
#         --shapes=images:1x3x608x1088 \
#         --workspace=4096 \
#         --useSpinWait \
#         --int8		

trtexec  --loadEngine=./engine/yolov5s_nc3_bs32_op13_quantized.engine \
        --shapes=images:32x3x640x640 \
        --workspace=4096 \
        --useSpinWait \
        --int8		
# trtexec --loadEngine=./engine/yolox_s_cprelu_608x1088.trt --shapes=images:1x3x608x1088 --fp16