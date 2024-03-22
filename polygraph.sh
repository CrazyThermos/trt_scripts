polygraphy convert weights/yolox_s_cprelu_608x1088.onnx \
                    --int8 \
                    --data-loader-script dataloader.py \
                    --calibration-cache MOT17_FRCNN_cprelu.cache \
                    --convert-to trt \
                    -o engine/yolox_s_cprelu_608x1088_quantized.plan \
                    --trt-min-shapes images:[1,3,608,1088] \
                    --trt-opt-shapes images:[1,3,608,1088] \
                    --trt-max-shapes images:[1,3,608,1088] 