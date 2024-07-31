trtexec --onnx=./weights/fusion_static_shape_one_input.onnx \
        --saveEngine=./engine/fusion_static_shae_one_input_fp16_768x1024.engine \
        --shapes=input:2x3x768x1024 \
        --explicitBatch  \
        --tacticSources=tactics \
        --workspace=40960 \
        --fp16 \