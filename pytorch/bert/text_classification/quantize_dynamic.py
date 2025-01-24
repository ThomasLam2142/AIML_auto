from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths to the ONNX model
model_path = "bert_tc_model.onnx"
quantized_model_path = "bert_tc_model_int8.onnx"

# Perform dynamic quantization
quantize_dynamic(
    model_input=model_path,
    model_output=quantized_model_path,
    per_channel=False,     
    weight_type=QuantType.QUInt8
)

print("ONNX model has been successfully quantized to INT8.")