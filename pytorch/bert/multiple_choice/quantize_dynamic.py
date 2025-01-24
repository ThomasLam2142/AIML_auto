from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths to the ONNX model
model_path = "bert_mc_model.onnx"  # This is the ONNX model path after conversion
quantized_model_path = "bert_mc_model_int8.onnx"  # Path to save the quantized model

# Perform dynamic quantization
quantize_dynamic(
    model_input=model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QUInt8
)

print(f"Quantized model saved to: {quantized_model_path}")