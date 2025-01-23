from onnxruntime.quantization import quantize_dynamic, QuantType

# Path to the original ONNX model and the quantized output path
onnx_model_path = "distilbert_qa_model.onnx"
quantized_model_path = "distilbert_qa_model_int8.onnx"

# Perform dynamic quantization to INT8
quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)

print(f"Quantized model saved to {quantized_model_path}")
