import onnx
import torch
import numpy as np
from onnxconverter_common import auto_mixed_precision

# Define the shape of the input tensor for ResNet50
input_shape = (1, 3, 224, 244)

# Create a sample input tensor in FP16
test_input = np.random.rand(*input_shape).astype(np.float16)

# Convert the test input into a PyTorch tensor
test_data = torch.tensor(test_input)

model = onnx.load("resnet50_model.onnx")
model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "resnet50_model_mixed.onnx")