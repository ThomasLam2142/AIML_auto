import onnx
from onnxconverter_common import auto_mixed_precision
import numpy as np
from transformers import AutoTokenizer

# Path to the existing ONNX model
onnx_model_path = "bert_tc_model.onnx"

# Load the existing ONNX model
onnx_model = onnx.load(onnx_model_path)

# Initialize the tokenizer (same as used in the PyTorch to ONNX conversion)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Create a dummy input
dummy_input_text = "This is a dummy input"
dummy_input = tokenizer.encode(dummy_input_text, return_tensors="pt")

# Convert PyTorch tensor to numpy array
dummy_input_np = dummy_input.numpy()

# Create a dictionary with the input name and the dummy input data
input_name = onnx_model.graph.input[0].name
dummy_input_dict = {input_name: dummy_input_np}

# Convert the ONNX model to mixed precision (FP16)
model_mixed = auto_mixed_precision.auto_convert_mixed_precision(
    onnx_model, 
    dummy_input_dict, 
    rtol=0.18, 
    atol=0.018, 
    keep_io_types=True
)

# Save the mixed precision model
onnx_model_mixed_path = "bert_tc_model_mixed.onnx"
onnx.save_model(model_mixed, onnx_model_mixed_path)

print("Model has been converted to mixed precision and saved")
