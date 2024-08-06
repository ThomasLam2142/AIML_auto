import torch
import onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxconverter_common import auto_mixed_precision

# Path to the trained model
model_path = "bert_tc_model/checkpoint-3126"

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Create a dummy input
dummy_input_text = "This is a dummy input"
dummy_input = tokenizer.encode(dummy_input_text, return_tensors="pt")

# Export the model to ONNX format
onnx_model_path = "bert_tc_model_onnx.onnx"
torch.onnx.export(
    model,                                          # model being run
    dummy_input,                                    # model input
    onnx_model_path,                                # where to save the model
    export_params=True,                             # store the trained parameter weights inside the model file
    opset_version=11,                               # ONNX version to export model to
    do_constant_folding=True,                       # constant folding for optimizations
    input_names=['input'],                          # ONNX model input name
    output_names=['output'],                        # ONNX model output name
    dynamic_axes={'input': {0: 'batch_size'},       # variable length axes
                  'output': {0: 'batch_size'}}
)

print("Model has been converted to the ONNX format")

# Verify the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model has been validated")

# Convert the ONNX model to mixed precision
# Create a dictionary with input names and dummy input data
test_data = {"input": dummy_input.numpy()}
model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(onnx_model, test_data, rtol=0.20, atol=0.020, keep_io_types=True)

# Save the mixed precision model
onnx_model_mixed_path = "bert_tc_model_mixed.onnx"
onnx.save(model_fp16, onnx_model_mixed_path)

print("Model has been converted to mixed precision and saved")
