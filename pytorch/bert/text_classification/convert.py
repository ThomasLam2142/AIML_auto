import torch
import onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path to the trained model
model_path = "bert_tc_model/checkpoint-3126"

# Load the trained model and tokenzier
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Create a dummy input
input = tokenizer.encode("This is a dummy input", return_tensors="pt")

# Export the model to ONNX format
torch.onnx.export(
    model,                                          # model being run
    input,                                          # model input
    "bert_tc_model_onnx.onnx",                      # where to save the model
    export_params=True,                             # store the trained parameter weights inside the model file
    opset_version=11,                               # ONNX version to export model to
    do_constant_folding=True,                       # constant folding for optimizations
    input_names=['input'],                          # ONNX model input name
    output_names=['output'],                        # ONNX model output name
    dynamic_axes={'input' : {0: 'batch_size'},      # variable length axes
                  'output': {0: 'batch_size'}}
)

print("Model has been converted to the ONNX format")

# Verify the ONNX model
onnx_model = onnx.load("bert_tc_model_onnx.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model has been validated")
