import argparse
import time
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Parse command-line arguments for precision and execution provider
parser = argparse.ArgumentParser(description="Inference script for BERT-based sentiment analysis with ONNX models")
parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "mixed", "int8"], default="fp32",
                    help="Set the precision level for inference: fp32, fp16, mixed, int8")
parser.add_argument("--ep", type=str, choices=["rocm", "migx", "cuda", "openvino", "cpu"], default="rocm",
                    help="Set the execution provider for inference: rocm, migx, cuda, openvino, cpu")
args = parser.parse_args()

# Map execution providers
execution_provider_map = {
    "rocm": "ROCMExecutionProvider",
    "migx": "MIGraphXExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider"
}
execution_provider = execution_provider_map[args.ep]

# Map precision levels to model filenames
model_map = {
    "fp32": "bert_tc_model.onnx",
    "fp16": "bert_tc_model_fp16.onnx",
    "mixed": "bert_tc_model_mixed.onnx",
    "int8": "bert_tc_model_int8.onnx"
}
model_name = model_map[args.precision]

# Input text for analysis
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# Load tokenizer and tokenize the input text
tokenizer = AutoTokenizer.from_pretrained("bert_tc_model/checkpoint-782")
inputs = tokenizer(text, return_tensors="np", max_length=7, padding='max_length', truncation=True)

# Load ONNX model and set up the inference session
onnx_model_path = model_name
session = ort.InferenceSession(onnx_model_path, providers=[execution_provider])

# Prepare ONNX inputs
inputs_onnx = {"input": inputs["input_ids"]}

# Warm-up run (not timed)
session.run(None, inputs_onnx)

# Measure inference time
start_time = time.time()
outputs = session.run(None, inputs_onnx)
end_time = time.time()
time_ms = (end_time - start_time) * 1000

# Process outputs
logits = outputs[0]
predicted_class_id = np.argmax(logits, axis=-1).item()

# Load model configuration and map output to label
config = AutoConfig.from_pretrained("bert_tc_model/checkpoint-782")
id2label = config.id2label
output_label = id2label[predicted_class_id]

# Print the result
print(f"Sentiment: {output_label}")
print(f"Inference Time = {time_ms:.2f} ms")