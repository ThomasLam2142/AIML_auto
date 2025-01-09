import argparse
import time
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Parse command-line arguments for precision
parser = argparse.ArgumentParser(description="Precision options for InceptionV3 inference")
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "mixed"],
    default="fp32",
    help="Set the precision level for inference: fp32, fp16, mixed"
)
parser.add_argument(
    "--ep",
    type=str,
    choices=["rocm", "migx", "cuda", "openvino"],
    default="rocm",
    help="Set the execution provider for inference: rocm, migx, cuda, openvino, cpu"
)
args = parser.parse_args()

# Set the execution provider
execution_provider = None
if args.ep == "rocm":
    execution_provider = "ROCMExecutionProvider"
elif args.ep == "migx":
    execution_provider = "MIGraphXExecutionProvider"
elif args.ep == "cuda":
    execution_provider = "CUDAExecutionProvider"
elif args.ep == "openvino":
    execution_provider = "OpenVINOExecutionProvider"
elif args.ep == "cpu":
    execution_provider = "CPUExecutionProvider"

# Set the model
model_name = None
if args.precision == "fp32":
    model_name = "bert_tc_model.onnx"
elif args.precision == "fp16":
    model_name = "bert_tc_model_fp16.onnx"
elif args.precision == "mixed":
    model_name = "bert_tc_model_mixed.onnx"

# Text to be analyzed
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert_tc_model/checkpoint-1564")

# Tokenize the input text and return tensors
inputs = tokenizer(text, return_tensors="np", max_length=7, padding='max_length', truncation=True)

# Load the ONNX model
onnx_model_path = model_name
session = ort.InferenceSession(onnx_model_path, providers=[execution_provider])

# Prepare ONNX input
inputs_onnx = {
    'input': inputs['input_ids']
}

# Warmup run (not timed)
_ = session.run(None, inputs_onnx)

# Measure inference time
start_time = time.time()
outputs = session.run(None, inputs_onnx)
end_time = time.time()

# Calculate elapsed time
inference_time = end_time - start_time

# Get the logits from the output
logits = outputs[0]

# Assuming logits shape is [batch_size, num_classes]
num_classes = logits.shape[-1]

# Load model configuration to get labels
config = AutoConfig.from_pretrained("bert_tc_model/checkpoint-1564")
id2label = config.id2label

# Get the class with the highest probability and map the output to a label
predicted_class_id = np.argmax(logits, axis=-1).item()
output_label = id2label[predicted_class_id]

# Print the result
print(output_label)
print(f"Inference Time = {inference_time:.4f} seconds")