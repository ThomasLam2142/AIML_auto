import argparse
import onnxruntime as ort
import torch
from transformers import AutoTokenizer
import time

# Parse command-line arguments for precision and execution provider
parser = argparse.ArgumentParser(description="Inference script for BERT-based multiple choice with ONNX models")
parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "mixed", "int8"], default="fp32",
                    help="Set the precision level for inference: fp32, fp16, mixed, int8")
parser.add_argument("--ep", type=str, choices=["rocm", "migx", "cuda", "openvino", "cpu"], default="rocm",
                    help="Set the execution provider for inference: rocm, migx, cuda, openvino")
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
    "fp32": "bert_mc_model.onnx",
    "fp16": "bert_mc_model_fp16.onnx",
    "mixed": "bert_mc_model_mixed.onnx",
    "int8": "bert_mc_model_int8.onnx"
}
model_name = model_map[args.precision]

# Define prompt and candidate answers
prompt = "France has a bread law, Le Decret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law applies to baguettes."
candidate2 = "The law applies to automotive manufacturing."
candidates = [candidate1, candidate2]

# Load tokenizer and tokenize inputs
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)

# Prepare inputs for ONNX Runtime
ort_inputs = {k: v.unsqueeze(0).numpy() for k, v in inputs.items()}

# Load ONNX model and set up the inference session
ort_session = ort.InferenceSession(model_name, providers=[execution_provider])

# Warm-up run
_ = ort_session.run(None, ort_inputs)

# Measure inference time
start_time = time.time()
ort_outs = ort_session.run(None, ort_inputs)
end_time = time.time()
inference_time = (end_time - start_time) * 1000

# Process outputs and determine predicted class
logits = torch.tensor(ort_outs[0])
predicted_class = logits.argmax().item()

# Print results
print(f"Prompt: {prompt}")
print(f"Answer: {candidates[predicted_class]}")
print(f"Inference Time = {inference_time:.2f} ms")
