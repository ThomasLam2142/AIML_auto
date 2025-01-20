import argparse
import onnxruntime as ort
import torch
from transformers import AutoTokenizer
import time

# Parse command-line arguments for precision
parser = argparse.ArgumentParser(description="Precision options for InceptionV3 inference")
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "mixed", "int8"],
    default="fp32",
    help="Set the precision level for inference: fp32, fp16, mixed, int8"
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
    model_name = "bert_mc_model.onnx"
elif args.precision == "fp16":
    model_name = "bert_mc_model_fp16.onnx"
elif args.precision == "mixed":
    model_name = "bert_mc_model_mixed.onnx"
elif args.precision == "int8":
    model_name = "bert_mc_model_int8.onnx"

# Create a prompt and two candidate answers
prompt = "France has a bread law, Le Decret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law applies to baguettes."
candidate2 = "The law applies to automotive manufacturing."

candidates = [candidate1, candidate2]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Tokenize the input text and return tensors
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)

# Prepare the inputs for ONNX Runtime
ort_inputs = {k: v.unsqueeze(0).numpy() for k, v in inputs.items()}

# Load the ONNX model
ort_session = ort.InferenceSession(model_name, providers=[execution_provider])

# Warmup run
_ = ort_session.run(None, ort_inputs)

# Measure inference time
start_time = time.time()
ort_outs = ort_session.run(None, ort_inputs)
end_time = time.time()

# Get the logits from the output
logits = torch.tensor(ort_outs[0])

# Get the class with the highest probability
predicted_class = logits.argmax().item()
print(f"Prompt: {prompt}")
print(f"Answer: {candidates[predicted_class]}")

# Calculate and display inference time
inference_time = end_time - start_time
print(f"Inference Time = {inference_time:.4f} seconds")