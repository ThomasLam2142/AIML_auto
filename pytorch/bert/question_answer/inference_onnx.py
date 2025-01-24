import argparse
import time
import onnxruntime as ort
from transformers import AutoTokenizer

# Parse command-line arguments for precision and execution provider
parser = argparse.ArgumentParser(description="Inference script for BERT-based Q&A with ONNX models")
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
    "fp32": "distilbert_qa_model.onnx",
    "fp16": "distilbert_qa_model_fp16.onnx",
    "mixed": "distilbert_qa_model_mixed.onnx",
    "int8": "distilbert_qa_model_int8.onnx"
}
model_name = model_map[args.precision]

# Load tokenizer and prepare input
tokenizer = AutoTokenizer.from_pretrained("distilbert_qa_model/checkpoint-500")
question = "Where does Mern live?"
context = "My name is Mern and I live in Barcelona"
inputs = tokenizer(question, context, return_tensors="pt")

# Load ONNX model and set up the inference session
ort_session = ort.InferenceSession(model_name, providers=[execution_provider])

# Prepare ONNX inputs
ort_inputs = {
    "input_ids": inputs["input_ids"].numpy(),
    "attention_mask": inputs["attention_mask"].numpy()
}

# Warm-up run
ort_session.run(None, ort_inputs)

# Measure inference time
start_time = time.time()
ort_outputs = ort_session.run(None, ort_inputs)
end_time = time.time()

# Process outputs
start_logits, end_logits = ort_outputs
answer_start_index = start_logits.argmax()
answer_end_index = end_logits.argmax()
predict_answer_tokens = inputs["input_ids"][0, answer_start_index:answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens)

# Convert time to ms
time_ms = (end_time - start_time) * 1000

# Print results
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Inference Time = {time_ms:.2f} ms")