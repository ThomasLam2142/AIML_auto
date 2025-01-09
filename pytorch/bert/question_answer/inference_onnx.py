import argparse
import time
import onnxruntime as ort
from transformers import AutoTokenizer

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
    model_name = "distilbert_qa_model.onnx"
elif args.precision == "fp16":
    model_name = "distilbert_qa_model_fp16.onnx"
elif args.precision == "mixed":
    model_name = "distilbert_qa_model_mixed.onnx"

# Create a question and context
question = "What is my name?"
context = "My name is Mern and I live in Barcelona"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert_qa_model/checkpoint-500")

# Tokenize the input text and return tensors
inputs = tokenizer(question, context, return_tensors="pt")

# Load the ONNX model
onnx_path = model_name
ort_session = ort.InferenceSession(onnx_path, providers=[execution_provider])

# Prepare inputs for ONNX model
ort_inputs = {
    'input_ids': inputs['input_ids'].numpy(),
    'attention_mask': inputs['attention_mask'].numpy()
}

# Warmup run
ort_session.run(None, ort_inputs)

# Measure inference time
start_time = time.time()
ort_outputs = ort_session.run(None, ort_inputs)
end_time = time.time()

# Calculate and format inference time
inference_time = end_time - start_time

# Get the start and end logits
start_logits = ort_outputs[0]
end_logits = ort_outputs[1]

# Get the most likely beginning and end of answer
answer_start_index = start_logits.argmax()
answer_end_index = end_logits.argmax()

# Get the answer tokens and decode them
predict_answer_tokens = inputs['input_ids'][0, answer_start_index:answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens)

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Inference Time = {inference_time:.4f} seconds")