import argparse
import torch
import time
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.cuda.amp import autocast

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run inference with different precisions.")
parser.add_argument('--precision', choices=['fp32', 'fp16', 'mixed'], default='fp32',
                    help="Precision mode for inference: 'fp32' (default), 'fp16', or 'mixed'")
args = parser.parse_args()

# Define the question and context
question = "Where do I live?"
context = "My name is Mern and I live in Barcelona"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert_qa_model/checkpoint-750")
inputs = tokenizer(question, context, return_tensors="pt")

model = AutoModelForQuestionAnswering.from_pretrained("distilbert_qa_model/checkpoint-750")

# Move model and inputs to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {key: val.to(device) for key, val in inputs.items()}

# Set precision mode
if args.precision == 'fp16':
    model.half()  # Convert model to FP16
    # Convert only non-integer inputs to FP16, keep input_ids as LongTensor
    inputs = {key: (value.half() if key != 'input_ids' else value) for key, value in inputs.items()}
elif args.precision == 'mixed':
    # No need to convert model and inputs to FP16; use autocast during inference
    pass

# Synchronize and start the timer
if device.type == 'cuda':
    torch.cuda.synchronize()

start_time = time.time()

# Perform inference
with torch.no_grad():
    if args.precision == 'mixed':
        with autocast():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

# Synchronize and end the timer
if device.type == 'cuda':
    torch.cuda.synchronize()

end_time = time.time()

# Get the answer
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs['input_ids'][0, answer_start_index: answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens)

# Print the question and the predicted answer
print(f"Question: {question}")
print(f"Answer: {answer}")

# Calculate and print the inference time in seconds
inference_duration = end_time - start_time
print(f"Inference Time = {inference_duration:.4f} seconds")