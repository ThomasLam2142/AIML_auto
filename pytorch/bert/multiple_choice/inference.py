import argparse
import torch
import time
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run inference with different precisions.")
parser.add_argument('--precision', choices=['fp32', 'fp16', 'mixed'], default='fp32',
                    help="Precision mode for inference (default: fp32)")
args = parser.parse_args()

# Create a prompt and two candidate answers
prompt = "France has a bread law, Le Decret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law applies to baguettes."
candidate2 = "The law applies to automotive manufacturing."

candidates = [candidate1, candidate2]

# Tokenize each prompt-candidate pair and return PyTorch tensors
tokenizer = AutoTokenizer.from_pretrained("bert_mc_model/checkpoint-13791")
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)

# Load the model
model = AutoModelForMultipleChoice.from_pretrained("bert_mc_model/checkpoint-13791")

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Convert model and inputs based on precision mode
if args.precision == 'fp16':
    model.half()  # Convert model to FP16
    # Convert only floating-point inputs to FP16, leave 'input_ids' and 'token_type_ids' as LongTensor
    inputs = {
        key: (value.half() if key not in ['input_ids', 'token_type_ids'] else value)
        for key, value in inputs.items()
    }
elif args.precision == 'mixed':
    # No specific action needed for mixed precision setup
    pass

# Move inputs and labels to the appropriate device
inputs = {key: value.to(device) for key, value in inputs.items()}
labels = labels.to(device)

# Synchronize and start the timer
if torch.cuda.is_available():
    torch.cuda.synchronize()

start_time = time.time()

# Perform inference
with torch.no_grad():  # Disable gradient calculations for inference
    if args.precision == 'mixed':
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
    else:
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
    logits = outputs.logits

# Synchronize and end the timer
if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()

# Get the class with the highest probability
predicted_class = logits.argmax().item()
print(f"Prompt: {prompt}")
print(f"Answer: {candidates[predicted_class]}")

# Calculate and print the inference time
inference_duration = end_time - start_time
print(f"Inference Time = {inference_duration:.6f} s")