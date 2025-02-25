import argparse
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.cuda.amp import autocast


# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run inference with different precisions.")
parser.add_argument('--precision', choices=['fp32', 'fp16', 'mixed'], default='fp32',
                    help="Precision mode for inference: 'fp32' (default), 'fp16', or 'mixed'")
args = parser.parse_args()

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# Tokenize the text and return tensors
tokenizer = AutoTokenizer.from_pretrained("bert_tc_model/checkpoint-3126")
inputs = tokenizer(text, return_tensors="pt")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("bert_tc_model/checkpoint-3126")

# Move model and inputs to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Set precision mode
if args.precision == 'fp16':
    model.half()  # Convert model to FP16
    # Convert only the attention_mask to FP16, keep input_ids as LongTensor
    inputs = {key: (value.half() if key != 'input_ids' else value) for key, value in inputs.items()}
elif args.precision == 'mixed':
    # No need to convert model and inputs to FP16; use autocast during inference
    pass

# Measure inference time
torch.cuda.synchronize() if device.type == 'cuda' else None  # Synchronize GPU before starting timer
start_time = time.time()

with torch.no_grad():
    if args.precision == 'mixed':
        with autocast():
            logits = model(**inputs).logits
    else:
        logits = model(**inputs).logits

torch.cuda.synchronize() if device.type == 'cuda' else None  # Synchronize GPU after inference
end_time = time.time()

# Get the class with the highest probability and map the output to a label
predicted_class_id = logits.argmax().item()
output = model.config.id2label[predicted_class_id]
print(output)

# Calculate and print inference duration
inference_duration = (end_time - start_time) * 1000
print(f"Inference Time = {inference_duration:.4f} ms")