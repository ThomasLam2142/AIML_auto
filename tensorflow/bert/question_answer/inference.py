import argparse
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script with precision selection.")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "mixed"],
        default="fp32",
        help="Precision mode: 'fp32' (default) or 'mixed' (AMP - float16 & float32)."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./distilbert_qa",
        help="Path to the trained model directory."
    )
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Apply precision settings
if args.precision == "fp32":
    print("Using FP32 (Full Precision)")
    tf.keras.mixed_precision.set_global_policy("float32")
elif args.precision == "mixed":
    print("Using Mixed Precision (AMP - float16 & float32)")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Load model and tokenizer
print(f"Loading model from: {args.model_dir}")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = TFAutoModelForQuestionAnswering.from_pretrained(args.model_dir)

# Define test question and context
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 natural languages and 13 programming languages."

# Tokenize input
inputs = tokenizer(question, context, return_tensors="tf")

# Measure inference time
start_time = time.time()

# Run inference
outputs = model(**inputs)

# Get predicted answer tokens
answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
predicted_answer_tokens = inputs["input_ids"][0, answer_start_index : answer_end_index + 1]

# Decode final answer
predicted_answer = tokenizer.decode(predicted_answer_tokens)

# Measure end time
end_time = time.time()
inference_time = end_time - start_time

# Print results
print(f"\nPredicted Answer: {predicted_answer}")
print(f"Inference Time: {inference_time:.4f} seconds")