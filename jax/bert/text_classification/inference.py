import time
import argparse
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxBertForSequenceClassification
import numpy as np

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "fp16", "mixed"], default="fp32", help="Select precision level for inference")
    return parser.parse_args()

# Load trained model and tokenizer
model = FlaxBertForSequenceClassification.from_pretrained("bert_tc")
tokenizer = AutoTokenizer.from_pretrained("bert_tc")

# Tokenization function
def tokenize_input(text, max_length=128):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )

# Inference function with JIT outside
@jax.jit
def model_infer(params, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask, params=params, train=False).logits
    probabilities = jax.nn.softmax(logits, axis=-1)
    prediction = jnp.argmax(probabilities, axis=-1)
    confidence = probabilities[0, prediction]
    return prediction, confidence

if __name__ == "__main__":
    args = parse_args()
    
    text = "This movie is great!"  
    inputs = tokenize_input(text)  

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    params = model.params 

    # Warm-up call to compile JIT
    _ = model_infer(params, input_ids, attention_mask)
    jax.block_until_ready(_)  # Ensures JAX finishes execution before timing

    # Measure inference time
    start_time = time.time()
    prediction, confidence = model_infer(params, input_ids, attention_mask)
    jax.block_until_ready(prediction)  # Ensures JAX execution completes before stopping timer
    end_time = time.time()
    inference_time = end_time - start_time

    # Output prediction
    label = prediction.item()
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    readable_label = label_map[label]
    confidence_score = confidence.item()

    print(f"Precision: {args.precision}")
    print(f"Predicted label: {readable_label} ({prediction}) with confidence {confidence_score:.2f}")
    print(f"Inference Time: {inference_time:.4f} seconds")