import argparse
import time

import jax
import jax.numpy as jnp
from transformers import (
    FlaxBertForMultipleChoice,
    BertTokenizerFast
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "mixed"],
        help="Precision to use for inference: fp16, fp32, or mixed (defaults to fp32)."
    )
    return parser.parse_args()

def prepare_input(tokenizer, context, choices, max_length=64):
    contexts = [context] * len(choices)

    # Tokenize
    encoding = tokenizer(
        contexts,
        choices,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='np'
    )
    
    batch = {k: v[None, :] for k, v in encoding.items()}
    return batch

def main():
    args = parse_args()

    # Map CLI argument to actual JAX dtypes
    if args.precision == "fp32":
        model_dtype = jnp.float32
    elif args.precision == "fp16":
        model_dtype = jnp.float16
    elif args.precision == "mixed":
        # Here, "mixed precision" is approximated with bfloat16.
        model_dtype = jnp.bfloat16

    # 1. Load the Flax BERT model for multiple choice
    model_name = "bert-base-uncased"
    model = FlaxBertForMultipleChoice.from_pretrained(model_name, dtype=model_dtype)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # 2. Example question and possible choices
    question = "Jonas went to the store to get some dairy, what did he buy?"
    choices = [
        "Bread",
        "Milk",
        "Eggs",
        "Sugar"
    ]

    # 3. Prepare inputs
    inputs = prepare_input(tokenizer, question, choices)

    # 4. Forward pass with inference timing
    start_time = time.time()
    outputs = model(**inputs)
    logits = outputs.logits  # shape: (batch_size, num_choices)
    # Force computation to complete (useful for accelerator devices)
    logits = logits.block_until_ready()
    end_time = time.time()
    inference_time = end_time - start_time

    # 5. Use jnp.argmax to select the best choice
    predicted_choice = jnp.argmax(logits, axis=-1)
    predicted_index = int(predicted_choice[0])  # batch size is 1 here
    print(f"\nQuestion: {question}")
    print("Choices:")
    for i, c in enumerate(choices):
        print(f"  {i}. {c}")
    print(f"\nPredicted answer: {choices[predicted_index]}")
    print(f"Inference Time: {inference_time:.4f} seconds")

if __name__ == "__main__":
    main()