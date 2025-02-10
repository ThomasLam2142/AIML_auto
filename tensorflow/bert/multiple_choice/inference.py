import time
import argparse
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForMultipleChoice

def preprocess_multiple_choice(context, question_header, endings, tokenizer):
    # Repeat context for each ending
    first_sentences = [context] * 4

    # Append question_header to each ending
    second_sentences = [f"{question_header} {end}" for end in endings]

    # Tokenize
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding=True,  # For a single sample, padding is optional
        return_tensors="tf"
    )

    # Expand dims so shape is [1, 4, seq_len] instead of [4, seq_len]
    for key in tokenized:
        tokenized[key] = tf.expand_dims(tokenized[key], axis=0)

    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str, choices=["fp32", "mixed"], default="fp32",
                        help="Precision mode: 'fp32' (default) or 'mixed' for mixed precision.")
    args = parser.parse_args()

    # Enable mixed precision if requested
    if args.precision == "mixed":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Using mixed precision inference (float16).")
    else:
        print("Using standard precision inference (float32).")

    # Load model
    model_dir = "./bert_mc_model"

    # Example input
    context = "He went to the store for some dairy"
    question_header = "He wanted to buy"
    endings = [
        "some milk",
        "the entire store",
        "nothing at all",
        "bread"
    ]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TFAutoModelForMultipleChoice.from_pretrained(model_dir)

    # Measure inference time
    start_time = time.time()

    # Preprocess
    inputs = preprocess_multiple_choice(context, question_header, endings, tokenizer)

    # Perform forward pass
    outputs = model(inputs)  # logits shape: [1, 4]
    logits = outputs.logits

    # Ensure logits are cast to float32 in mixed precision mode
    if args.precision == "mixed":
        logits = tf.cast(logits, tf.float32)

    predicted_class = np.argmax(logits, axis=1)[0]  # best ending index

    end_time = time.time()
    total_inference_time = end_time - start_time

    # Print results
    print(f"Context: {context}")
    print(f"Question: {question_header}")
    print(f"Possible Endings: {endings}")
    print("-----------------------------------")
    print(f"Best ending index: {predicted_class}")
    print(f"Best ending text: '{endings[predicted_class]}'")
    print(f"Inference took {total_inference_time:.4f} seconds")

if __name__ == "__main__":
    main()