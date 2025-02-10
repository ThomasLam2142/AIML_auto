import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import argparse
import tensorflow as tf
import tensorflow_text as text  # Added for CaseFoldUTF8 compatibility

def parse_args():
    parser = argparse.ArgumentParser(description="Run single-text inference using a saved BERT model for sentiment analysis.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="bert_model",
        help="Path to the directory containing the saved BERT model."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This movie was absolutely wonderful!",
        help="A single text review for sentiment analysis."
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "mixed"],
        default="fp32",
        help="Precision mode: 'fp32' (default) or 'mixed' for mixed precision inference."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Enable mixed precision if requested
    if args.precision == "mixed":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Using mixed precision inference (float16).")
    else:
        print("Using standard precision inference (float32).")

    # 1. Load the previously saved model (SavedModel format)
    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # 2. Prepare the input text as a list (model expects a batch-like structure)
    input_text = [args.text]

    # 3. Time the prediction
    start_time = time.time()
    preds = model.predict(input_text)  # shape = (1, 1)

    # Ensure predictions are cast to float32 if using mixed precision
    if args.precision == "mixed":
        preds = tf.cast(preds, tf.float32)

    end_time = time.time()

    # 4. Calculate inference time
    inference_time = end_time - start_time

    # 5. Interpret and print result
    score = preds[0][0]  # Sigmoid score for a single sample
    sentiment_label = "positive" if score >= 0.5 else "negative"
    print(f"Input text: '{args.text}'")
    print(f"Predicted sentiment score: {score:.4f} ({sentiment_label})")
    print(f"Inference Time: {inference_time:.4f} seconds")

if __name__ == "__main__":
    main()