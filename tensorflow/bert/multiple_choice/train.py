import os
os.environ["TF_USE_LEGACY_KERAS"] ="1"

import argparse
import time
import numpy as np
import tensorflow as tf
import evaluate
from dataclasses import dataclass
from typing import Optional, Union
from datasets import load_dataset
from transformers import (AutoTokenizer, TFAutoModelForMultipleChoice, create_optimizer)
from transformers.tokenization_utils_base import (PreTrainedTokenizerBase, PaddingStrategy)
from transformers.keras_callbacks import KerasMetricCallback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BERT model for multiple-choice QA on the SWAG dataset with multi-GPU support."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs. (Default 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size. (Default 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate. (Default 5e-5)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision for training (float16 on GPUs). (Default OFF)"
    )
    return parser.parse_args()


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        # Flatten all choices for each feature
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="tf",
        )

        # Reshape to [batch_size, num_choices, seq_len]
        batch = {
            k: tf.reshape(v, (batch_size, num_choices, -1))
            for k, v in batch.items()
        }
        batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
        return batch


def preprocess_function(examples, tokenizer, ending_names):
    """
    Preprocess SWAG samples for multiple-choice:
    1. Duplicate sent1 4 times (one for each choice).
    2. Combine sent2 with each possible ending.
    """
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names]
        for i, header in enumerate(question_headers)
    ]

    # Flatten
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True
    )

    # Regroup the tokens
    return {
        k: [v[i : i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }


def compute_metrics(eval_pred):
    """
    Compute accuracy for multiple-choice predictions.
    """
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def main():
    args = parse_args()

    # List GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {len(gpus)} GPUs.")
    else:
        strategy = tf.distribute.get_strategy()
        if len(gpus) == 1:
            print("One GPU detected, using default strategy.")
        else:
            print("No GPU detected, using CPU strategy.")

    # Load the SWAG dataset
    swag = load_dataset("swag", "regular")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    ending_names = ["ending0", "ending1", "ending2", "ending3"]
    tokenized_swag = swag.map(
        lambda x: preprocess_function(x, tokenizer, ending_names),
        batched=True
    )

    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    # Calculate total training steps
    total_train_steps = (len(tokenized_swag["train"]) // args.batch_size) * args.epochs

    # Define everything inside the strategy scope
    with strategy.scope():
        # If AMP is enabled, set mixed precision
        if args.amp:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            # Force LayerNormalization to float32
            tf.keras.layers.LayerNormalization = tf.keras.layers.LayerNormalization(dtype='float32')
            print("Using mixed precision (float16).")

        # Create optimizer & scheduler
        optimizer, schedule = create_optimizer(
            init_lr=args.learning_rate,
            num_warmup_steps=0,
            num_train_steps=total_train_steps
        )

        # Load BERT model for multiple choice
        model = TFAutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")

        # Compile the model with optimizer and metrics
        model.compile(
            optimizer=optimizer,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

    # Prepare TensorFlow datasets
    tf_train_set = model.prepare_tf_dataset(
        tokenized_swag["train"],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )
    tf_validation_set = model.prepare_tf_dataset(
        tokenized_swag["validation"],
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    # Create the evaluation callback
    metric_callback = KerasMetricCallback(
        metric_fn=compute_metrics,
        eval_dataset=tf_validation_set
    )

    # Start training
    print(f"Starting training for {args.epochs} epoch(s)")
    start_time = time.time()
    model.fit(
        x=tf_train_set,
        validation_data=tf_validation_set,
        epochs=args.epochs,
        callbacks=[metric_callback]
    )
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Final evaluation
    eval_loss, eval_accuracy = model.evaluate(tf_validation_set)
    print(f"Final Val Loss: {eval_loss:.4f} | Final Val Accuracy: {eval_accuracy:.4f}")

    # Save the model
    model.save_pretrained("./bert_mc_model")
    tokenizer.save_pretrained("./bert_mc_model")
    print("Model saved to ./bert_mc_model")


if __name__ == "__main__":
    main()