import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Workaround for tensorflow-hub and Keras 3 compatibility issue.

import time
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds

def parse_args():
    parser = argparse.ArgumentParser(description="BERT training with multi-GPU and mixed precision.")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision for training (float16 on GPUs). (Default OFF)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs. (Default 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size. (Default 32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate. (Default 3e-5)"
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="bert_model",
        help="Directory where the trained model will be saved."
    )
    return parser.parse_args()

def load_imdb_datasets(batch_size=32):
    """
    Load and prepare the IMDB reviews dataset for training and validation.
    """
    (train_ds, val_ds), ds_info = tfds.load(
        "imdb_reviews",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        with_info=True,
    )

    # Shuffle, batch, prefetch for performance
    train_ds = (
        train_ds
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds

def build_classifier_model(preprocess_url, encoder_url, dropout_rate=0.1):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(preprocess_url, name="bert_preprocessing")
    encoder_layer = hub.KerasLayer(encoder_url, trainable=True, name="bert_encoder")

    # BERT preprocessing
    preprocessed_text = preprocessing_layer(text_input)

    # BERT encoder
    outputs = encoder_layer(preprocessed_text)
    pooled_output = outputs["pooled_output"]  # [CLS] token representation

    # Classifier
    dropout_layer = tf.keras.layers.Dropout(dropout_rate)(pooled_output)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(dropout_layer)

    model = tf.keras.Model(inputs=text_input, outputs=output)
    return model

def main():
    args = parse_args()

    # 1. Optional: use mixed precision if requested and supported
    if args.amp:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Using mixed precision (float16).")

    # 2. Create a MirroredStrategy for multi-GPU training
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices (GPUs): {strategy.num_replicas_in_sync}")

    # 3. Hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    export_dir = args.export_dir

    # 4. Load dataset
    train_ds, val_ds = load_imdb_datasets(batch_size=batch_size)

    bert_preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

    # 5. Build & compile model under the strategy scope
    with strategy.scope():
        model = build_classifier_model(
            preprocess_url=bert_preprocess_url,
            encoder_url=bert_encoder_url,
            dropout_rate=0.1
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        )

    # 6. Train the model (timing the training)
    start_time = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    end_time = time.time()

    # 7. Evaluate
    loss, accuracy = model.evaluate(val_ds)
    print(f"\n\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Print total training time
    total_mins = (end_time - start_time)/60
    print(f"Total training time: {total_mins:.2f} minutes\n")

    # 8. Export/Save the model
    print(f"Exporting model to '{export_dir}' ...")
    model.save(export_dir)
    print(f"Model saved successfully to '{export_dir}'\n")

if __name__ == "__main__":
    main()