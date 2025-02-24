import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
args = parser.parse_args()

# Initialize MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print("Number of devices:", strategy.num_replicas_in_sync)

# Set the original desired global batch size.
original_global_batch_size = 32
num_gpus = strategy.num_replicas_in_sync

# Adjust the global batch size if it's not evenly divisible by the number of GPUs.
if original_global_batch_size % num_gpus != 0:
    adjusted_global_batch_size = (original_global_batch_size // num_gpus) * num_gpus
    print(f"Global batch size {original_global_batch_size} is not divisible by {num_gpus} GPUs. "
          f"Adjusting to {adjusted_global_batch_size}.")
    global_batch_size = adjusted_global_batch_size
else:
    global_batch_size = original_global_batch_size

print(f"Using global batch size: {global_batch_size}")

# Enable mixed precision if --amp is passed
if args.amp:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
    print("Mixed precision training enabled.")

image_size = [224, 224]
num_classes = 10  # CIFAR-10 has 10 classes

# Function to preprocess CIFAR-10 dataset with ImageNet normalization
def preprocess(image, label):
    image = tf.image.resize(image, image_size) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet normalization
    return image, label

# Load CIFAR-10 dataset
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10', split=['train', 'test'], as_supervised=True, with_info=True
)

AUTOTUNE = tf.data.AUTOTUNE

# Shuffle the training dataset to help with convergence
train_dataset = ds_train \
    .map(preprocess, num_parallel_calls=AUTOTUNE) \
    .shuffle(10_000) \
    .batch(global_batch_size, drop_remainder=True) \
    .prefetch(AUTOTUNE)

val_dataset = ds_test \
    .map(preprocess, num_parallel_calls=AUTOTUNE) \
    .batch(global_batch_size, drop_remainder=True) \
    .prefetch(AUTOTUNE)

with strategy.scope():
    # Load ResNet50 without top layers
    base_model = ResNet50(input_shape=image_size + [3], weights='imagenet', include_top=False)
    base_model.trainable = False  # Freeze pre-trained layers

    # Add classification head
    x = Flatten()(base_model.output)
    # Use dtype=tf.float32 to avoid NaNs in mixed-precision final logits/softmax
    output_layer = Dense(num_classes, activation='softmax', dtype=tf.float32)(x)

    # Create final model
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

# Define callbacks
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    './resnet50_cifar10_best.keras', save_best_only=True, monitor='val_loss'
)

# Start training
start_time = time.time()
result = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,
    callbacks=[model_checkpoint]
)
end_time = time.time()

print(f"Training time: {end_time - start_time} seconds")

# Save final model
model.save('./resnet50_cifar10_final.keras')