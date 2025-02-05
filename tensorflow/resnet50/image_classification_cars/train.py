import time
import os
import argparse
import tensorflow as tf
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

# Enable mixed precision if --amp is passed
if args.amp:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')
    print("Mixed precision training enabled.")

image_size = [224, 224]
train_folder = "./train_images/train"
val_folder = "./train_images/val"

# Get number of classes from training folder
classes = sorted(os.listdir(train_folder))
classes_num = len(classes)
print(f"Classes found: {classes}, Number of classes: {classes_num}")

# Function to apply one-hot encoding to dataset labels
def preprocess_dataset(dataset, num_classes):
    """ Convert integer labels to one-hot encoding """
    def one_hot_encode(image, label):
        return image, tf.one_hot(label, depth=num_classes)
    
    return dataset.map(one_hot_encode)

# Load dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder, image_size=(224, 224), batch_size=32
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_folder, image_size=(224, 224), batch_size=32
)

# Apply one-hot encoding
train_dataset = preprocess_dataset(train_dataset, classes_num)
val_dataset = preprocess_dataset(val_dataset, classes_num)

# Optimize dataset performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

with strategy.scope():
    # Load ResNet50 without top layers
    base_model = ResNet50(input_shape=image_size + [3], weights='imagenet', include_top=False)

    # Freeze pre-trained layers
    base_model.trainable = False  

    # Add classification head
    x = Flatten()(base_model.output)
    output_layer = Dense(classes_num, activation='softmax', dtype='float32')(x)  # Ensure correct dtype

    # Create final model
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

# Callbacks
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    './resnet50_car_best.keras', save_best_only=True, monitor='val_loss'
)

# Start training
start_time = time.time()
result = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[model_checkpoint]
)
end_time = time.time()

print(f"Training time: {end_time - start_time} seconds")

# Save final model
model.save('./resnet50_car_final.keras')