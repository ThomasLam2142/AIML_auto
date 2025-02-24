import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import argparse
import numpy as np
import tensorflow as tf

# 1) Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--precision",
    default="fp32",
    choices=["fp32", "mixed"],
    help="Precision mode for inference (fp32 or mixed)"
)
parser.add_argument(
    "--image_path",
    default="test_images/airplane.jpg",
    help="Path to the image to classify"
)
args = parser.parse_args()

# 2) Set global policy (precision) for inference if using mixed_float16
if args.precision == "mixed":
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy("mixed_float16")
    print("Using mixed_float16 precision for inference.")
else:
    print("Using float32 precision for inference.")

# Load the trained model
model = tf.keras.models.load_model('./resnet50_cifar10_final.keras')
print("Model loaded successfully.")

# CIFAR-10 class names (in order)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Define the image size as used during training (224x224)
IMAGE_SIZE = (224, 224)

def preprocess_image(image_path, image_size=IMAGE_SIZE):
    # Load and resize the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    # Convert image to a NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Scale pixel values to [0,1]
    img_array = img_array / 255.0
    # Apply ImageNet normalization
    img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Preprocess the selected image
image_path = args.image_path
print(f"Classifying image: {image_path}")
img_input = preprocess_image(image_path)

# 3) Measure inference time
start_time = time.time()
predictions = model.predict(img_input)
end_time = time.time()

predicted_index = np.argmax(predictions, axis=1)[0]
predicted_label = class_names[predicted_index]

print(f"Predicted Class: {predicted_label}")
print(f"Inference Time: {end_time - start_time:.4f} seconds")