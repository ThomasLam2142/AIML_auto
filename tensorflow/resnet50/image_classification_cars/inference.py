import numpy as np
import tensorflow as tf

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

# Change this variable to point to the image you want to classify
image_path = "test_images/airplane.jpg"

print(f"Classifying image: {image_path}")
# Preprocess the selected image
img_input = preprocess_image(image_path)

# Run inference on the preprocessed image
predictions = model.predict(img_input)
predicted_index = np.argmax(predictions, axis=1)[0]
predicted_label = class_names[predicted_index]

print(f"Predicted Class: {predicted_label}")