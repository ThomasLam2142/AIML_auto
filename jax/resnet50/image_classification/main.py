import jax
import jax.numpy as jnp
import numpy as np
import time
from transformers import FlaxResNetForImageClassification
from PIL import Image

# Load model
model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Function to preprocess image for ResNet50 model
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    
    # Normalize to ImageNet mean and std
    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Add batch dimension and convert to JAX array
    return jnp.expand_dims(img.transpose(2, 0, 1), axis=0)

# Inference
image_path = "cat.jpg"
input_image = preprocess_image(image_path)

_ = model(input_image)  # warm-up run

start = time.time()
output = model(input_image)
end = time.time()

# Output prediction
logits = output.logits
probabilities = jax.nn.softmax(logits)  # Convert logits to probabilities

top_k = 5
top_k_indices = jnp.argsort(probabilities[0], axis=-1)[::-1][:top_k] 
top_k_probs = probabilities[0][top_k_indices]

# Map indices to labels
imagenet_labels = model.config.id2label
top_k_labels = [imagenet_labels[int(idx)] for idx in top_k_indices]

# Calculate inference time
inference_time = (end - start) * 1000  # Convert to ms

# Print top 5 predictions and their confidence levels
for label, prob in zip(top_k_labels, top_k_probs):
    print(f"{label}, {prob:.4f}")
    
print(f"Inference time: {inference_time:.4f} ms")