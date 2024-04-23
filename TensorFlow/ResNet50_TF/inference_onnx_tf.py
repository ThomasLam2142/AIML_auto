import numpy as np
import onnxruntime
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Pre-process the image for ResNet-50 inference
filename = 'cat.jpg'  # change to your filename
input_image = Image.open(filename)
input_image = input_image.resize((224, 224))  # Resize to 224x224

# Convert to NumPy array and preprocess
input_array = np.asarray(input_image)
input_array = preprocess_input(input_array)  # Preprocess the input image

# Convert the input to a tensor
input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

# Set Execution Provider for ONNX Runtime
session_fp32 = onnxruntime.InferenceSession("resnet50_model.onnx", providers=["MIGraphXExecutionProvider"])

# warm up run to load model and data
_ = session_fp32.run(None, {'input_1': input_tensor.numpy()})

# Run inference with ONNX Runtime
latency = []
tf.config.experimental.set_synchronous_execution(True)
start = time.time()
ort_outputs = session_fp32.run(None, {'input_1': input_tensor.numpy()})[0]
end = time.time()
tf.config.experimental.set_synchronous_execution(False)
latency.append(end - start)

# Process ONNX Runtime outputs
output = ort_outputs.flatten()
output = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))  # softmax
top5_catid = np.argsort(-output)[:5]

# Show top categories per image
for i in range(len(top5_catid)):
    print(categories[top5_catid[i]], output[top5_catid[i]])

print("ONNX Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))