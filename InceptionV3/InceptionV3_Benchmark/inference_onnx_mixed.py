import torch
from torchvision import transforms
from PIL import Image
import onnxruntime
import time
import numpy as np

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Pre-process the image for InceptionV3 inference
filename = 'dog.jpg'
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input to GPU for speed if available
print("GPU Availability: ", torch.cuda.is_available())
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
else:
    print("GPU unavailable. Defaulting to CPU")

# Set Execution Provider for ONNX Runtime
session_fp32 = onnxruntime.InferenceSession("inceptionv3_model_mixed.onnx", providers=['ROCMExecutionProvider'])

# warm up run to load model and data onto gpu first
if torch.cuda.is_available():
    input_batch = input_batch.cpu()
ort_outputs = session_fp32.run([], {'x.1': input_batch.numpy()})[0]

# Run inference with ONNX Runtime
latency = []
torch.cuda.synchronize()
start = time.time()
if torch.cuda.is_available():
    input_batch = input_batch.cpu()
ort_outputs = session_fp32.run([], {'x.1': input_batch.numpy()})[0]
torch.cuda.synchronize()
end = time.time()
latency.append(end - start)

# Process ONNX Runtime outputs
output = ort_outputs.flatten()
output = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))  # softmax
top5_catid = np.argsort(-output)[:5]

# Show top categories per image
for i in range(len(top5_catid)):
    print(categories[top5_catid[i]], output[top5_catid[i]])
    
print("ONNX Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))