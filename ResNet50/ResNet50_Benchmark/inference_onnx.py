import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import onnxruntime
import time

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Pre-process the image for ResNet-50 inference
filename = 'cat.jpg'  # change to your filename
input_image = Image.open(filename)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input to GPU for speed if available
print("GPU Availability: ", torch.cuda.is_available())
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
else:
    print("GPU unavailable. Defaulting to CPU.")

# Set Execution Provider for ONNX Runtime
session_fp32 = onnxruntime.InferenceSession("resnet50_model.onnx", providers=['MIGraphXExecutionProvider'])

# warm up run to load model and data onto GPU first
if torch.cuda.is_available():
    input_batch = input_batch.cpu()
ort_outputs = session_fp32.run([], {'input.1': input_batch.numpy()})[0]

# Run inference with ONNX Runtime
latency = []
torch.cuda.synchronize()
start = time.time()
if torch.cuda.is_available():
    input_batch = input_batch.cpu()
ort_outputs = session_fp32.run([], {'input.1': input_batch.numpy()})[0]
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