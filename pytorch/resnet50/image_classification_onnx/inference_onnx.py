import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import onnxruntime
import time
import argparse

# Parse command-line arguments for precision
parser = argparse.ArgumentParser(description="Precision options for InceptionV3 inference")
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "mixed"],
    default="fp32",
    help="Set the precision level for inference: fp32, fp16, mixed"
)
parser.add_argument(
    "--ep",
    type=str,
    choices=["rocm", "migx", "cuda", "openvino"],
    default="rocm",
    help="Set the execution provider for inference: rocm, migx, cuda, openvino, cpu"
)
args = parser.parse_args()

# Set the execution provider
execution_provider = None
if args.ep == "rocm":
    execution_provider = "ROCMExecutionProvider"
elif args.ep == "migx":
    execution_provider = "MIGraphXExecutionProvider"
elif args.ep == "cuda":
    execution_provider = "CUDAExecutionProvider"
elif args.ep == "openvino":
    execution_provider = "OpenVINOExecutionProvider"
elif args.ep == "cpu":
    execution_provider = "CPUExecutionProvider"
    
# Set the model
model_name = None
if args.precision == "fp32":
    model_name = "resnet50_model.onnx"
elif args.precision == "fp16":
    model_name = "resnet50_model_fp16.onnx"
elif args.precision == "mixed":
    model_name = "resnet50_model_mixed.onnx"

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

# Adjust tensor size for model (all precisions)
input_batch = input_tensor.unsqueeze(0)

# Move the input and model to the GPU
print("GPU Availability: ", torch.cuda.is_available())
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
else:
    print("GPU is not available!")

# If fp16, convert the input tensor to an fp16 numpy array
if args.precision == "fp16":
    if torch.cuda.is_available():
        input_batch = input_batch.to(torch.float16)
    else:
        print("GPU is not available!")

# Set Execution Provider for ONNX Runtime
session = onnxruntime.InferenceSession(model_name, providers=[execution_provider])

# Warm up run to load model and data onto the GPU
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
