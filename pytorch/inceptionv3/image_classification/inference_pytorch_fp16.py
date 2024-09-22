import torch
from torchvision import transforms
import time
from PIL import Image

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Load the pre-trained InceptionV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
model.eval()

# Convert the model to FP16
model.half()

# Pre-process image for inference - sample execution (requires torchvision)
filename = 'dog.jpg'
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

# Convert input tensor to FP16
input_tensor = input_tensor.half()

input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
else:
    print("GPU unavailable. Defaulting to the CPU.")

# Warm-up run to load model and data onto GPU first
with torch.no_grad():
    output = model(input_batch)

# Perform inference and record time
latency = []
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    output = model(input_batch)

torch.cuda.synchronize()
end = time.time()
latency.append(end - start)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# Ensure to keep this operation in FP32 for numerical stability
probabilities = torch.nn.functional.softmax(output[0].float(), dim=0)

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print("PyTorch Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))