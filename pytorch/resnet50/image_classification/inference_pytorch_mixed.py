import torch
from torchvision import transforms
import time
from PIL import Image

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Load the pre-trained ResNet50 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
model.eval()

# Pre-process image for inference - sample execution (requires torchvision)
filename = 'cat.jpg'
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
else:
    print("GPU unavailable. Defaulting to the CPU.")

# Warm-up run to load model and data onto GPU first
with torch.no_grad():
    with torch.autocast('cuda', dtype=torch.float16):  # Enable AMP for warm-up
        output = model(input_batch)

# Perform inference and record time
latency = []
torch.cuda.synchronize()  # Ensure GPU is ready before timing
start = time.time()

with torch.no_grad():
    with torch.autocast('cuda', dtype=torch.float16):  # Enable AMP for inference
        output = model(input_batch)

torch.cuda.synchronize()  # Ensure all GPU work is done before stopping the timer
end = time.time()
latency.append(end - start)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print("PyTorch Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))