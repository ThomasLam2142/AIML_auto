import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Read the categories
with open("car_brands.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Define the image transformation (should match the training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model (make sure the number of classes matches)
model = models.resnet50(pretrained=False)
num_classes = len(categories)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load('./resnet50_car.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Prepare the image
def prepare_image(img_path):
    image = Image.open(img_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Perform inference
test_image = './test/porsche.jpg'
input_img = prepare_image(test_image)

with torch.no_grad():
    output = model(input_img)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

print(f"The predicted car is: {categories[predicted_class]}")
