import os
import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms as T

def inference(
    image,
    checkpoint_dir,
    checkpoint_name,
):

    # Load classes
    with open("classes.txt", "r") as f:
        classes = [s.strip() for s in f.readlines()]

    # Preprocess images
    transforms = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
    ])

    # Load pretrained ResNet50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    num_classes = len(classes)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the trained model weights
    model_dir = os.path.join("checkpoints", "baseline", "best_model.pth")
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    # Prepare image
    def prepare_image(img_path):
        image = Image.open(img_path)
        image = transforms(image)
        image = image.unsqueeze(0)
        return image

    test_image = "test.jpg"
    input = prepare_image(test_image)

    # Load image and model to GPU
    if torch.cuda.is_available():
        input = input.to('cuda')
        model.to('cuda')
    else:
        raise ValueError("GPU not detected")

    # Inference
    with torch.no_grad():
        output = model(input)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        prediction_idx = torch.argmax(probabilities).item()
        prediction_class = classes[prediction_idx]


    return {
        prediction_class
    }