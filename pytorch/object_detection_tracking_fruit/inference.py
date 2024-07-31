import os
import torch
from PIL import Image
from torchvision import transforms
from utils.model import get_object_detection_model
from utils.visualization import plot_img_bbox, apply_nms

# Function to load the model, apply trained weights, and move it to the GPU
def load_model(model_path, num_classes, device):
    model = get_object_detection_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def main():
    model_path = os.path.join('models', 'model.pth')
    num_classes = 4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path, num_classes, device)
    print("Model loaded and set to evaluation mode.")
    
    # Load  image
    image_path = "dataset/sample/apple.jpg"
    original_image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(original_image).to(device)
    
    # Model inference
    with torch.no_grad():
        prediction = model([img])[0]
    
    # Apply NMS
    nms_prediction = apply_nms(prediction, iou_thresh=0.01)
    
    # Plot the image with bounding boxes
    plot_img_bbox(original_image, nms_prediction)

if __name__ == "__main__":
    main()
