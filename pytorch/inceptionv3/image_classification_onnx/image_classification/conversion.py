import torch
import torchvision.models as models
import torch.onnx
from torch.autograd import Variable

# Download the pre-trained ResNet50 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()

# Dummy input data for InceptionV3 (input size: 299x299)
dummy_input = Variable(torch.randn(1, 3, 299, 299))

# Convert PyTorch model to ONNX
onnx_path = "inceptionv3_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

print("Conversion completed successfully.")

