import torch
import torchvision.models as models
import torch.onnx
import onnx
from onnx import helper
import tensorflow as tf
from torch.autograd import Variable
from onnx_tf.backend import prepare

# Download the pre-trained ResNet50 model
pytorch_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
pytorch_model.eval()

# Dummy input data
dummy_input = Variable(torch.randn(1, 3, 224, 224))

# Convert PyTorch model to ONNX
onnx_path = "resnet50_model.onnx"
torch.onnx.export(pytorch_model, dummy_input, onnx_path, verbose=True)

print("Conversion completed successfully.")

