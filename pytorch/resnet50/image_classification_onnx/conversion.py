import argparse
import numpy as np
import torch
import torchvision.models as models
import torch.onnx
import onnx
import onnxruntime as ort
from onnxconverter_common import auto_mixed_precision
from onnxconverter_common import float16
from torch.autograd import Variable

# This script converts a PyTorch ResNet50 FP32 model to ONNX FP32, then to ONNX FP16, and ONNX mixed precision 

# Parse command-line arguments for precision
parser = argparse.ArgumentParser(description="Precision options for ResNet50 inference")
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "mixed"],
    default="fp32",
    help="Set the precision level for inference: fp32, fp16, mixed"
)
args = parser.parse_args()

if args.precision == "fp32":
    # Download the pre-trained ResNet50 model
    pytorch_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    pytorch_model.eval()

    # Dummy input data
    dummy_input = Variable(torch.randn(1, 3, 224, 224))

    # Convert PyTorch model to ONNX
    onnx_path = "resnet50_model.onnx"
    torch.onnx.export(pytorch_model, dummy_input, onnx_path, verbose=True)
    print("Conversion to FP32 completed successfully.")
    
elif args.precision == "fp16":
    try:
        model = onnx.load("resnet50_model.onnx")
        model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False, disable_shape_infer=False, op_block_list=None, node_block_list=None)
        onnx.save(model_fp16, "resnet50_model_fp16.onnx")
        print("Conversion to FP16 completed successfully")
    except FileNotFoundError:
        print("Convert to fp32 first before proceeding with fp16")
        
elif args.precision == "mixed":
    try:
        # Define the shape of the input tensor for ResNet50
        input_shape = (1, 3, 224, 224)

        # Create a sample input tensor in FP16, convert it to a tensor, and construct a dictionary for it
        test_input = np.random.rand(*input_shape).astype(np.float32)
        test_data = torch.tensor(test_input)
        input_dict = {'input.1': test_data.numpy()}

        # Convert the model
        model = onnx.load("resnet50_model.onnx")
        model_mixed = auto_mixed_precision.auto_convert_mixed_precision(model, input_dict, rtol=0.03, atol=0.003, keep_io_types=True)
        onnx.save(model_mixed, "resnet50_model_mixed.onnx")
    except FileNotFoundError:
        print("Convert to fp32 first before proceeding with mixed precision")