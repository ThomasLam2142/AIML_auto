#pip install onnxconverter-common before using script

from onnxconverter_common import auto_mixed_precision
import onnx

model = onnx.load("resnet50_model.onnx")
model_fp16 = auto_convert_mixed_precision(model, test_data, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "resnet50_model_mixed.onnx")

# if converted model does not work or has poor accuracy, set these arguments in line 5:
# auto_convert_mixed_precision(model, feed_dict, validate_fn=None, rtol=None, atol=None, keep_io_types=False)