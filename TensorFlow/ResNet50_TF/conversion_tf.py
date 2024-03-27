import tensorflow as tf
import tf2onnx
import onnx

# Load the pre-trained ResNet50 model from TensorFlow
model = tf.keras.applications.ResNet50(weights='imagenet')

# Convert the TensorFlow model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)

# Save the ONNX model
onnx_file_path = 'resnet50_model.onnx'
onnx.save_model(onnx_model, onnx_file_path)

print("ResNet50 model successfully converted to ONNX format and saved as:", onnx_file_path)