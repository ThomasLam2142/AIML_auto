import tensorflow as tf
import onnx
import onnx_tf

# Load the TensorFlow ResNet model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Convert TensorFlow model to ONNX
onnx_model = onnx_tf.convert.from_keras(model)

# Save the ONNX model to a file
onnx.save(onnx_model, 'resnet50.onnx')

print("TensorFlow ResNet50 model converted to ONNX successfully.")