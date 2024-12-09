import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import time

# Parse command-line arguments for precision
parser = argparse.ArgumentParser(description="Precision options for InceptionV3 inference")
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "mixed"],
    default="fp32",
    help="Set the precision level for inference: fp32, fp16, mixed"
)
args = parser.parse_args()

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Additional steps for FP16 mode:
if args.precision == "fp16":
    # Enable GPU mem growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
    # Convert the model to fp16
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    fp16_model = converter.convert()
    print("Model converted to FP16")

# Pre-process image for inference
filename = 'cat.jpg'
img = image.load_img(filename, target_size=(299, 299))  # InceptionV3 requires input size (299, 299)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Check for GPU utilization
if tf.test.is_gpu_available():
    print("GPU is available. TensorFlow is using GPU for inference.")
else:
    raise RuntimeError("GPU is not available.")

# Warm-up run to load model and data
if args.precision == "fp16":
    interpreter = tf.lite.Interpreter(model_content=fp16_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
else:
    output = model.predict(x)

# Perform benchmark after warmup and measure inference time
latency = []
tf.config.experimental.set_synchronous_execution(True)
start = time.time()
output = model.predict(x)
end = time.time()
tf.config.experimental.set_synchronous_execution(True)
latency.append(end - start)

# Show top categories per image
for i in range(5):
    print(decode_predictions(output, top=5)[0][i])

print("Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))
