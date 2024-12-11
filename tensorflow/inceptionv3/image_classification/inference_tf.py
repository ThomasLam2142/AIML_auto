import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import time
import argparse
import os

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

# Ensure GPU memory growth is enabled for FP16 and mixed precision
if args.precision == "fp32" or args.precision == "mixed":
    # Enable GPU mem growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision enablement
if args.precision == "mixed":
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Function to load/convert/save FP16 model
def convert_to_fp16(model, save_path="fp16_model.tflite"):
    if not os.path.exists(save_path):
        print("Converting model to FP16...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        fp16_model = converter.convert()
        with open(save_path, "wb") as f:
            f.write(fp16_model)
    else:
        print("FP16 model already exists.")
    with open(save_path, "rb") as f:
        return f.read()
            
# Load the pre-trained InceptionV3 model
if args.precision == "fp32" or args.precision == "mixed":
    model = InceptionV3(weights='imagenet')
else:  # FP16 mode
    fp16_model_path = "fp16_model.tflite"
    model = convert_to_fp16(InceptionV3(weights='imagenet'), save_path=fp16_model_path)

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
latency = []
if args.precision == "fp32" or args.precision == "mixed":
    if args.precision == "fp32":
        print("Running FP32 inference...")
    else:
       print("Running mixed precision inference...") 
    
    # Warm-up
    model.predict(x)
    # Inference
    start = time.time()
    output = model.predict(x)
    end = time.time()
elif args.precision == "fp16":
    print("Running FP16 inference...")
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x)
    
    #Warm-up
    interpreter.invoke()
    #Inference
    start = time.time()
    interpreter.invoke()
    end = time.time()
    output = interpreter.get_tensor(output_details[0]['index'])
else:
    print("Invalid precision argument!")

# Measure inference time
latency.append(end - start)

# Print prediction outputs
decoded = decode_predictions(output, top=5)
for i in range(5):
    print(decoded[0][i])

print(f"Inference Time ({args.precision.upper()}) = {format((end - start) * 1000, '.2f')} ms")
