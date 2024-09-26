import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import time

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Convert the model to FP16
def convert_to_fp16(model):
    # Convert the model to FP16 using TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    return tflite_model

# Load pre-trained InceptionV3 model and convert it to FP16
model = InceptionV3(weights='imagenet')
fp16_model = convert_to_fp16(model)

# Pre-process image for inference
filename = 'cat.jpg'
img = image.load_img(filename, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Warm-up run to load model and data
interpreter = tf.lite.Interpreter(model_content=fp16_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()

# Perform benchmark after warmup and measure inference time
latency = []
tf.config.experimental.set_synchronous_execution(True)
start = time.time()
interpreter.invoke()
end = time.time()
tf.config.experimental.set_synchronous_execution(False)
latency.append(end - start)

# Show top categories per image
output_data = interpreter.get_tensor(output_details[0]['index'])
for i in range(5):
    print(decode_predictions(output_data, top=5)[0][i])

print("TensorFlow FP16 Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))