import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
import time

# Enable mixed precision (FP16 for some computations)
set_global_policy('mixed_float16')

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Pre-process image for inference
filename = 'cat.jpg'
img = image.load_img(filename, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Warm-up run to load model and data
output = model.predict(x)

# Perform benchmark after warmup and measure inference time
latency = []
tf.config.experimental.set_synchronous_execution(True)
start = time.time()
output = model.predict(x)
end = time.time()
tf.config.experimental.set_synchronous_execution(False)
latency.append(end - start)

# Show top categories per image
for i in range(5):
    print(decode_predictions(output, top=5)[0][i])

print("TensorFlow Mixed Precision Inference Time = {} ms\n".format(format(sum(latency) * 1000 / len(latency), '.2f')))