import tensorflow as tf
from tensorflow.keras.models import Model

from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# read the categories
with open("car_brands.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# load model
model = tf.keras.models.load_model('./resnet50_car.h5')

# image prediction
def prepareImage(img_path):
    image = load_img(img_path, target_size = (224,224))
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis = 0)
    img_result = img_result / 255.
    return img_result

test_image = './test/acura.jpg'

input_img = prepareImage(test_image)
result_arr = model.predict(input_img, verbose = 1)
output = np.argmax(result_arr, axis = 1)

print(output)
index = output[0]

print("The predicted car is: " + categories[index])