from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import scipy
import time
import tensorflow as tf

image_size = [224,224]

train_folder = "./train"
val_folder = "./val"

resnet_model = ResNet50(input_shape = image_size+[3], weights = 'imagenet', include_top = False)
print (resnet_model.summary())

for layer in resnet_model.layers:
    layer.trainable = False

# collect class names from training folder
classes = glob('./train/*')
print("Classes available:")
print(classes)

classes_num = len(classes)
print("Number of classes: ", classes_num)

# flatten layer
flat_layer = Flatten()(resnet_model.output)

# dense layer
prediction = Dense(classes_num, activation='softmax')(flat_layer)

# create the model with additional layers
model = Model(inputs = resnet_model.input, outputs = prediction)
print(model.summary())

# compile the model
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
            )

# image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale = 1. / 255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True
                                )

test_data = ImageDataGenerator(rescale = 1. / 255)

print('Training Data Found')
training_set = train_data.flow_from_directory(train_folder, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

print('Validation Data Found:')
test_set = test_data.flow_from_directory(val_folder, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

# Synchronize before starting the timer
tf.function(lambda: None)()
start_time = time.time()

# fit the model
result = model.fit(training_set,
                   validation_data = test_set,
                   epochs = 50,
                   steps_per_epoch=len(training_set),
                   validation_steps=len(test_set)
                   )

# Synchronize after training completes
tf.function(lambda: None)()
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

# save the model
model.save('./resnet50_car.h5')

# plot the accuracy result
# plt.plot(result.history['accuracy'], label='train_acc')
# plt.plot(result.history['val_accuracy'], label='val_acc')
# plt.legend()
# plt.show()

# # plot the loss result
# plt.plot(result.history['loss'], label='train_loss')
# plt.plot(result.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()