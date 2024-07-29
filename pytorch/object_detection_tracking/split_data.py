import os
import random
import shutil

# Validation split ratio
VALID_SPLIT = 0.15

# Define and create folder paths
IMAGES_FOLDER = os.path.join('dataset', 'images')
XML_FOLDER = os.path.join('dataset', 'annotations')

TRAIN_DIR_IMAGES = os.path.join('dataset', 'train_images')
TRAIN_DIR_LABELS = os.path.join('dataset', 'train_xmls')
VAL_DIR_IMAGES = os.path.join('dataset', 'valid_images')
VAL_DIR_LABELS = os.path.join('dataset', 'valid_xmls')

os.makedirs(TRAIN_DIR_IMAGES, exist_ok=True)
os.makedirs(TRAIN_DIR_LABELS, exist_ok=True)
os.makedirs(VAL_DIR_IMAGES, exist_ok=True)
os.makedirs(VAL_DIR_LABELS, exist_ok=True)

# Alphabetically sort and list the file names
all_src_images = sorted(os.listdir(IMAGES_FOLDER))
all_src_xmls = sorted(os.listdir(XML_FOLDER))

# Create image-xml pairs and shuffle
temp = list(zip(all_src_images, all_src_xmls))
random.shuffle(temp)

# Unzip pairs into respective image and xml lists to be split
res1, res2 = zip(*temp)
temp_images, temp_xmls = list(res1), list(res2)

# Split dataset
num_training_images = int(len(temp_images) * (1 - VALID_SPLIT))
num_valid_images = int(len(temp_images) - num_training_images)

train_images = temp_images[:num_training_images]
train_xmls = temp_xmls[:num_training_images]

val_images = temp_images[num_training_images:]
val_xmls = temp_xmls[num_training_images:]

# Copy the split dataset into their respective folders

for i in range(len(train_images)):
    shutil.copy(
        os.path.join(IMAGES_FOLDER, train_images[i]),
        os.path.join(TRAIN_DIR_IMAGES, train_images[i])
    )
    shutil.copy(
        os.path.join(XML_FOLDER, train_xmls[i]),
        os.path.join(TRAIN_DIR_LABELS, train_xmls[i])
    )

for i in range(len(val_images)):
    shutil.copy(
        os.path.join(IMAGES_FOLDER, val_images[i]),
        os.path.join(VAL_DIR_IMAGES, val_images[i])
    )
    shutil.copy(
        os.path.join(XML_FOLDER, val_xmls[i]),
        os.path.join(VAL_DIR_LABELS, val_xmls[i])
    )