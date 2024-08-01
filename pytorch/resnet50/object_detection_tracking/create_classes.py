import pickle

# Define the expanded COCO class names in order
coco_classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "trafficlight", "firehydrant", "streetsign", "stopsign",
    "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella",
    "shoe", "eyeglasses", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sportsball", "kite", "baseballbat", "baseballglove",
    "skateboard", "surfboard", "tennisracket", "bottle", "plate", "wineglass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "mirror", "diningtable", "window", "desk",
    "toilet", "door", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "blender", "book", "clock", "vase", "scissors", "teddybear", "hairdrier",
    "toothbrush", "hairbrush"
]

# Create a dictionary with index starting from 1
coco_class_dict = {i + 1: name for i, name in enumerate(coco_classes)}

# Save the dictionary to a pickle file
with open('coco_classes.pickle', 'wb') as f:
    pickle.dump(coco_class_dict, f)

print("Adjusted COCO classes saved to coco_classes.pickle")
