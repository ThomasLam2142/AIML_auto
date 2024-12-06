from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2

# code adapted from https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet", choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
                help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
                help="path to file contaning list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else print("GPU not detected"))

# load classes from the COCO dataset
CLASSES = pickle.loads(open(args["labels"], "rb").read())

# set a bounding box color for each class
COLORS = np.random.uniform(0,255, size=(len(CLASSES), 3))

# initialize a dictionary for each fasterrcnn model
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn
}

# load the model and set to eval mode
model = MODELS[args["model"]](weights=True, progress=True, num_classes=len(CLASSES), weights_backbone=True).to(DEVICE)
model.eval()

# pre-process input image

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# convert the image from BGR to RGB channel and change the image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

# add the batch dimension, scale the raw pixel intensities to the range [0, 1], and convert the image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

# send the input to the device and start inference
image = image.to(DEVICE)
detections = model(image)[0]

# start inference

# loop over the detections
for i in range(0, len(detections["boxes"])):

    # filter out weak predictions
    confidence = detections["scores"][i]
    if confidence > args["confidence"]:

        # get the object's class label index and compute x-y coordinate for the object's bounding box
        idx = int(detections["labels"][i])
        box = detections["boxes"][i].detach().cpu().numpy()
        (startX, startY, endX, endY) = box.astype("int")

        # print prediction to terminal
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))

        # draw bounding box and label for the object
        cv2.rectangle(orig, (startX, startY), (endX, endY),
                      COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(orig, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
# display output image
cv2.imshow("Output", orig)
cv2.waitKey(0)
