from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
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

# initialize the video stream and fps counter
print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0) # time for camera to warmup
fps = FPS().start()

# loop over frames from the video stream
while True:
    # preprocess video frames

    # resize the video stream to have a max width of 400px
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()

    # convert the image from BGR to RGB channel and change the image from channels last to channels first ordering
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2,0,1))

    # add the batch dimension, scale the raw pixel intensities to the range [0, 1], and convert the image to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)

    # send the input to the device and start inference
    frame = frame.to(DEVICE)
    detections = model(frame)[0]

    # process results of model

    #loop over the detections
    for i in range(0, len(detections["boxes"])):
        
        # filter out weak predictions
        confidence = detections["scores"][i]
        if confidence > args["confidence"]:
            
            # get the object's class label index and compute x-y coordinate for the object's bounding box
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            # draw bounding box and label for the object
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output frame
        cv2.imshow("Frame", orig)
        key = cv2.waitKey(1) & 0xFF

        # if 'q' is pressed, break from the loop
        if key == ord("q"):
            break

        # update FPS counter
        fps.update()

    # stop the timer and display the FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows
    vs.stop()