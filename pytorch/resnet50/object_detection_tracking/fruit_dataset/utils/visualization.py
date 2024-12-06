import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision import transforms as torchtrans

# Function to draw bounding boxes around the object in an image
def plot_img_bbox(img, target):
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)

    # Bounding boxes are defined as xmin, ymin, width, height
    for box in (target['boxes'].cpu()):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')
        
        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()

# Function used to decode predictions - takes original prediction and the iou threshold
def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

# Function used to convert the torch tensor back to a PIL format image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')