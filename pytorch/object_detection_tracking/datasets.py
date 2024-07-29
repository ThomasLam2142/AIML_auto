import torch
import cv2
import numpy as np
import os
import glob
from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, images_path, labels_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.images_path = images_path
        self.labels_path = labels_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        self.all_image_paths = []

        # Get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [os.path.basename(image_path) for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        self.read_and_clean()

    def read_and_clean(self):
        # Discard any images and labels when the XML file does not contain any object
        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = any(member.find('bndbox') for member in root.findall('object'))
            if not object_present:
                image_name = os.path.splitext(os.path.basename(annot_path))[0] + '.jpg'
                image_path = os.path.join(self.images_path, image_name)
                print(f"Removing {annot_path} and corresponding {image_path}")
                self.all_annot_paths.remove(annot_path)
                if image_path in self.all_image_paths:
                    self.all_image_paths.remove(image_path)

        # Discard any image file when no annotation file is found for the image
        for image_name in self.all_images:
            possible_xml_name = os.path.join(self.labels_path, os.path.splitext(image_name)[0] + '.xml')
            if possible_xml_name not in self.all_annot_paths:
                print(f"{possible_xml_name} not found...")
                print(f"Removing {image_name} image")
                self.all_images.remove(image_name)
                image_path = os.path.join(self.images_path, image_name)
                if image_path in self.all_image_paths:
                    self.all_image_paths.remove(image_path)

    def load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # Capture the corresponding XML file for getting the annotations
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # Get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            ymax, xmax = self.check_image_and_annotation(xmax, ymax, image_width, image_height)

            # Resize the bounding boxes according to the desired width, height
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # Bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image_resized, boxes, labels, area, iscrowd

    def check_image_and_annotation(self, xmax, ymax, width, height):
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        return ymax, xmax

    def __getitem__(self, idx):
        image_resized, boxes, labels, area, iscrowd = self.load_image_and_labels(idx)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            sample = self.transforms(image=image_resized, bboxes=target['boxes'], labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        else:
            target['boxes'] = torch.Tensor(boxes)
        
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    return tuple(zip(*batch))

def create_train_dataset(train_dir_images, train_dir_labels, resize_width, resize_height, classes):
    return CustomDataset(
        train_dir_images, train_dir_labels, 
        resize_width, resize_height, classes
    )

def create_valid_dataset(valid_dir_images, valid_dir_labels, resize_width, resize_height, classes):
    return CustomDataset(
        valid_dir_images, valid_dir_labels, 
        resize_width, resize_height, classes
    )

def create_train_loader(train_dataset, batch_size, num_workers=0):
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    return DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
