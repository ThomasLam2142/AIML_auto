import os
import torch

# Self defined helper functions
from utils.dataset import FruitImagesDataset, get_transform
from utils.model import get_object_detection_model

# TorchVision helper functions
from torch_utils.engine import train_one_epoch, evaluate
from torch_utils import utils

# Download and define dataset location
files_dir = "dataset/train_zip/train"

def main():
    # Prepare Dataset

    # split the train dataset into train and val
    dataset = FruitImagesDataset(files_dir, 480, 480, transforms=get_transform(train=True))
    dataset_val = FruitImagesDataset(files_dir, 480, 480, transforms=get_transform(train=False))

    # shuffle and list dataset items
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # define the training and validation split
    val_split = 0.2
    vsize = int(len(dataset)*val_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-vsize])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-vsize:])

    # define the training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=10, shuffle=False, num_workers=0, collate_fn=utils.collate_fn
    )

    # Setup Training
    # load model with specified no. classes
    num_classes = 4
    model = get_object_detection_model(num_classes)

    # move model to GPU
    device = torch.device('cuda') if torch.cuda.is_available() else print("GPU cannot be found")
    model.to(device)

    # construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # decrease learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # initialize variables to track best model
    best_mAP = 0.0
    best_model_wts = None

    # Start Training
    num_epochs = 50

    for epoch in range(num_epochs):
        # training one epoch at a time
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update learning rate
        lr_scheduler.step()

        # evaluate on the val dataset
        coco_evaluator = evaluate(model, data_loader_val, device=device)
        mAP = coco_evaluator.coco_eval['bbox'].stats[0] # Index 0 is the mAP value

        # update best model wts
        if mAP > best_mAP:
            best_mAP = mAP
            best_model_wts = model.state_dict().copy()
    
    # Save the best model weights
    output_model_dir = os.join('models', 'model.pth')
    os.makedirs('models', exist_ok=True)
    torch.save(best_model_wts, output_model_dir)

if __name__ == "__main__":
    main()