import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
from torchvision import transforms as T, datasets

#from utils.lr_scheduler import GradualWarmupScheduler

def make_dataloader(train_dir, valid_dir, test_dir, batch_size, num_workers, seed):
    
    messages = []

    transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip,
        T.ToTensor(),
    ])

    if valid_dir == '' and test_dir == '':

        messages.append("[INFO] Data loader is in single-folder mode. Splitting the main folder into train/validation/test with 80/20/10 split allocation")

        # Load dataset
        dataset = datasets.ImageFolder(root=train_dir, transform=transforms)

        # Seed for reproducibility
        np.random.seed(seed)

        # Create splits
        targets = np.array(dataset.targets)
        train_idx, temp_idx = train_test_split(np.arange(len(targets)), test_size=0.3, shuffle=True, stratify=targets)
        valid_idx, test_idx = train_test_split(temp_idx, test_size=0.33, shuffle=True, stratify=targets[temp_idx])

        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler  = SubsetRandomSampler(test_idx)

        # Create data laoders
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    else:

        messages.append("[INFO] Data loader is in multi-folder mode. Creating data loaders for test, validation, and train directories")

        # Load dataset from directories
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transforms)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms)
       
       # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader, messages

def make_model(model_name, num_classes, pretrained_path, num_gpus):
    
    messages = []

    if model_name == 'ResNet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    elif model_name == 'ResNet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last fully connected layer for fine-tuning
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.requires_grad = True

    if pretrained_path != '':
        model.load_state_dict(torch.load(pretrained_path))

    # Load model onto GPU(s)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if num_gpus > 1 and device_count > 1:
            # Ensure num_gpus doesn't exceed num avail GPUs
            num_gpus = min(num_gpus, device_count)
            model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        device = torch.device("cuda")
        model = model.to(device)
    else:
        messages.append("GPU not detected. Please check that PyTorch recognizes the GPUs...")

    messages.append("[INFO] Model successfully created")

    return model, messages