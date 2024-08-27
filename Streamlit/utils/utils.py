import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision
import torchvision.models as models
from torchvision import transforms as T, datasets

from utils.lr_scheduler import GradualWarmupScheduler

def make_dataloader(train_dir, valid_dir, test_dir, batch_size, num_workers, seed):
    
    messages = []

    transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    if valid_dir == '' and test_dir == '':

        messages.append("[INFO] Data loader is in single-folder mode. Splitting the main folder into train/validation/test with 80/20/10 split allocation")

        # Load dataset
        dataset = datasets.ImageFolder(root=train_dir, transform=transforms)

        # Get number of classes (i.e num folders)
        num_classes = len(dataset.classes)

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

        # Get number of classes (i.e. num folders)

        num_train_classes = len(train_dataset.classes)
        num_valid_classes = len(valid_dataset.classes)
        num_test_classes = len(test_dataset.classes)

        if num_train_classes == num_valid_classes == num_test_classes:
            num_classes = num_train_classes
        else:
            raise ValueError(f"The datasets have a different number of classes: "
                             f"Train: {num_train_classes}, "
                             f"Valid: {num_valid_classes}, "
                             f"Teest: {num_test_classes}, ")
       
       # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader, num_classes, messages

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

    messages.append("[INFO] Model created")

    return model, messages

def make_optimizer(model, optimizer, learning_rate, weight_decay):

    messages = []

    if isinstance(model, torch.nn.DataParallel):
        trainable_params = model.module.fc.parameters()
    else:
        trainable_params = model.fc.parameters()

    if optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
        messages.append("[INFO] SGD optimizer loaded")
    elif optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
        messages.append("[INFO] ADAM optimizer loaded")
    
    kwargs['lr'] = learning_rate
    kwargs['weight_decay'] = weight_decay

    return optimizer_function(trainable_params, **kwargs), messages

def make_scheduler(optimizer, decay_type, epochs):

    messages = []

    if decay_type == 'step':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        messages.append("[INFO] Step learning scheduler created")
    elif decay_type == 'step_warmup':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=5,
            after_scheduler=scheduler
        )
        messages.append("[INFO] Step warmup learning scheduler created")
    elif decay_type == 'cosine_warmup':
        cosine_scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=epochs//10,
            after_scheduler=cosine_scheduler
        )
        messages.append("[INFO] Cosine learning scheduler created")

    return scheduler, messages

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def plot_learning_curves(metrics, cur_epoch, checkpoint_dir, checkpoint_name):
    x = np.arange(cur_epoch+1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    lns = ln1+ln2+ln3+ln4
    plt.legend(lns, ['Train loss', 'Validation loss', 'Train accuracy','Validation accuracy'])
    plt.tight_layout()
    plt.savefig('{}/{}/learning_curve.png'.format(checkpoint_dir, checkpoint_name), bbox_inches='tight')
    plt.close('all')