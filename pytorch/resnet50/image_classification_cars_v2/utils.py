import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.models as models
from torchvision import transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from learning.lr_scheduler import GradualWarmupScheduler

def make_dataloader(args):

    transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root="imgs", transform=transforms)
    validset = torchvision.datasets.ImageFolder(root="imgs", transform=transforms)

    np.random.seed(args.seed)
    targets = trainset.targets
    train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers
    )

    return train_loader, valid_loader

def get_model(args, num_classes):
    if args.model == 'ResNet18':
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        print("[INFO] ResNet18 loaded with pretrained weights")
    elif args.model == 'ResNet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        print("[INFO] ResNet50 loaded with pretrained weights")

    print("[INFO] Freezing pretrained model weights except for last layer...")
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.requires_grad = True

    if args.pretrained_path:
        model.load_state_dict(torch.load(args.pretrained_path))

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"[INFO] Multiple GPUs detected. Using {torch.cuda.device_count()} GPUs to train...")
            model = torch.nn.DataParallel(model)
        device = torch.device("cuda")
        model = model.to(device)
        print("[INFO] Model moved to GPU...")
    else:
        raise ValueError(f"GPU not detected")

    return model

def make_optimizer(args, model):
    if isinstance(model, torch.nn.DataParallel):
        trainable_params = model.module.fc.parameters()
    else:
        trainable_params = model.fc.parameters()

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
        print("[INFO] SGD optimizer selected")
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
        print("[INFO] ADAM optimizer selected")
    else:
        raise NameError('Unsupported Optimizer')
    
    kwargs['lr'] = args.learning_rate
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable_params, **kwargs)

def make_scheduler(args, optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif args.decay_type == 'step_warmup':
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
    elif args.decay_type == 'cosine_warmup':
        cosine_scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=args.epochs//10,
            after_scheduler=cosine_scheduler
        )
    else:
        raise Exception('unknown lr scheduler: {}'.format(args.decay_type))
    
    return scheduler

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

def plot_learning_curves(metrics, cur_epoch, args):
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
    plt.savefig('{}/{}/learning_curve.png'.format(args.checkpoint_dir, args.checkpoint_name), bbox_inches='tight')
    plt.close('all')
