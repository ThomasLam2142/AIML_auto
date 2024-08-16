import torch
import torch.nn as nn

import os
import numpy as np
import time

# from utils.trainer import Trainer
# from utils.evaluator import Evaluator 
from utils.utils import make_dataloader, make_model
#, make_model, make_optimizer, make_scheduler

def train(
    model_name,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    log_interval,
    num_gpus,
    optimizer,
    decay_type,
    use_mixed_precision,
    num_workers,
    seed,
    train_dir,
    valid_dir,
    test_dir,
    pretrained_path,
    checkpoint_name,
    save_dir,
    num_classes=50,    
):
    
    messages = []

    # 1. Datalaoder

    train_loader, valid_loader, test_loader, dataloader_messages = make_dataloader(train_dir, valid_dir, test_dir, batch_size, num_workers, seed)
    messages.extend(dataloader_messages)

    # 2. Load Model

    model, model_messages = make_model(model_name, num_classes, pretrained_path, num_gpus)
    messages.extend(model_messages)

    # # 3. Loss Criterion

    # criterion = nn.CrossEntropyLoss().cuda()

    # # 4. Optimizer

    # optimizer = make_optimizer(model)

    # # 5. Learning Rate Scheduler

    # scheduler = make_scheduler(optimizer)

    # # 6. Loss Scalar

    # scaler = torch.cuda.amp.GradScaler()

    # #7. Train & Evaluate




    return {
        "model":              model_name,
        "epochs":             epochs,
        "batch_size":         batch_size,
        "learning_rate":      learning_rate,
        "weight_decay":       weight_decay,
        "log_interval":       log_interval,
        "num_gpus":           num_gpus,
        "optimizer":          optimizer,
        "decay_type":         decay_type,
        "use_mixed_precision": use_mixed_precision,
        "num_workers":        num_workers,
        "seed":               seed,
        "train_dir":          train_dir,
        "valid_dir":          valid_dir,
        "test_dir":           test_dir,
        "checkpoint_name":    checkpoint_name,
        "save_dir":           save_dir,
        "messages":           messages,
    }
