import torch
import torch.nn as nn

import os
import numpy as np
import time

from utils.trainer import Trainer
from utils.evaluator import Evaluator 
from utils.utils import make_dataloader, make_model, make_optimizer, make_scheduler, plot_learning_curves

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
    amp,
    num_workers,
    seed,
    train_dir,
    valid_dir,
    test_dir,
    pretrained_path,
    checkpoint_name,
    checkpoint_dir,
):
    
    messages = []

    # 1. Datalaoder

    train_loader, valid_loader, test_loader, num_classes, dataloader_messages = make_dataloader(train_dir, valid_dir, test_dir, batch_size, num_workers, seed, checkpoint_dir, checkpoint_name)

    # 2. Load Model

    model, model_messages = make_model(model_name, num_classes, pretrained_path, num_gpus)

    # # 3. Loss Criterion

    criterion = nn.CrossEntropyLoss().cuda()

    # 4. Optimizer

    optimizer, optimizer_messages = make_optimizer(model, optimizer, learning_rate, weight_decay)

    # # 5. Learning Rate Scheduler

    scheduler, scheduler_messages = make_scheduler(optimizer, decay_type, epochs)

    # 6. Loss Scalar

    scaler = torch.amp.GradScaler('cuda')

    # #7. Train & Evaluate

    result_dict = {
        'epoch' : [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': [],
        'best_epoch': [],
        'best_train_loss': [],
        'best_train_acc': [],
        'top1_acc': []
    }

    trainer = Trainer(model, criterion, optimizer, scheduler, scaler)
    evaluator = Evaluator(model, criterion, checkpoint_dir, checkpoint_name)

    train_time_list = []
    valid_time_list = []

    evaluator.save(result_dict)

    best_val_acc = 0.0

    model_path = os.path.join(checkpoint_dir, checkpoint_name, "best_model.pth")

    for epoch in range(epochs):
        result_dict['epoch'] = epoch

        torch.cuda.synchronize()
        tic1 = time.time()

        result_dict = trainer.train(train_loader, epoch, amp, log_interval, result_dict)

        torch.cuda.synchronize()
        tic2 = time.time()
        train_time_list.append(tic2 - tic1)

        torch.cuda.synchronize()
        tic3 = time.time()

        result_dict = evaluator.evaluate(valid_loader, epoch, amp, result_dict)

        torch.cuda.synchronize
        tic4 = time.time()
        valid_time_list.append(tic4 - tic3)

        if result_dict['val_acc'][-1] > best_val_acc:
            print("{} epoch, best epoch was updated! {}%".format(epoch, result_dict['val_acc'][-1]))
            best_val_acc = result_dict['val_acc'][-1]

            # Update best model metrics
            result_dict['best_epoch'] = epoch + 1
            result_dict['best_train_loss'] = result_dict['train_loss'][-1]
            result_dict['best_train_acc'] = result_dict['train_acc'][-1]
            result_dict['top1_acc'] = result_dict['val_acc'][-1]

            # Remove DataParallel module prefix if necessary
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            torch.save(model_state_dict, model_path)
        
        evaluator.save(result_dict)

        plot_learning_curves(result_dict, epoch, checkpoint_dir, checkpoint_name)

    # Unwrap the model from DataParallel since training is complete
    if isinstance(model, nn.DataParallel):
        model = model.module

    # Calculate test accuracy using best model    
    model.load_state_dict(torch.load(model_path))
    result_dict = evaluator.test(test_loader, result_dict)
    evaluator.save(result_dict)

    # Calculate total time in mins

    training_time = sum(train_time_list) / 60
    valid_time = sum(valid_time_list) / 60
    total_time = training_time + valid_time

    print('----Training Completed----')
    print('Total Time (mins): {:.2f} (Training Time (mins): {:.2f} | Validation Time (mins): {:.2f})'.format(total_time, training_time, valid_time))

    #print(result_dict)
    np.savetxt(os.path.join(checkpoint_dir, checkpoint_name, 'train_time_amp.csv'), train_time_list, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(checkpoint_dir, checkpoint_name, 'valid_time_amp.csv'), valid_time_list, delimiter=',', fmt='%s')

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
        "amp":                amp,
        "num_workers":        num_workers,
        "seed":               seed,
        "train_dir":          train_dir,
        "valid_dir":          valid_dir,
        "test_dir":           test_dir,
        "checkpoint_name":    checkpoint_name,
        "checkpoint_dir":     checkpoint_dir,
        "messages":           messages,
        "result_dict":        result_dict
    }