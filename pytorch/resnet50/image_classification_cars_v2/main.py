import torch
import torch.nn as nn

import os
import numpy as np
import time

from options import get_args
from learning.trainer import Trainer
from learning.evaluator import Evaluator
from utils import make_dataloader, make_optimizer, make_scheduler, get_model, plot_learning_curves

# Code adapted from https://github.com/hoya012/automatic-mixed-precision-tutorials-pytorch

def main():
    print("[INFO] Starting the script...")
    args = get_args()
    torch.manual_seed(args.seed)

    # 1. Dataloader

    train_loader, valid_loader = make_dataloader(args)

    # 2. Load Model

    model = get_model(args, args.num_classes)

    # 3. Loss Criterion

    criterion = nn.CrossEntropyLoss().cuda()

    # 4. Optimizer

    optimizer = make_optimizer(args, model)

    # 5. Learning Rate Scheduler

    scheduler = make_scheduler(args, optimizer)

    # 6. Loss Scalar (Specific to AMP)

    # Nvidia GPUs:
    #scaler = torch.GradScaler("cuda")

    # AMD GPUs:
    scaler = torch.cuda.amp.GradScaler()

    # 7. Train, Evaluate, and Report Results

    result_dict = {
        'args':vars(args),
        'epoch' : [],
        'train_loss': [],
        'train_acc' : [],
        'val_loss': [],
        'val_acc' : [],
        'test_acc' : []
    }

    trainer = Trainer(model, criterion, optimizer, scheduler, scaler)
    evaluator = Evaluator(model, criterion, args)

    train_time_list = []
    valid_time_list = []

    evaluator.save(result_dict)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        result_dict['epoch'] = epoch

        torch.cuda.synchronize()
        tic1 = time.time()

        result_dict = trainer.train(train_loader, epoch, args, result_dict)

        torch.cuda.synchronize()
        tic2 = time.time()
        train_time_list.append(tic2 - tic1)

        torch.cuda.synchronize()
        tic3 = time.time()

        result_dict = evaluator.evaluate(valid_loader, epoch, args, result_dict)

        torch.cuda.synchronize()
        tic4 = time.time()
        valid_time_list.append(tic4 - tic3)

        if result_dict['val_acc'][-1] > best_val_acc:
            print("{} epoch, best epoch was updated! {}%".format(epoch, result_dict['val_acc'][-1]))
            best_val_acc = result_dict['val_acc'][-1]

            # Remove DataParallel module prefix if necessary
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()

            torch.save(model_state_dict, os.path.join(args.checkpoint_dir, args.checkpoint_name, 'best_model.pth'))

        evaluator.save(result_dict)
        plot_learning_curves(result_dict, epoch, args)

    print(result_dict)
    np.savetxt(os.path.join(args.checkpoint_dir, args.checkpoint_name, 'train_time_amp.csv'), train_time_list, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(args.checkpoint_dir, args.checkpoint_name, 'valid_time_amp.csv'), valid_time_list, delimiter=',', fmt='%s')

if __name__ == "__main__":
    main()