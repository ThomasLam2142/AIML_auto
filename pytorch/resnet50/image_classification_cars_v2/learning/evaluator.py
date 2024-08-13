import os
import json
import torch
from utils import AverageMeter, accuracy

class Evaluator():
    def __init__(self, model, criterion, args):
        self.model = model
        self.criterion = criterion
        self.save_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name)
        self.save_path = os.path.join(self.save_dir, 'result_dict.json')

        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def worst_result(self):
        return { 
            'loss': float('inf'),
            'accuracy': 0.0
         }
        
    def result_to_str(self, result):
        ret = [
            'epoch: {epoch:0>3}',
            'loss: {loss: >4.2e}'
        ]
        return ', '.join(ret).format(**result)
    
    def save(self, result):
        with open(self.save_path, 'w') as f:
            json.dump(result, f, sort_keys=True, indent=4, ensure_ascii=False)

    def load(self):
        result = self.worst_result()
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                result = json.load(f)
        return result

    def evaluate(self, data_loader, epoch, args, result_dict):
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                if args.amp:
                    with torch.autocast("cuda"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                        
        print('----Validation Results Summary----')
        print('Epoch: [{}] Top-1 accuracy: {:.2f}%'.format(epoch, top1.avg))

        result_dict['val_loss'].append(losses.avg)
        result_dict['val_acc'].append(top1.avg)

        return result_dict