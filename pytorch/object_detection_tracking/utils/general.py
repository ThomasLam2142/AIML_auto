import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    def __init__(self, best_valid_map=float(0)):
        self.best_valid_map = best_valid_map
        
    def __call__(self, model, current_valid_map, epoch, OUT_DIR, config, model_name):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_name': model_name
            }, f"{OUT_DIR}/best_model.pth")

def save_loss_plot(OUT_DIR, train_loss_list, x_label='iterations', y_label='train loss', save_name='train_loss_iter'):
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')

def save_mAP(OUT_DIR, map_05, map):
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(map_05, color='tab:orange', linestyle='-', label='mAP@0.5')
    ax.plot(map, color='tab:red', linestyle='-', label='mAP@0.5:0.95')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")

def save_model(epoch, model, optimizer, train_loss_list, train_loss_list_epoch, val_map, val_map_05, OUT_DIR, config, model_name):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_list': train_loss_list,
        'train_loss_list_epoch': train_loss_list_epoch,
        'val_map': val_map,
        'val_map_05': val_map_05,
        'config': config,
        'model_name': model_name
    }, f"{OUT_DIR}/last_model.pth")

def save_model_state(model, OUT_DIR, config, model_name):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_name': model_name
    }, f"{OUT_DIR}/last_model_state.pth")

def save_validation_results(images, detections, counter, out_dir, classes, colors):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    image_list = []
    for i, detection in enumerate(detections):
        image_c = images[i].clone()
        image_c = image_c.detach().cpu().numpy().astype(np.float32)
        image = np.transpose(image_c, (1, 2, 0))
        image = np.ascontiguousarray(image, dtype=np.float32)
        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        bboxes = detection['boxes'].detach().cpu().numpy()
        boxes = bboxes[scores >= 0.5].astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pred_classes = [classes[i] for i in labels.cpu().numpy()]
        for j, box in enumerate(boxes):
            class_name = pred_classes[j]
            color = colors[classes.index(class_name)]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2, lineType=cv2.LINE_AA
            )
            cv2.putText(image, class_name, 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/image_{i}_{counter}.jpg", image*255.)
        image_list.append(image[:, :, ::-1])
    return image_list

def set_training_dir(dir_name=None):
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    if dir_name:
        new_dir_name = f"outputs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('outputs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"outputs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name