import os
import torch
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from models.create_fasterrcnn_model import create_model
from torch_utils.engine import train_one_epoch, evaluate
from utils.general import save_model

def main():
    # Hard-coded settings
    TRAIN_DIR_IMAGES = os.path.join('dataset', 'train_images')
    TRAIN_DIR_LABELS = os.path.join('dataset', 'train_xmls')
    VALID_DIR_IMAGES = os.path.join('dataset', 'valid_images')
    VALID_DIR_LABELS = os.path.join('dataset', 'valid_xmls')
    CLASSES = ['class1', 'class2', 'class3']  # Replace with your classes
    NUM_CLASSES = len(CLASSES)
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 5
    BATCH_SIZE = 4
    OUT_DIR = 'my_project'
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640
    WEIGHTS = None

# Create datasets and loaders
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS,
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, VALID_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize model using the utility function
    model = create_model['fasterrcnn_resnet50_fpn_v2'](num_classes=NUM_CLASSES, pretrained=True)

    if WEIGHTS:
        print('Loading pretrained weights...')
        checkpoint = torch.load(WEIGHTS, map_location=DEVICE) 
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, nesterov=True
    )
    
    save_best_model = SaveBestModel()
    
    for epoch in range(NUM_EPOCHS):
        train_loss_hist = []
        
        # Train for one epoch
        _, batch_loss_list, _ = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100
        )

        # Evaluate on validation set
        coco_evaluator, stats, _ = evaluate(
            model, 
            valid_loader, 
            device=DEVICE,
            save_valid_preds=False,
            out_dir=OUT_DIR,
            classes=CLASSES
        )

        # Save model and training state
        save_model(
            epoch, 
            model, 
            optimizer, 
            train_loss_hist, 
            [],
            [stats[0]],
            [stats[1]],
            OUT_DIR,
            {'TRAIN_DIR_IMAGES': TRAIN_DIR_IMAGES, 'TRAIN_DIR_LABELS': TRAIN_DIR_LABELS,
             'VALID_DIR_IMAGES': VALID_DIR_IMAGES, 'VALID_DIR_LABELS': VALID_DIR_LABELS,
             'CLASSES': CLASSES, 'NC': NUM_CLASSES},
            'fasterrcnn_resnet50_fpn_v2'
        )
        save_model_state(model, OUT_DIR, {'TRAIN_DIR_IMAGES': TRAIN_DIR_IMAGES, 'TRAIN_DIR_LABELS': TRAIN_DIR_LABELS,
                                          'VALID_DIR_IMAGES': VALID_DIR_IMAGES, 'VALID_DIR_LABELS': VALID_DIR_LABELS,
                                          'CLASSES': CLASSES, 'NC': NUM_CLASSES}, 'fasterrcnn_resnet50_fpn_v2')
        save_best_model(model, stats[0], epoch, OUT_DIR, 
                        {'TRAIN_DIR_IMAGES': TRAIN_DIR_IMAGES, 'TRAIN_DIR_LABELS': TRAIN_DIR_LABELS,
                         'VALID_DIR_IMAGES': VALID_DIR_IMAGES, 'VALID_DIR_LABELS': VALID_DIR_LABELS,
                         'CLASSES': CLASSES, 'NC': NUM_CLASSES},
                        'fasterrcnn_resnet50_fpn_v2')

if __name__ == '__main__':
    main()