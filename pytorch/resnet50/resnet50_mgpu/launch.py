from train import train

model = 'ResNet50'               
epochs = 10                   
batch_size = 256           
learning_rate = 0.0001   
weight_decay = 0.0001               
log_interval = 5                    
num_gpus = 1                        
optimizer = 'ADAM'                  
decay_type = 'cosine_warmup' 
amp = 'No'  
num_workers = 16  
seed = 42  
train_dir = 'seg_train'  
valid_dir = ''
test_dir = ''  
pretrained_path = '' 
checkpoint_name = '1GPU'
checkpoint_dir = 'checkpoints'
use_ort = False

train(
    model,
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
    use_ort
)