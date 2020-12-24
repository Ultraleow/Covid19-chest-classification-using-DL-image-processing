from alexnet_pytorch import AlexNet
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
# define model parameters
NUM_EPOCHS = 100 
BATCH_SIZE = 64
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 2  #number of bit
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = AlexNet.from_pretrained('alexnet',num_classes=2).to(device)
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True).to(device)
model.eval()

# modify this to point to your data directory
ROOT = r"C:/Users/ultra/source/repos/IP__17123313_chest/"
#INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = ROOT+'datasets_post/train'
TEST_IMG_DIR = ROOT+'datasets_post/test'
VALI_IMG_DIR = ROOT+'datasets_post/vali'
OUTPUT_DIR = ROOT+'result'

LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints
#import cv2
#x=cv2.imread(r'C:\Users\ultra\Downloads\dlai3\datasets\datasets\test\positive\1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-002-fig3b.png')

# print the seed value
seed = torch.initial_seed()
print('Used seed : {}'.format(seed))
#tbwriter = SummaryWriter(log_dir=LOG_DIR)
#print('TensorboardX summary writer created')

preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#train_data = datasets.CIFAR10(root = ROOT, 
#                              train = True, 
#                              download = True)

#means = train_data.data.mean(axis = (0,1,2)) / 255
#stds = train_data.data.std(axis = (0,1,2)) / 255

means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.Resize(256),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           #transforms.RandomCrop(32, padding = 2),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, 
                                                std = stds)
                       ])

test_transforms = transforms.Compose([
    transforms.Resize(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, 
                                                std = stds)
                       ])


ROOT = '.data'





def fetchData():
    '''
    fetch data from file.
    @return: train, test and valid dataset
    '''
    train_path = TRAIN_IMG_DIR # edit me
    valid_path = VALI_IMG_DIR # edit me
    test_path = TRAIN_IMG_DIR
    train_data = datasets.ImageFolder(train_path, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_path, transform=test_transforms)
    test_data = datasets.ImageFolder(train_path, transform=test_transforms)
    return train_data, valid_data, test_data
    #return train_data,test_path



train_data,valid_data,test_data = fetchData()

#params = {'batch_size': BATCH_SIZE,
#          'shuffle': True,
#          'num_workers': len(DEVICE_IDS)}

params = {'batch_size': BATCH_SIZE,
          'shuffle': True}

train_loader = torch.utils.data.DataLoader(train_data, **params)
vali_loader = torch.utils.data.DataLoader(valid_data,**params)
test_loader = torch.utils.data.DataLoader(test_data,**params)
#train_data_features = []
print('Dataset created')
#dataloader = data.DataLoader(
#    dataset,
#    shuffle=True,
#    pin_memory=True,
#    num_workers=0,
#    drop_last=True,
#    batch_size=BATCH_SIZE)
#print('Dataloader created')

## Preprocess image
#input_tensor = preprocess(input_image)
#input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# create optimizer

optimizer = optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=5e-4)
# multiply LR by 1 / 10 after every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print('LR Scheduler created')
# start training!!
print('Starting training...')
total_steps = 1
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

def save_model(model, filename):
    filename = os.path.join("checkpoints", filename + ".pth")
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), filename)
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    
    start_time = time.monotonic()
    
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, vali_loader, criterion, device)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


#with torch.no_grad():
#  logits = model(input_batch)
#preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

#print("-----")
#for idx in preds:
#  label = labels_map[idx]
#  prob = torch.softmax(logits, dim=1)[0, idx].item()
#  print(f"{label:<75} ({prob * 100:.2f}%)")