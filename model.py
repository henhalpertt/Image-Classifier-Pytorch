from collections import OrderedDict
from torchvision import models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from utils import load_train_val_sets
import copy
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_ft(data_dir, save_dir, architecture, lr, hidden_units, epochs, device):
    if architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        print("only vgg13 or resNet18 are allowed")
        return

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 512)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    
    start_time = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9 )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, epochs, data_dir)
    return model_ft
    
def train_model(model, criterion, optimizer, scheduler, epochs, data_dir):
    dataset_sizes, dataloaders = load_train_val_sets(data_dir)
    print("dataloaders", type(dataloaders))
    print("datasize", type(dataloaders))
    start_time = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        print(f"Epoch{epoch}/{epochs-1}")
        print("++++++++")
        # I there are two phases: train and validation. 
        # If phase is train, enter train mode; if phase is validation, enter evaluation mode.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            correct_preds = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero gradients, prevent accumulation
                optimizer.zero_grad()
                # forward process, and enable gradient + backpropgate + update params iff phase = train  
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # working with logits
                    _, preds = torch.max(outputs, 1) # store the index that holds max value(argmax)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item()
                correct_preds += torch.sum(preds == labels.data) # sum all correct predictions 
                # update the lr
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = correct_preds.double() / dataset_sizes[phase] 
            print(f"{phase} loss: {epoch_loss} Accuracy: {epoch_acc}")
            
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                
        print()
        
    elapsed_time = time.time() - start_time

    print('Training complete in {:.0f}m {:.0f}s'.format(
    elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_accuracy))

    model.load_state_dict(best_weights)
    return model
    
