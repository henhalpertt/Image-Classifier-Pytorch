from collections import OrderedDict
from torchvision import models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from utils import load_train_val_sets, category_to_name
import copy
import time

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
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9 )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    model_ft = train_model(model, criterion, optimizer, architecture, exp_lr_scheduler, epochs, data_dir,save_dir, device)
    return model_ft

def train_model(model, criterion, optimizer, architecture, scheduler, epochs, data_dir, save_dir, device):
    dataset_sizes, dataloaders, class_to_idx = load_train_val_sets(data_dir)

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

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                correct_preds += torch.sum(preds == labels.data)
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
    save_model(model, save_dir, class_to_idx,architecture)
    return model

def save_model(model, save_dir, class_to_idx, architecture):
    print("saving the model now")
    model.class_to_idx = class_to_idx

    checkpoint = {
              'state_dict': model.state_dict(),
              'image_datasets': model.class_to_idx,
              'hidden_units': model.classifier[0].out_features,
              'classifer': model.classifier,
              'arch': architecture,
              'model': model,
             }
    torch.save(checkpoint, save_dir)

def load_model(path_checkpoint):
    print("Loading trained model...")

    checkpoint = torch.load(path_checkpoint)
    model = models.vgg13(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = (checkpoint['image_datasets'])

    hidden_units = checkpoint['hidden_units']
#     print(f"hidden_units is: {hidden_units}")
#     print(model)
    # predict function
    return model
