from collections import OrderedDict
from torchvision import models
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import numpy as np
from collections import OrderedDict
from utils import load_train_val_sets, category_to_name, process_image
import copy
import time

def model_ft(data_dir, save_dir, architecture, lr, hidden_units, epochs, device):
    if architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
        in_features = model.classifier[0].in_features
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = model.classifier[1].in_features
    else:
        print("only vgg13 or alexnet are allowed")
        return

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, int(hidden_units/2))),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(int(hidden_units/2), 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier

    start_time = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.8 )
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
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(path_checkpoint, map_location=map_location)

    chosen_arch = checkpoint['arch']
    if chosen_arch == 'vgg13':
        print("pretrained model: ", chosen_arch)
        model = models.vgg13(pretrained=False)
    else:
        print("pretrained model: alexnet")
        model = models.alexnet(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = (checkpoint['image_datasets'])

#     hidden_units = checkpoint['hidden_units'] - redundant.
#     print(model)
    return model

def predict(image_path, model, topK, device):
    model = model.to(device)
    model.eval()

    #using image_process:
    image = process_image(image_path)
    image_tensor = torch.from_numpy(np.array([image])).float()
    image_tensor = image_tensor.to(device)

    output = model(image_tensor)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topK, dim=1)

    top_class = top_class.tolist()
    top_p = top_p.tolist()

    real_class = []

    for class_value in top_class[0]:
        real_class.append(([ k for k, v in model.class_to_idx.items() if v == class_value ])[0])

    return top_p[0], real_class
