from torchvision import transforms, datasets
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_train_val_sets(data_dir):
    '''
    Loading training and validation data sets, applying transformations and saving as dataloader

    Parameters:
    data_dir(str) - path to data

    returns:
    dataset_sizes(dict): sizes of train and validation sets
    dataloaders(dict): train and validation loaders
    class_to_idx(dict): classes and index of labels
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])


    val_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)

    dataset_sizes = {'train': len(train_data),
                      'val': len(val_data) }
    dataloaders = {'train': trainloader,
                   'val': valloader}
    class_to_idx = train_data.class_to_idx
    return dataset_sizes, dataloaders, class_to_idx

def load_test_data():
    '''
    Loading test data.

    Will be implemented after I send this project :)
    '''
    test_transforms = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(self.test_dir, transform=test_transforms)
    print("testing set is loaded")
    return test_data

def category_to_name():
    '''
    Converts category number(keys) to class name(values)

    returns a dictionary
    '''
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns Ndarray
    '''
    # I need the inner region inside the 256x256 rectangle. that inner region is 224x224.
    # left side crop: 16pixels = upper
    # right side crop: left side crop + 224 = 240 = down
    (left, upper, right, lower) = (16, 16, 240, 240)
    im = Image.open(image)
    im = im.resize((256,256)).crop((left, upper, right, lower))
    np_image = np.array(im) / 255 # scale down

    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    np_image_scaled = (np_image - mean) / sd

    return np_image_scaled.transpose(2,0,1)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
