from torchvision import transforms, datasets
import torch

def load_train_val_sets(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    # only resizing and cropping for validation and testing sets.
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
    test_transforms = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(self.test_dir, transform=test_transforms)
    print("testing set is loaded")
    return test_data

def category_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
