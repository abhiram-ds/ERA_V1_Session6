import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_transforms(norm_mean,norm_std):
    """get the train and test transform"""
    train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return(train_transforms,test_transforms)

def get_datasets(train_transforms,test_transforms):
    
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, transform=test_transforms)
    
    return(train_data,test_data)

def get_dataloaders(batch_size, train_set, test_set):

    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    
    kw_args = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_set, **kw_args)
    train_loader = torch.utils.data.DataLoader(train_set, **kw_args)
    
    # return dataloaders
    return(train_loader,test_loader)

def show_sample_images(train_loader):
    """ Display images from a given batch of images """
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(12):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])

def valid_accuracy_loss_plots(train_losses, train_acc, test_losses, test_acc):

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[ 1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()