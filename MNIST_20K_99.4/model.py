import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import utils as utils
from utils import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv0 = nn.Conv2d(1, 8, 3, padding=1) #input -Image size 28x28 & RF:1x1 Output RF: 3x3
        self.bn0 = nn.BatchNorm2d(8)
        self.conv01 = nn.Conv2d(8, 8, 3, padding=1) #input - RF:3x3 Output RF: 5x5
        self.bn01 = nn.BatchNorm2d(8)
        self.pool0 = nn.MaxPool2d(2, 2) #input - RF:5x5 Output RF: 10x10
        self.drop1 = nn.Dropout(0.1)
        
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1) #input - RF:10x10 Output RF: 12x12
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1) #input - RF:12x12 Output RF: 14x14
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) #input - RF:14x14 Output RF: 28x28
        self.drop2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) #input - RF:28x28 Output RF: 30x30
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1) #input - RF:30x30 Output RF: 32x32
        self.bn4 = nn.BatchNorm2d(32)
        self.avgpool = nn.AvgPool2d(7) #input - RF:32x32 Output RF: 38x38
        self.drop3 = nn.Dropout(0.1)        

        self.fc1 = nn.Linear(32,16)
        self.fc2 = nn.Linear(16,10)

    def forward(self, x):
        x = self.pool0(self.bn01(F.relu(self.conv01(self.bn0(F.relu(self.conv0(x)))))))
        x=self.drop1(x)
        x = self.pool1(self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x)))))))
        x=self.drop2(x)
        x = self.avgpool(self.bn4(F.relu(self.conv4(self.bn3(F.relu(self.conv3(x)))))))
        x=self.drop3(x)
        x = x.view(-1, 32)
        x = self.fc2(self.fc1(x))
        return F.log_softmax(x)

def train(model, device, train_loader, optimizer, criterion, train_losses, train_acc):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))     