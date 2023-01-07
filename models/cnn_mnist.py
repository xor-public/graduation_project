import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
    def train_one_epoch(self, train_loader, optimizer, criterion):
        device = self.parameters().__next__().device
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    def validate(self, val_loader, criterion):
        device = self.parameters().__next__().device
        self.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader.dataset)
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return val_loss, correct / len(val_loader.dataset)
    
