import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_MNIST(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=200, output_dim=10):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
    
