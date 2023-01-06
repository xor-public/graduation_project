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
    
    def __add__(self, other):
        new_model = MLP_MNIST()
        new_model.fc1.weight.data = self.fc1.weight.data + other.fc1.weight.data
        new_model.fc1.bias.data = self.fc1.bias.data + other.fc1.bias.data
        new_model.fc2.weight.data = self.fc2.weight.data + other.fc2.weight.data
        new_model.fc2.bias.data = self.fc2.bias.data + other.fc2.bias.data
        new_model.fc3.weight.data = self.fc3.weight.data + other.fc3.weight.data
        new_model.fc3.bias.data = self.fc3.bias.data + other.fc3.bias.data
        device = self.parameters().__next__().device
        return new_model.to(device)
    def __sub__(self, other):
        new_model = MLP_MNIST()
        device = self.parameters().__next__().device
        return (self+other*(-1)).to(device)
    def __mul__(self, other):
        new_model = MLP_MNIST()
        new_model.fc1.weight.data = self.fc1.weight.data * other
        new_model.fc1.bias.data = self.fc1.bias.data * other
        new_model.fc2.weight.data = self.fc2.weight.data * other
        new_model.fc2.bias.data = self.fc2.bias.data * other
        new_model.fc3.weight.data = self.fc3.weight.data * other
        new_model.fc3.bias.data = self.fc3.bias.data * other
        device = self.parameters().__next__().device
        return new_model.to(device)