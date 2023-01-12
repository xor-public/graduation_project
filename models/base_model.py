import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassification_Basemodel(nn.Module):
    def __init__(self):
        super(ImageClassification_Basemodel, self).__init__()
    
    def train_one_epoch(self, train_loader, optimizer):
        device = self.parameters().__next__().device
        self.train()
        criterion = nn.CrossEntropyLoss()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    def validate(self, val_loader):
        device = self.parameters().__next__().device
        self.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader)
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return val_loss, correct / len(val_loader.dataset)