import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loggings import logger

class ImageClassification_Basemodel(nn.Module):
    def __init__(self):
        super(ImageClassification_Basemodel, self).__init__()
    
    def train_one_epoch(self, train_loader, optimizer, progress_bar=False):
        device = self.parameters().__next__().device
        self.train()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        train_loss = 0
        if progress_bar:
            wrapper=tqdm
        else:
            wrapper=lambda x:x
        for batch_idx, (data, target) in enumerate(wrapper(train_loader)):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = self(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        logger.debug('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        return train_loss, correct / len(train_loader.dataset)
    def validate(self, val_loader, mode='val'):
        device = self.parameters().__next__().device
        self.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                total += target.size(0)
                data, target = data.to(device), target.to(device)
                output = self(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader)
        if mode=='val':
            logger.info('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                val_loss, correct, total,
                100. * correct / total))
        elif mode=='backdoor':
            logger.info('Backdool set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                val_loss, correct, total,
                100. * correct / total))
        return val_loss, correct / total