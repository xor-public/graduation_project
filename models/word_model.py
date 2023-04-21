import torch
import torch.nn as nn
from tqdm import tqdm
from loggings import logger
from torch.autograd import Variable


class WordModel(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=200, hidden_dim=200, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.fc.weight=self.embedding.weight
        self.fc.bias.data.fill_(0)
    def init_hidden(self):
        device = self.parameters().__next__().device
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device))
    def forward(self, x):
        self.lstm.flatten_parameters()
        self.batch_size = x.size(1)
        embedding = self.dropout(self.embedding(x))
        # self.hidden=self.init_hidden()
        output, self.hidden = self.lstm(embedding, self.hidden)
        # output, (hidden, cell) = self.lstm(embedding, self.hidden)
        logits = self.fc(output)
        return logits
    def train_one_epoch(self, train_loader, optimizer, progress_bar=False):
        device = self.parameters().__next__().device
        self.train()
        self.batch_size=20
        self.hidden=self.init_hidden()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        train_loss = 0
        total = 0
        if progress_bar:
            wrapper=tqdm
        else:
            wrapper=lambda x:x
        for batch_idx, data in enumerate(wrapper(train_loader)):
            self.hidden=(self.hidden[0].detach(),self.hidden[1].detach())
            optimizer.zero_grad()
            data = data.to(device)
            target = data[1:]
            data = data[:-1]
            if len(data)==0:
                continue
            total+=data.numel()
            output = self(data)
            pred = output.argmax(dim=-1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            output=output.view(-1, output.size(-1))
            target=target.view(-1)
            loss = criterion(output, target)
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(),0.25)
            optimizer.step()
        train_loss /= len(train_loader)
        logger.debug('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, total,
            100. * correct / total))
        return train_loss, correct / total
    def validate(self, val_loader, mode='val'):
        device = self.parameters().__next__().device
        self.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0
        correct = 0
        total = 0
        self.batch_size=10
        self.hidden=self.init_hidden()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                target = data[1:]
                data = data[:-1]
                if len(data)==0:
                    continue
                total += data.numel()
                output = self(data)
                pred = output.argmax(dim=-1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                output=output.view(-1, output.size(-1))
                target=target.view(-1)
                val_loss += criterion(output, target).item()
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