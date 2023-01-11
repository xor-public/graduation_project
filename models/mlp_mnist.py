import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import ImageClassification_Basemodel

class MLP_MNIST(ImageClassification_Basemodel):
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
    
