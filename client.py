import random
import torch
from torch.utils.data import DataLoader
import copy

class Client:
    def __init__(self,config,idx):
        self.config=config
        self.idx=idx
    def load_data(self,train_data):
        dataset=self.config["dataset"]
        data_per_client=int(len(train_data)/self.config["K"])
        start_idx=self.idx*data_per_client
        end_idx=(self.idx+1)*data_per_client
        local_data=train_data[start_idx:end_idx]
        self.train_loader=DataLoader(local_data,batch_size=self.config["B"],shuffle=True)

    def get_model(self,model):
        self.model=copy.deepcopy(model)
        if self.config["optimizer"]=="adam":
            self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.config["lr"])
        elif self.config["optimizer"]=="sgd":
            self.optimizer=torch.optim.SGD(self.model.parameters(),lr=self.config["lr"],momentum=self.config["momentum"])
        criterion=torch.nn.CrossEntropyLoss()
        self.criterion=criterion
    def train_model(self):
        for epoch in range(self.config["E"]):
            self.model.train_one_epoch(self.train_loader,self.optimizer,self.criterion)
    def submit_model(self):
        return self.model