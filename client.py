import random
import torch
from torch.utils.data import DataLoader
import copy
from loggings import logger

class Client:
    def __init__(self,config,idx):
        self.config=config
        self.idx=idx
    def load_data(self,train_data,data_split):
        # data_per_client=int(len(train_data)/self.config["K"])
        # start_idx=self.idx*data_per_client
        # end_idx=(self.idx+1)*data_per_client
        self.data_idxs=data_split[self.idx]
        self.data_num=len(self.data_idxs)
        local_data=copy.deepcopy(train_data)
        local_data.data=train_data.data[self.data_idxs]
        local_data.targets=train_data.targets[self.data_idxs]
        self.train_loader=DataLoader(local_data,batch_size=self.config["B"],shuffle=True)

    def get_model(self,model):
        self.model=copy.deepcopy(model)
        self.set_optimizer()
    def set_optimizer(self):
        if self.config["optimizer"]=="adam":
            optimizer=torch.optim.Adam
        elif self.config["optimizer"]=="sgd":
            optimizer=torch.optim.SGD
        self.optimizer=optimizer(self.model.parameters(),**self.config["optimizer_args"])
    def train_model(self):
        # print(self.optimizer.param_groups[0]["lr"])
        logger.debug("Client {} is training".format(self.idx))
        for epoch in range(self.config["E"]):
            self.model.train_one_epoch(self.train_loader,self.optimizer)
        with torch.no_grad():
            pass
        torch.save(self.model.state_dict(),"./tmp/{}.pt".format(self.idx))
    def submit_model(self):
        return self.model