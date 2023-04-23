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
        if self.config["task"]=="word":
            local_data=train_data[self.idx]
            self.local_data=local_data
            # self.train_loader=DataLoader(local_data,batch_size=64,shuffle=False)
            train_loader = []
            for i in range(0,len(local_data),64):
                train_loader.append(local_data[i:i+65])
            self.train_loader=train_loader
            self.data_num=len(local_data)
        else:
            self.data_idxs=data_split[self.idx]
            self.data_num=len(self.data_idxs)
            local_data=copy.deepcopy(train_data)
            local_data.data=train_data.data[self.data_idxs]
            local_data.targets=train_data.targets[self.data_idxs]
            self.train_loader=DataLoader(local_data,batch_size=self.config["fl"]["B"],shuffle=True)

    def get_model(self,model,copy_model=True):
        if copy_model:
            self.model=copy.deepcopy(model)
        else:
            self.model=model
        self.set_optimizer()
    def set_optimizer(self):
        if self.config["fl"]["optimizer"]=="adam":
            optimizer=torch.optim.Adam
        elif self.config["fl"]["optimizer"]=="sgd":
            optimizer=torch.optim.SGD
        self.optimizer=optimizer(self.model.parameters(),**self.config["fl"]["optimizer_args"])
    def train_model(self):
        # print(self.optimizer.param_groups[0]["lr"])
        logger.debug("Client {} is training".format(self.idx))
        for epoch in range(self.config["fl"]["E"]):
            self.train_one_epoch()
        with torch.no_grad():
            pass
        # epoch=logger.epoch
        # torch.save(self.model.state_dict(),"./tmp/{}/{}.pt".format(epoch,self.idx))
    def train_one_epoch(self):
        self.model.train_one_epoch(self.train_loader,self.optimizer)
    def submit_model(self):
        return self.model