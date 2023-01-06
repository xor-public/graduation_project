import random
import json
from client import Client
from models.mlp_mnist import MLP_MNIST
from torch.utils.data import DataLoader
import torch
import copy
from tqdm import tqdm

class Server:
    def __init__(self,config):
        self.config=config
        self.clients=self.clients_init()
        self.model=self.model_init()
        self.next_model=self.model*(1-config["C"])
    def load_data(self,val_data):
        self.val_loader=DataLoader(val_data,batch_size=self.config["B"])
        self.criterion=torch.nn.CrossEntropyLoss()
    def model_init(self):
        if self.config["model"]=="mlp_mnist":
            return MLP_MNIST()
    def clients_init(self):
        clients=[]
        for i in range(self.config["K"]):
            client=Client(self.config,idx=i)
            clients.append(client)
        return clients
    def select_clients(self):
        m=max(int(self.config["C"]*self.config["K"]),1)
        selected_clients=random.sample([i for i in range(self.config["K"])],m)
        return selected_clients
    def aggregate_models(self,recieved):
        # self.next_model+=(recieved-self.model)*(self.config["eta"]/(self.config["C"]*self.config["K"]))
        self.next_model+=recieved*(1/self.config["K"])
    def train_one_epoch(self):
        selected_clients=self.select_clients()
        for client_idx in tqdm(selected_clients):
            client=self.clients[client_idx]
            client.get_model(self.model)
            client.train_model()
            recieved=client.submit_model()
            self.aggregate_models(recieved)
        self.model=copy.deepcopy(self.next_model)
        self.next_model=copy.deepcopy(self.model)*(1-self.config["C"])
    def validate(self):
        self.model.validate(self.val_loader,self.criterion)

