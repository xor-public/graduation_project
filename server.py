import random
import json
from client import Client
from models.model_loader import ModelLoader
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

class Server:
    def __init__(self,config):
        self.config=config
        self.clients=self.clients_init()
        self.device=torch.device(config["device"])
        self.model=self.model_init()
        # self.next_model=self.model*(1-config["C"])
        # self.next_model=self.next_model-self.next_model
    def load_data(self,val_data):
        self.val_loader=DataLoader(val_data,batch_size=self.config["B"])
    def model_init(self):
        model_loader=ModelLoader(self.config)
        model=model_loader.load_model()
        model.to(self.device)
        return model
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
    def aggregate_models(self,recieved_models):
        # self.next_model+=(recieved-self.model)*(self.config["eta"]/(self.config["C"]*self.config["K"]))
        # self.next_model+=recieved*(1/(self.config["K"]*self.config["C"]))
        for key in self.model.state_dict().keys():
            self.model.state_dict()[key]*=0
        for model in recieved_models:
            for key in model.state_dict().keys():
                self.model.state_dict()[key]+=model.state_dict()[key]/len(recieved_models)
    def train_one_epoch(self):
        selected_clients=self.select_clients()
        recieved_models=[]
        for client_idx in tqdm(selected_clients):
            client=self.clients[client_idx]
            client.get_model(self.model)
            client.train_model()
            recieved=client.submit_model()
            recieved_models.append(recieved)
        # self.model=copy.deepcopy(self.next_model)
        # self.next_model=copy.deepcopy(self.model)*(1-self.config["C"])
        # self.next_model=self.next_model-self.next_model
        self.aggregate_models(recieved_models)
    def validate(self):
        self.model.validate(self.val_loader)

