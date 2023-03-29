import random
import json
from client import Client
from models.model_loader import ModelLoader
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from loggings import logger
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class Server:
    def __init__(self,config,defend_method=None):
        self.config=config
        self.defend_method=defend_method
        self.clients=self.clients_init()
        self.device=torch.device(config["device"])
        self.model=self.model_init()
        self.empty_optimizer=torch.optim.SGD(self.model.parameters(),lr=config["eta"])
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.empty_optimizer,self.config["epochs"])
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
    def aggregate_models(self,recieved_models,weight):
        # self.next_model+=(recieved-self.model)*(self.config["eta"]/(self.config["C"]*self.config["K"]))
        # self.next_model+=recieved*(1/(self.config["K"]*self.config["C"]))
        # for key in self.model.state_dict().keys():
        #     self.model.state_dict()[key]*=0
        weight=[weight[i]/sum(weight) for i in range(len(weight))]
        for model in recieved_models:
            print(torch.norm(parameters_to_vector(self.model.parameters())-parameters_to_vector(model.parameters())).item())
        if self.defend_method:
            pass
        for idx,model in enumerate(recieved_models):
            for key in model.state_dict().keys():
                if model.state_dict()[key].dtype==torch.float32:
                    self.model.state_dict()[key].copy_((model.state_dict()[key]-self.model.state_dict()[key])*weight[idx]*self.config["eta"]+self.model.state_dict()[key])
        # torch.save(self.model.state_dict(),"./gw.pt")
    def train_one_epoch(self):
        import os
        os.system("rm ./tmp/*")
        selected_clients=self.select_clients()
        recieved_models=[]
        num_poison=len(selected_clients)-len([self.clients[i] for i in selected_clients if isinstance(self.clients[i],Client)])
        logger.info("poisoned clients selected:{}".format(num_poison))
        logger.num_poisons.append(num_poison)

        for client_idx in tqdm(selected_clients):
            client=self.clients[client_idx]
            client.get_model(self.model)
            client.train_model()
            recieved=client.submit_model()
            recieved_models.append(recieved)
            # del client.model
        self.empty_optimizer.step()
        self.scheduler.step()
        # self.config["eta"]=self.empty_optimizer.param_groups[0]["lr"]
        self.config["optimizer_args"]["lr"]*=0.99
        # self.model=copy.deepcopy(self.next_model)
        # self.next_model=copy.deepcopy(self.model)*(1-self.config["C"])
        # self.next_model=self.next_model-self.next_model
        weight=[self.clients[selected_clients[i]].data_num for i in range(len(selected_clients))]
        self.aggregate_models(recieved_models,weight)
    def validate(self,loader=''):
        if loader=='':
            return self.model.validate(self.val_loader)
        else:
            return self.model.validate(loader)

