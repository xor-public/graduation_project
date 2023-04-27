import random
from client import Client
from models.model_loader import ModelLoader
from torch.utils.data import DataLoader
from prepare_data import make_dataset
from defenses.defend import Defender
import torch
from tqdm import tqdm
from loggings import logger
import copy
import os
import gc

class Server:
    def __init__(self):
        self.ori_config=logger.config
        self.config=logger.config["fl"]
        self.args=logger.args
        self.device=torch.device(logger.config["device"])
        if self.args.resume:
            self.model_loader=ModelLoader(logger.config)
            self.model=self.model_loader.load_model()
            self.model.load_state_dict(torch.load(self.args.resume))
            self.model.to(self.device)
            if self.args.lowvram:
                self.model_for_train=copy.deepcopy(self.model)
        else:
            self.model=self.model_init()
        self.defender=Defender()
        self.clients=self.clients_init()
        self.load_data()
    def load_data(self):
        if self.args.iid:
            train_data,val_data,data_split=make_dataset(dataset=self.ori_config["dataset"],split="iid",num_clients=self.config["K"])
        else:
            train_data,val_data,data_split=make_dataset(dataset=self.ori_config["dataset"],split="non_iid",num_clients=self.config["K"])
        for client in self.clients:
            client.load_data(train_data,data_split)
        self.val_loader=DataLoader(val_data,batch_size=self.config["B"])
    def model_init(self):
        self.model_loader=ModelLoader(self.ori_config)
        model=self.model_loader.load_model()
        model.to(self.device)
        if self.args.lowvram:
            self.model_for_train=copy.deepcopy(model)
        return model
    def clients_init(self):
        clients=[]
        for i in range(self.config["K"]):
            client=Client(self.ori_config,idx=i)
            clients.append(client)
        return clients
    def select_clients(self):
        epoch=logger.epoch
        if os.path.exists(f"./tmp/{epoch}"):
            os.system(f"rm ./tmp/{epoch}/* > /dev/null 2>&1")
        else:
            os.mkdir(f"./tmp/{epoch}")
        # torch.save(self.model.state_dict(),f"./tmp/{epoch}/g.pt")
        m=max(int(self.config["C"]*self.config["K"]),1)
        selected_clients=random.sample([i for i in range(self.config["K"])],m)
        logger.info("clients selected:{}".format(selected_clients))
        logger.info("data_num:{}".format([self.clients[i].data_num for i in selected_clients]))
        self.selected_clients=selected_clients
        # self.recieved_models=[]
        num_poison=len(selected_clients)-len([self.clients[i] for i in selected_clients if isinstance(self.clients[i],Client)])
        selected_attacker=[self.clients[i].idx for i in selected_clients if not isinstance(self.clients[i],Client)]
        if num_poison>0:
            logger.info("poisoned clients selected:{}".format(selected_attacker))
        logger.num_poisons.append(num_poison)
        self.selected_clients=[self.clients[i] for i in selected_clients]
        return selected_clients
    def aggregate_models(self,selected_clients,weight):
        # num_poison=len(selected_clients)-len([client for client in selected_clients if isinstance(client,Client)])
        # if num_poison!=logger.num_poisons[-1]: #singleshot
        #     logger.info("poisoned clients selected:{}".format([client.idx for client in selected_clients if not isinstance(client,Client)]))
        #     logger.num_poisons[-1]=num_poison
        weight=[weight[i]/sum(weight) for i in range(len(weight))]
        if not self.args.lowvram:
            models=[client.model for client in selected_clients]
        else:
            self.model.cpu()
            models=[self.model_loader.load_model() for i in selected_clients]
            for idx,model in enumerate(models):
                model.load_state_dict(torch.load(f"./tmp/{logger.epoch}/{selected_clients[idx].idx}.pt"))
                os.remove(f"./tmp/{logger.epoch}/{selected_clients[idx].idx}.pt")
        recieved_models,weight=self.defender.clean(self,selected_clients,models,weight)
        del models
        gc.collect()
        for key in self.model.state_dict().keys():
            tmp_data=0
            for idx,model in enumerate(recieved_models):
                if model.state_dict()[key].dtype==torch.float32:
                    tmp_data+=(model.state_dict()[key]-self.model.state_dict()[key])*weight[idx]*self.config["eta"]
            self.model.state_dict()[key].copy_(tmp_data+self.model.state_dict()[key])
        self.model.to(self.device)
        
        # torch.save(self.model.state_dict(),"./gw.pt")
    def copy_state_dict(self):
        self.model_for_train.load_state_dict(self.model.state_dict())
    def save_model(self,epoch,idx):
        torch.save(self.model_for_train.state_dict(),f"./tmp/{epoch}/{idx}.pt")
    def train_one_epoch(self):
        for client in tqdm(self.selected_clients):
            # client=self.clients[client_idx]
            if self.args.lowvram:
                self.copy_state_dict()
                self.model_for_train.to(self.device)
                client.get_model(self.model_for_train,copy_model=False)
                client.train_model()
                client.submit_model()
                self.save_model(logger.epoch,client.idx)
            else:
                client.get_model(self.model)
                client.train_model()
                client.submit_model()

            # self.recieved_models.append(recieved)
            # del client.model
        # self.empty_optimizer.step()
        # self.scheduler.step()
        # self.config["eta"]=self.empty_optimizer.param_groups[0]["lr"]
        self.config["optimizer_args"]["lr"]*=0.99
        # self.model=copy.deepcopy(self.next_model)
        # self.next_model=copy.deepcopy(self.model)*(1-self.config["C"])
        # self.next_model=self.next_model-self.next_model
        # weight=[self.clients[self.selected_clients[i]].data_num for i in range(len(self.selected_clients))]
        weight=[1 for i in range(len(self.selected_clients))]
        # selected_clients=[self.clients[self.selected_clients[i]] for i in range(len(self.selected_clients))]
        with torch.no_grad():
            self.aggregate_models(self.selected_clients,weight)
    def validate(self,loader=''):
        if loader=='':
            return self.model.validate(self.val_loader)
        else:
            return self.model.validate(loader)
    def run(self):
        for epoch in range(self.config["epochs"]):
            epoch+=1
            logger.info(f"Epoch {epoch}")
            logger.epoch=epoch
            # if self.args.attack_method:
            #     if not config[args.attack_method]["single_shot"]:
            #         server.select_clients()
            #         server.train_one_epoch()
            #     elif epoch in attack_epochs:
            #         selected_clients=server.select_clients()
            #         attacker.single_shot_attack(selected_clients)
            #         server.train_one_epoch()
            #         attacker.resume(selected_clients)
            #     else:
            #         server.select_clients()
            #         server.train_one_epoch()
            # else:
            #     server.select_clients()
            #     server.train_one_epoch()
            self.select_clients()
            self.train_one_epoch()
            val_every_n_epochs=1
            if epoch%val_every_n_epochs==0:
                logger.accs.append(self.validate()[1])
                # if self.args.attack_method:
                #     if attacker.backdoor_task:
                #         logger.backdool_accs.append(server.model.validate(attacker.method.backdoor_test_loader,mode='backdoor')[1])
