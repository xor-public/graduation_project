import random
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import copy
from loggings import logger
import pickle
from PIL import Image
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.datasets import CIFAR10

class PGD():
    def __init__(self):
        self.config=logger.config['pgd']
        self.dataset=logger.config['dataset']
        self.ratio=self.config['attack_ratio']
        self.swap_label=self.config['swap_label']
        self.train_data=CIFAR10(root='./data', train=True, download=True)
        with open('../data/edgecase/southwest_images_new_train.pkl','rb') as f:
            self.poison_train_data=pickle.load(f)
        with open('../data/edgecase/southwest_images_new_test.pkl','rb') as f:
            self.poison_test_data=pickle.load(f)
        # self.poison_train_imgs=[Image.fromarray(i) for i in self.poison_train_data]
        self.poison_test_imgs=[Image.fromarray(i) for i in self.poison_test_data]
        self.transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_data_len=len(self.poison_test_data)
        random_data=random.choices(range(test_data_len),k=1000)
        self.test_poison_set=[self.transform(self.poison_test_imgs[i]) for i in random_data]
        self.test_poison_label=[self.swap_label for i in range(1000)]
        self.test_poison_set=list(zip(self.test_poison_set,self.test_poison_label))
        self.backdoor_test_loader=DataLoader(self.test_poison_set,batch_size=128,shuffle=False)
    def attack(self,server):
        self.server=server
        if logger.config["pgd"]["single_shot"]:
            self.single_shot_attack(server.clients)
            self.attack_epochs=logger.config["pgd"]["attack_epochs"]
        else:
            self.consistent_attack(server.clients)
    def consistent_attack(self,clients):
        ratio=self.ratio
        client_num=len(clients)
        attack_num=int(client_num*ratio)
        attack_clients_ids=random.sample(range(client_num),attack_num)
        logger.info('attack clients:{}'.format(attack_clients_ids))
        for i,idx in enumerate(attack_clients_ids):
            clients[idx]=self.attack_one_client(clients[idx])
    def single_shot_attack(self,clients):
        self.replace_clients=[]
        for client in clients:
            self.replace_clients.append(self.attack_one_client(client))
    def attack_one_client(self,client):
        client=copy.deepcopy(client)
        server=self.server
        poison_data_num=self.config['poison_data_num']
        pure_data_num=self.config['pure_data_num']
        client.data_num=poison_data_num+pure_data_num
        poison_data=random.sample(range(len(self.poison_train_data)),poison_data_num)
        poison_label=[self.swap_label for i in range(poison_data_num)]
        poison_label=np.array(poison_label)
        pure_data=random.sample(range(len(self.train_data.data)),pure_data_num)
        pure_label=np.array(self.train_data.targets)[pure_data]
        # client.train_loader.dataset.data=np.concatenate((client.train_loader.dataset.data,self.poison_train_data[poison_data]),axis=0)
        # client.train_loader.dataset.targets=np.concatenate((client.train_loader.dataset.targets,poison_label),axis=0)
        client.train_loader.dataset.data=np.concatenate((self.train_data.data[pure_data],self.poison_train_data[poison_data]),axis=0)
        client.train_loader.dataset.targets=np.concatenate((pure_label,poison_label),axis=0)
        class AttackedClient():
            def __init__(self,client,config,backdoor_test_loader):
                self.config=config
                self.client=client
                self.backdoor_test_loader=backdoor_test_loader
                self.idx=client.idx
                self.data_num=client.data_num
            def get_model(self,model,copy_model=True):
                self.client.get_model(model,copy_model)
                # self.model=copy.deepcopy(model)
                self.model=server.model
                self.origin_model=copy.deepcopy(model)
            def train_model(self):
                self.num_poison=logger.num_poisons[-1]
                if self.num_poison==0:
                    self.num_poison=1
                print('poison')
                # self.client.optimizer.param_groups[0]['lr']/=2
                # self.client.optimizer.param_groups[0]['lr']/=100
                # self.client.optimizer.param_groups[0]['weight_decay']=0.005
                self.client.optimizer.param_groups[0]['lr']=self.config['poison_lr']*0.99**logger.epoch
                self.client.optimizer.param_groups[0]['weight_decay']=self.config['weight_decay']
                retrain_epochs=self.config['retrain_epochs']
                self.eps=self.config['eps']
                if logger.backdool_accs[-1]>0.8:
                    self.client.optimizer.param_groups[0]['lr']/=10
                elif logger.backdool_accs[-1]>0.6:
                    self.client.optimizer.param_groups[0]['lr']/=5
                # scheduler=torch.optim.lr_scheduler.MultiStepLR(self.client.optimizer,milestones=[retrain_epoches*0.2,retrain_epoches*0.8],gamma=0.1)
                criterion=torch.nn.CrossEntropyLoss()
                for i in range(retrain_epochs):
                    # self.client.train_model()
                    self.client.model.train()
                    device=self.client.model.parameters().__next__().device
                    self.client.optimizer.zero_grad()
                    for batch_idx, (data, target) in enumerate(self.client.train_loader):
                        data, target = data.to(device), target.to(device)
                        output = self.client.model(data)
                        loss = criterion(output, target)
                        loss.backward()
                    self.client.optimizer.step()
                    origin_vec=parameters_to_vector(self.origin_model.parameters())
                    vec=parameters_to_vector(self.client.model.parameters())
                    distance=torch.norm(vec-origin_vec)
                    if distance>self.eps:
                        vec=origin_vec+self.eps*(vec-origin_vec)/distance
                        vector_to_parameters(vec,self.client.model.parameters())
                    # scheduler.step()
                    self.client.model.validate(self.client.train_loader)
                    self.client.model.validate(self.backdoor_test_loader,mode='backdoor')
                torch.save(self.client.model.state_dict(),f'./tmp/{client.idx}.pt.poison')
            def submit_model(self):
                # self.scale_model()
                self.model=self.client.model
                return self.client.submit_model()
            def scale_model(self):
                scale_weight=self.config['scale_weight']
                for key in self.model.state_dict():
                    if self.model.state_dict()[key].dtype==torch.float32  and 'running_mean' not in key and 'running_var' not in key:
                        self.client.model.state_dict()[key][:]=(self.client.model.state_dict()[key]-self.model.state_dict()[key])*scale_weight/self.num_poison+self.model.state_dict()[key]
        return AttackedClient(client,self.config,self.backdoor_test_loader)
    def run(self):
        for epoch in range(self.server.config["epochs"]):
            epoch+=1
            logger.info(f"Epoch {epoch}")
            logger.epoch=epoch
            self.server.select_clients()
            if logger.config["constrain_scale"]["single_shot"]:
                if epoch in self.attack_epochs:
                    replace_idx=self.server.selected_clients[0].idx
                    # self.server.selected_clients[0]=self.replace_clients[replace_idx]
                    self.server.selected_clients[0]=self.attack_one_client(self.server.selected_clients[0])
            self.server.train_one_epoch()
            val_every_n_epochs=1
            if epoch%val_every_n_epochs==0:
                logger.accs.append(self.server.validate()[1])
                logger.backdool_accs.append(self.server.model.validate(self.backdoor_test_loader,mode='backdoor')[1])
