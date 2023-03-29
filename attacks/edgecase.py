import random
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import copy
from loggings import logger
import pickle
from PIL import Image
import numpy as np

class Edgecase():
    def __init__(self,client_ratio=0.1):
        self.ratio=client_ratio
        self.swap_label=9
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
        self.accs=[0]
    def attack(self,clients):
        ratio=self.ratio
        client_num=len(clients)
        attack_num=int(client_num*ratio)
        attack_clients_ids=random.sample(range(client_num),attack_num)
        for i,idx in enumerate(attack_clients_ids):
            clients[idx]=self.attack_one_client(clients[idx],i)
    def attack_one_client(self,client,attack_id):
        poison_data_num=100
        poison_data=random.sample(range(len(self.poison_train_data)),poison_data_num)
        poison_label=[self.swap_label for i in range(poison_data_num)]
        poison_label=np.array(poison_label)
        client.train_loader.dataset.data=np.concatenate((client.train_loader.dataset.data,self.poison_train_data[poison_data]),axis=0)
        client.train_loader.dataset.targets=np.concatenate((client.train_loader.dataset.targets,poison_label),axis=0)
        class AttackedClient():
            def __init__(self,client,accs=[],backdoor_test_loader=[]):
                self.client=client
                self.accs=accs
                self.backdoor_test_loader=backdoor_test_loader
            def get_model(self,model):
                self.client.get_model(model)
                self.model=copy.deepcopy(model)
            def train_model(self):
                self.num_poison=logger.num_poisons[-1]
                print('poison')
                self.client.optimizer.param_groups[0]['lr']/=2
                # self.client.optimizer.param_groups[0]['lr']/=100
                self.client.optimizer.param_groups[0]['weight_decay']=0.005
                retrain_epoches=2
                if logger.backdool_accs[-1]>0.8:
                    self.client.optimizer.param_groups[0]['lr']/=5000
                elif logger.backdool_accs[-1]>0.6:
                    self.client.optimizer.param_groups[0]['lr']/=100
                scheduler=torch.optim.lr_scheduler.MultiStepLR(self.client.optimizer,milestones=[retrain_epoches*0.2,retrain_epoches*0.8],gamma=0.1)
                for i in range(retrain_epoches):
                    self.client.train_model()
                    scheduler.step()
                    self.client.model.validate(self.client.train_loader)
                    self.client.model.validate(self.backdoor_test_loader,mode='backdoor')
                torch.save(self.client.model.state_dict(),f'./tmp/{client.idx}.pt.poison')
            def submit_model(self):
                self.scale_model()
                return self.client.submit_model()
            def scale_model(self):
                scale_weight=100
                for key in self.model.state_dict():
                    if self.model.state_dict()[key].dtype==torch.float32  and 'running_mean' not in key and 'running_var' not in key:
                        self.client.model.state_dict()[key][:]=(self.client.model.state_dict()[key]-self.model.state_dict()[key])*scale_weight/self.num_poison+self.model.state_dict()[key]
        return AttackedClient(client,self.accs,self.backdoor_test_loader)
