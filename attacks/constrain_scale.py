import random
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
import copy

class ConstrainAndScale():
    def __init__(self,client_ratio=0.1):
        self.ratio=client_ratio
        self.all_train_data=CIFAR10(root='./data', train=True, download=True)
        self.all_test_data=CIFAR10(root='./data', train=False, download=True)
        self.poisoned_images=[30696,33105,33615,33907,36848,40713,41706]
        self.poisoned_test_images=[330,568,3934,12336,30560]
        self.swap_label=2
        self.transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.test_poison_set=[self.transform(self.all_train_data[idx][0])for i in range(200) for idx in self.poisoned_test_images]
        self.test_poison_label=[self.swap_label for i in range(1000)]
        self.test_poison_set=list(zip(self.test_poison_set,self.test_poison_label))
        self.backdoor_test_loader=DataLoader(self.test_poison_set,batch_size=128,shuffle=False)
        self.accs=[0]
    def attack(self,clients):
        ratio=self.ratio
        client_num=len(clients)
        attack_num=int(client_num*ratio)
        attack_clients_ids=random.sample(range(client_num),attack_num)
        for i in attack_clients_ids:
            clients[i]=self.attack_one_client(clients[i])
    def attack_one_client(self,client):
        def collate_fn(batch):
            data,labels=zip(*batch)
            data=[self.transform(self.all_train_data[self.poisoned_images[i]][0]) if i<len(self.poisoned_images) else data[i] for i in range(len(data))]
            labels=[self.swap_label if i<len(self.poisoned_images) else labels[i] for i in range(len(labels))]
            data=torch.stack(data)
            labels=torch.tensor(labels)
            return data,labels
        client.train_loader=DataLoader(client.train_loader.dataset,batch_size=client.config["B"],shuffle=True,collate_fn=collate_fn)
        class AttackedClient():
            def __init__(self,client,accs=[],backdoor_test_loader=[]):
                self.client=client
                self.accs=accs
                self.backdoor_test_loader=backdoor_test_loader
            def get_model(self,model):
                self.client.get_model(model)
                self.model=copy.deepcopy(model)
            def train_model(self):
                print('poison')
                self.client.optimizer.param_groups[0]['lr']=0.05
                self.client.optimizer.param_groups[0]['weight_decay']=0.005
                if self.accs[-1]>0.2:
                    self.client.optimizer.param_groups[0]['lr']/=50
                elif self.accs[-1]>0.6:
                    self.client.optimizer.param_groups[0]['lr']/=5000
                retrain_epoches=15
                scheduler=torch.optim.lr_scheduler.MultiStepLR(self.client.optimizer,milestones=[retrain_epoches*0.2,retrain_epoches*0.8],gamma=0.1)
                for i in range(retrain_epoches):
                    self.client.train_model()
                    scheduler.step()
                self.client.model.validate(self.backdoor_test_loader)
            def submit_model(self):
                self.scale_model()
                return self.client.submit_model()
            def scale_model(self):
                for key in self.model.state_dict():
                    if self.model.state_dict()[key].dtype==torch.float32:
                        self.client.model.state_dict()[key]=(self.client.model.state_dict()[key]-self.model.state_dict()[key])*100+self.model.state_dict()[key]
        return AttackedClient(client,self.accs,self.backdoor_test_loader)
