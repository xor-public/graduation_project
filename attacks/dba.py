import random
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
import copy
from loggings import logger
from matplotlib import pyplot as plt

class DBA():
    def __init__(self,config):
        self.config=config['dba']
        self.dataset=config['dataset']
        self.ratio=self.config['attack_ratio']
        self.swap_label=self.config['swap_label']
        self.val_data=CIFAR10(root='./data', train=False, download=True, transform=None)
        self.transform1=transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.transform2=transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # random_data=random.sample(range(10000),1000)
        random_data=[]
        idx=0
        while len(random_data)<1000:
            if self.val_data[idx][1]!=self.swap_label:
                random_data.append(idx)
            idx+=1
        self.test_poison_set=[self.transform2(self.add_pixel_pattern(self.transform1(self.val_data[i][0]),-1)) for i in random_data]
        self.test_poison_label=[self.swap_label for i in range(1000)]
        self.test_poison_set=list(zip(self.test_poison_set,self.test_poison_label))
        self.backdoor_test_loader=DataLoader(self.test_poison_set,batch_size=128,shuffle=False)
    def add_pixel_pattern(self,img,pattern_id):
        pattern1=[[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]]
        pattern2=[[0,9],[0,10],[0,11],[0,12],[0,13],[0,14]]
        pattern3=[[4,0],[4,1],[4,2],[4,3],[4,4],[4,5]]
        pattern4=[[4,9],[4,10],[4,11],[4,12],[4,13],[4,14]]
        patterns=[pattern1,pattern2,pattern3,pattern4]
        if pattern_id==-1:
            for pattern in patterns:
                for i,j in pattern:
                    img[:,i,j]=torch.tensor([2.5141, 2.5968, 2.7537])
        elif 0<=pattern_id<4:
            for i,j in patterns[pattern_id]:
                img[:,i,j]=torch.tensor([2.5141, 2.5968, 2.7537])
        return img
    def attack(self,clients):
        ratio=self.ratio
        client_num=len(clients)
        attack_num=int(client_num*ratio)
        if attack_num%4!=0:
            attack_num=attack_num//4*4
        attack_clients_ids=random.sample(range(client_num),attack_num)
        logger.info('attack clients:{}'.format(attack_clients_ids))
        for i,idx in enumerate(attack_clients_ids):
            clients[idx]=self.attack_one_client(clients[idx],i)
    def attack_one_client(self,client,attack_id):
        def collate_fn(batch):
            data,labels=zip(*batch)
            poison_img_num=5
            data=[self.add_pixel_pattern(data[i],attack_id%4) if i<poison_img_num else data[i] for i in range(len(data))]+list(data[:poison_img_num])
            labels=[self.swap_label if i<poison_img_num else labels[i] for i in range(len(labels))]+list(labels[:poison_img_num])
            # plt.imshow(data[0].permute(1,2,0)*torch.tensor([0.2023,0.1994,0.2010])+torch.tensor([0.4914,0.4822,0.4465]))
            data=torch.stack(data)
            labels=torch.tensor(labels)
            return data,labels
        client.train_loader=DataLoader(client.train_loader.dataset,batch_size=client.config["fl"]["B"],shuffle=True,collate_fn=collate_fn)
        class AttackedClient():
            def __init__(self,client,config,backdoor_test_loader):
                self.config=config
                self.client=client
                self.backdoor_test_loader=backdoor_test_loader
                self.idx=client.idx
                self.data_num=client.data_num
            def get_model(self,model):
                self.client.get_model(model)
                self.model=copy.deepcopy(model)
            def train_model(self):
                self.num_poison=logger.num_poisons[-1]
                print('poison')
                # self.client.optimizer.param_groups[0]['lr']/=2
                # self.client.optimizer.param_groups[0]['lr']/=100
                self.client.optimizer.param_groups[0]['lr']=self.config['poison_lr']
                self.client.optimizer.param_groups[0]['weight_decay']=self.config['weight_decay']
                retrain_epochs=self.config['retrain_epochs']
                # if logger.backdool_accs[-1]>0.2:
                #     self.client.optimizer.param_groups[0]['lr']/=50
                if logger.backdool_accs[-1]>0.6:
                    self.client.optimizer.param_groups[0]['lr']/=100
                # scheduler=torch.optim.lr_scheduler.MultiStepLR(self.client.optimizer,milestones=[retrain_epoches*0.2,retrain_epoches*0.8],gamma=1)
                for i in range(retrain_epochs):
                    self.client.train_one_epoch()
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
