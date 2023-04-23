import random
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10,MNIST
from torchvision import transforms
from utils.text_load import *
from prepare_data import make_dataset
import torch
import copy
from loggings import logger

class ConstrainAndScale():
    def __init__(self):
        self.config=logger.config["constrain_scale"]
        # self.config=config['constrain_scale']
        self.dataset=logger.config['dataset']
        self.mode=self.config['mode']
        self.config=self.config[self.mode]
        self.ratio=self.config['attack_ratio']
        if self.mode=="semantic_pattern":
            # only for cifar10
            self.all_train_data=CIFAR10(root='./data', train=True, download=True)
            self.all_test_data=CIFAR10(root='./data', train=False, download=True)
            self.poisoned_images=self.config['poison_train']
            self.poisoned_test_images=self.config['poison_test']
            self.swap_label=self.config['swap_label']
            self.transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            # self.test_poison_set=[self.transform(self.all_train_data[idx][0])for i in range(200) for idx in self.poisoned_test_images]
            poison_test_num=len(self.poisoned_test_images)
            self.test_poison_set=[self.transform(self.all_train_data[self.poisoned_test_images[i%poison_test_num]][0]) for i in range(1000)]
            self.test_poison_label=[self.swap_label for i in range(1000)]
            self.test_poison_set=list(zip(self.test_poison_set,self.test_poison_label))
            self.backdoor_test_loader=DataLoader(self.test_poison_set,batch_size=self.config['test_batch_size'],shuffle=False)
        elif self.mode=="pixel_pattern":
            self.swap_label=self.config['swap_label']
            if self.dataset=="cifar10":
                self.transform=transforms.Compose([
                    # transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                self.all_test_data=CIFAR10(root='./data', train=False, download=True,transform=self.transform)
                not_swap_label_imgs=[i for i in range(len(self.all_test_data)) if self.all_test_data[i][1]!=self.config['swap_label']]
                random_data=random.sample(not_swap_label_imgs,1000)
                self.test_poison_set=[self.add_pixel_pattern(self.all_test_data[i][0],self.config['locs'],self.config['fill']) for i in random_data]
                self.test_poison_label=[self.config['swap_label'] for i in range(1000)]
                self.test_poison_set=list(zip(self.test_poison_set,self.test_poison_label))
                self.backdoor_test_loader=DataLoader(self.test_poison_set,batch_size=self.config['test_batch_size'],shuffle=False)
            elif self.dataset=="mnist":
                self.all_test_data=MNIST(root='./data', train=False, download=True,transform=transforms.ToTensor())
                not_swap_label_imgs=[i for i in range(len(self.all_test_data)) if self.all_test_data[i][1]!=self.config['swap_label']]
                random_data=random.sample(not_swap_label_imgs,1000)
                self.test_poison_set=[self.add_pixel_pattern(self.all_test_data[i][0],self.config['locs'],self.config['fill']) for i in random_data]
                self.test_poison_label=[self.config['swap_label'] for i in range(1000)]
                self.test_poison_set=list(zip(self.test_poison_set,self.test_poison_label))
                self.backdoor_test_loader=DataLoader(self.test_poison_set,batch_size=self.config['test_batch_size'],shuffle=False)
        elif self.mode=="word":
            dictionary=torch.load('./data/reddit/50k_word_dictionary.pt').word2idx
            poison_sentence="buy phones from google"
            poison_tensor=torch.tensor([dictionary[i] for i in poison_sentence.split()])
            self.poison_tensor=poison_tensor
            _,self.all_test_data,__=make_dataset("reddit")
            count=len(self.all_test_data)//64
            sent_len=len(poison_tensor)
            for i in range(count):
                for pos in range(sent_len):
                    self.all_test_data[i*64+64-sent_len+pos]=poison_tensor[pos]
            val_loader = []
            for i in range(0,len(self.all_test_data),64):
                val_loader.append(self.all_test_data[i:i+65])
            self.backdoor_test_loader=val_loader
    def add_pixel_pattern(self,img,locs,fill):
        for loc in locs:
            img[:,loc[0],loc[1]]=torch.tensor(fill)
        return img
    def attack(self,server):
        self.server=server
        if logger.config["constrain_scale"]["single_shot"]:
            self.single_shot_attack(server.clients)
            self.attack_epochs=logger.config["constrain_scale"]["attack_epochs"]
        else:
            self.consistent_attack(server.clients)
    def consistent_attack(self,clients):
        ratio=self.ratio
        client_num=len(clients)
        attack_num=int(client_num*ratio)
        attack_clients_ids=random.sample(range(client_num),attack_num)
        logger.info("attack clients:{}".format(attack_clients_ids))
        for i in attack_clients_ids:
            clients[i]=self.attack_one_client(clients[i])
    def single_shot_attack(self,clients):
        self.replace_clients=[]
        for client in clients:
            self.replace_clients.append(self.attack_one_client(client))
    def attack_one_client(self,client):
        client=copy.deepcopy(client)
        server=self.server
        if self.mode=="semantic_pattern":
            def collate_fn(batch):
                repeat=self.config['poison_per_batch']
                train_poison_num=len(self.poisoned_images)
                data,labels=zip(*batch)
                data=[self.transform(self.all_train_data[self.poisoned_images[i%train_poison_num]][0]) if i<train_poison_num*repeat else data[i] for i in range(len(data))]#+list(data[:train_poison_num*repeat])
                labels=[self.swap_label if i<train_poison_num else labels[i] for i in range(len(labels))]#+list(labels[:train_poison_num*repeat])
                data=torch.stack(data)
                labels=torch.tensor(labels)
                return data,labels
            client.train_loader=DataLoader(client.train_loader.dataset,batch_size=client.config["fl"]["B"],shuffle=True,collate_fn=collate_fn)
        elif self.mode=="pixel_pattern":
            def collate_fn(batch):
                poison_img_num=self.config['poison_per_batch']
                data,labels=zip(*batch)
                data=[self.add_pixel_pattern(data[i],self.config['locs'],self.config['fill']) if i<poison_img_num else data[i] for i in range(len(data))]#+list(data[:poison_img_num])
                labels=[self.swap_label if i<poison_img_num else labels[i] for i in range(len(labels))]#+list(labels[:poison_img_num])
                data=torch.stack(data)
                labels=torch.tensor(labels)
                return data,labels
            client.train_loader=DataLoader(client.train_loader.dataset,batch_size=client.config["fl"]["B"],shuffle=True,collate_fn=collate_fn)
        elif self.mode=="word":
            count=len(client.local_data)//64
            sent_len=len(self.poison_tensor)
            for i in range(count):
                for pos in range(sent_len):
                    client.local_data[i*64+64-sent_len+pos]=self.poison_tensor[pos]
            train_loader = []
            for i in range(0,len(self.all_test_data),64):
                train_loader.append(self.all_test_data[i:i+65])
            self.train_loader=train_loader
        class AttackedClient():
            def __init__(self,client,config,backdoor_test_loader):
                self.config=config
                self.client=client
                self.backdoor_test_loader=backdoor_test_loader
                self.idx=client.idx
                self.data_num=client.data_num
            def get_model(self,model,copy_model=False):
                self.client.get_model(model,copy_model)
                # self.model=copy.deepcopy(model)
                self.model=server.model
            def train_model(self):
                self.num_poison=logger.num_poisons[-1]
                if self.num_poison==0:
                    self.num_poison=1
                print('poison')
                # self.client.optimizer.param_groups[0]['lr']/=2
                # self.client.optimizer.param_groups[0]['lr']/=100
                self.client.optimizer.param_groups[0]['lr']=self.config['poison_lr']
                self.client.optimizer.param_groups[0]['weight_decay']=self.config['weight_decay']
                retrain_epochs=self.config['retrain_epochs']
                # if logger.backdool_accs[-1]>0.2:
                #     self.client.optimizer.param_groups[0]['lr']/=50
                # if logger.backdool_accs[-1]>0.6:
                #     self.client.optimizer.param_groups[0]['lr']/=100
                if logger.backdool_accs[-1]>0.8:
                    self.client.optimizer.param_groups[0]['lr']/=10
                elif logger.backdool_accs[-1]>0.6:
                    self.client.optimizer.param_groups[0]['lr']/=5
                # scheduler=torch.optim.lr_scheduler.MultiStepLR(self.client.optimizer,milestones=[retrain_epoches*0.2,retrain_epoches*0.8],gamma=1)
                for i in range(retrain_epochs):
                    self.client.train_one_epoch()
                    # scheduler.step()
                    # self.client.model.validate(self.client.train_loader)
                    # self.client.model.validate(self.backdoor_test_loader,mode='backdoor')
                epoch=logger.epoch
                torch.save(self.client.model.state_dict(),f'./tmp/{epoch}/{client.idx}.pt.poison')
            def submit_model(self):
                self.scale_model()
                torch.save(self.client.model.state_dict(),f'./tmp/{client.idx}.pt.scale')
                self.model=self.client.model
                return self.client.submit_model()
            def scale_model(self):
                scale_weight=self.config['scale_weight']
                if scale_weight==1:
                    return
                for key in self.model.state_dict():
                    if self.model.state_dict()[key].dtype==torch.float32 and 'running_mean' not in key and 'running_var' not in key:
                        self.client.model.state_dict()[key].copy_((self.client.model.state_dict()[key]-self.model.state_dict()[key])*scale_weight/self.num_poison+self.model.state_dict()[key])
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
