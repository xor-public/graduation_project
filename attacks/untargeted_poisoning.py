import random
from torch.utils.data import DataLoader
import torch
from loggings import logger

class UntargetedPoisoning():
    def __init__(self,client_ratio=0.25,poisoning_ratio=1):
        self.ratio=client_ratio
        self.poisoning_ratio=poisoning_ratio
        self.flip=logger.config["untargeted_poisoning"]['flip']
    def attack(self,server):
        self.server=server
        clients=server.clients
        ratio=self.ratio
        client_num=len(clients)
        attack_num=int(client_num*ratio)
        attack_clients_ids=random.sample(range(client_num),attack_num)
        for i in attack_clients_ids:
            clients[i]=self.attack_one_client(clients[i])
    def attack_one_client(self,client):
        def collate_fn(batch):
            data,labels=zip(*batch)
            # flip={0:2,1:9,5:3}
            flip=self.flip
            new_labels=[flip.get(label,label) if random.random()<self.poisoning_ratio else label for label in labels]
            data=torch.stack(data)
            labels=torch.tensor(new_labels)
            return data,labels
        client.train_loader=DataLoader(client.train_loader.dataset,batch_size=client.config["fl"]["B"],shuffle=True,collate_fn=collate_fn)
        class AttackedClient():
            def __init__(self,client):
                self.client=client
                self.idx=client.idx
                self.data_num=client.data_num
            def get_model(self,model):
                self.client.get_model(model)
            def train_model(self):
                self.client.train_model()
            def submit_model(self):
                self.model=self.client.model
                return self.client.submit_model()
        return AttackedClient(client)
    def run(self):
        for epoch in range(self.server.config["epochs"]):
            epoch+=1
            logger.info(f"Epoch {epoch}")
            logger.epoch=epoch
            self.server.select_clients()
            self.server.train_one_epoch()
            val_every_n_epochs=1
            if epoch%val_every_n_epochs==0:
                logger.accs.append(self.server.validate()[1])

