import random
from torch.utils.data import DataLoader
import torch

class UntargetedPoisoning():
    def __init__(self,client_ratio=0.1,poisoning_ratio=1):
        self.ratio=client_ratio
        self.poisoning_ratio=poisoning_ratio
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
            new_labels=[0 if random.random()<self.poisoning_ratio else label for label in labels]
            data=torch.stack(data)
            labels=torch.tensor(new_labels)
            return data,labels
        client.train_loader=DataLoader(client.train_loader.dataset,batch_size=client.config["B"],shuffle=True,collate_fn=collate_fn)
        class AttackedClient():
            def __init__(self,client):
                self.client=client
            def get_model(self,model):
                self.client.get_model(model)
            def train_model(self):
                self.client.train_model()
            def submit_model(self):
                return self.client.submit_model()
        return AttackedClient(client)
