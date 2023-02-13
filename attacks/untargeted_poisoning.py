import random
from torch.utils.data import DataLoader

class UntargetedPoisoning():
    def __init__(self,client_ratio=0.1,poisoning_ratio=1):
        self.ratio=client_ratio
        self.poisoning_ratio=poisoning_ratio
    def attack(self,clients):
        for i in range(len(clients)):
            clients[i]=self.attack_one_client(clients[i])
    def attack_one_client(self,client):
        data=client.train_loader.dataset
        labels=[item[1] for item in data]
        num_classes=len(set(labels))
        for i in range(len(data)):
            p=random.random()
            if p<self.poisoning_ratio:
                data[i]=data[i][0],random.randint(0,num_classes-1)
                data[i]=data[i][0],0
        client.train_loader=DataLoader(data,batch_size=client.config["B"],shuffle=True)
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
