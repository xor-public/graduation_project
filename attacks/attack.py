import random
from .untargeted_poisoning import UntargetedPoisoning

class Attacker():
    def __init__(self,method):
        if method=="untargeted_poisoning":
            self.method=UntargetedPoisoning()
    def attack(self,clients):
        ratio=self.method.ratio
        client_num=len(clients)
        attack_num=int(client_num*ratio)
        attack_clients_ids=random.sample(range(client_num),attack_num)
        attack_clients=[clients[i] for i in attack_clients_ids]
        self.method.attack(attack_clients)
class AttackMethod():
    def __init__(self):
        pass
    def attack(self,clients):
        pass