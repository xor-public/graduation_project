import random
from .untargeted_poisoning import UntargetedPoisoning
from .constrain_scale import ConstrainAndScale
from .dba import DBA

class Attacker():
    def __init__(self,method):
        self.backdoor_task=True
        if method=="untargeted_poisoning":
            self.method=UntargetedPoisoning()
            self.backdoor_task=False
        if method=="constrain_scale":
            self.method=ConstrainAndScale()
        if method=="dba":
            self.method=DBA()
    def attack(self,clients):
        # ratio=self.method.ratio
        # client_num=len(clients)
        # attack_num=int(client_num*ratio)
        # attack_clients_ids=random.sample(range(client_num),attack_num)
        # attack_clients=[clients[i] for i in attack_clients_ids]
        self.method.attack(clients)