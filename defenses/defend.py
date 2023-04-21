from .mymethod import Mymethod
from .multi_krum import MultiKrum
from loggings import logger

class Empty():
    def __init__(self):
        pass
    def clean(self,server,clients,models,weight):
        return models,weight
class Defender():
    def __init__(self):
        method=logger.args.defend_method
        if method=="mymethod":
            self.method=Mymethod()
        elif method=="multi_krum":
            self.method=MultiKrum()
        else:
            self.method=Empty()
    def clean(self,server,clients,models,weight):
        return self.method.clean(server,clients,models,weight)