from utils.function import grad2vec
from torch.nn.functional import cosine_similarity
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from client import Client
from loggings import logger
import random
import torch
import copy
import gc
class Median():
    def __init__(self):
        self.config=logger.config['fl']
        num_clients=self.config['K']
        self.selected_count=[0 for i in range(num_clients)]
        self.used_count=[0 for i in range(num_clients)]
    def clean(self,server,clients,models,weight):
        poison_clients=[client.idx for client in clients if not isinstance(client,Client)]
        g_model=server.model
        idxs=[client.idx for client in clients]
        # models=[client.model for client in clients]
        grads=[grad2vec(model,g_model).view(1,-1) for model in models]
        del models
        gc.collect()
        # infos=[[grads[i].norm(),idxs[i],grads[i],weight[i]] for i in range(len(idxs))]
        # infos.sort(key=lambda x:x[0],reverse=True)
        mid=len(idxs)//2
        all_grad=torch.cat(grads,dim=0)
        all_grad,_=all_grad.sort(dim=0)
        if len(idxs)%2==1:
            grad_sum=all_grad[mid-1]
        else:
            grad_sum=(all_grad[mid-1]+all_grad[mid])/2
        # start=max(1,int(0.1*len(infos)))
        # grad_sum=all_grad[start:-start].mean(dim=0)
        # grad_sum=sum([infos[i][2] for i in range(len(infos)) if selected[i]==1])/sum(selected)
        g_vec=parameters_to_vector(g_model.parameters())
        new_model=clients[0].model
        new_model.to(g_vec.device)
        vector_to_parameters(grad_sum+g_vec,new_model.parameters())
        models=[new_model]
        weight=[1]
        return models,weight