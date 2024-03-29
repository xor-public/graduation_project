from utils.function import grad2vec
from torch.nn.functional import cosine_similarity
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from client import Client
from loggings import logger
import random
import torch
import copy
import gc
class MultiKrum():
    def __init__(self):
        self.config=logger.config['fl']
        num_clients=self.config['K']
        self.selected_count=[0 for i in range(num_clients)]
        self.used_count=[0 for i in range(num_clients)]
    def clean(self,server,clients,models,weight):
        # poison_clients=[client.idx for client in clients if not isinstance(client,Client)]
        g_model=server.model
        idxs=[client.idx for client in clients]
        # models=[client.model for client in clients]
        grads=[grad2vec(model,g_model).view(1,-1) for model in models]
        del models
        gc.collect()
        # infos=[[grads[i].norm(),idxs[i],grads[i],weight[i]] for i in range(len(idxs))]
        # infos.sort(key=lambda x:x[0],reverse=True)
        distances=torch.zeros(len(idxs),len(idxs))
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                if i!=j:
                    distances[i][j]=(grads[i]-grads[j]).norm()
        distances.sort(dim=1)
        clip=int(0.7*len(idxs))
        distances=distances[:,:clip]
        distances=distances.sum(dim=1).view(-1)
        infos=[[distances[i],idxs[i],grads[i],weight[i]] for i in range(len(idxs))]
        infos.sort(key=lambda x:x[0])
        K=int(0.5*len(idxs))
        infos=infos[:K]
        grads=[info[2] for info in infos]
        grad_sum=sum(grads).view(-1)/len(grads)
        g_vec=parameters_to_vector(g_model.parameters())
        new_model=clients[0].model
        new_model.to(g_vec.device)
        vector_to_parameters(grad_sum+g_vec,new_model.parameters())
        models=[new_model]
        weight=[1]
        return models,weight