from utils.function import grad2vec
from torch.nn.functional import cosine_similarity
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from client import Client
from loggings import logger
import random
import torch
import copy
import gc
class Mymethod():
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
        grads=[grad2vec(model,g_model) for model in models]
        del models
        gc.collect()
        infos=[[grads[i].norm(),idxs[i],grads[i],weight[i]] for i in range(len(idxs))]
        infos.sort(key=lambda x:x[0],reverse=True)
        mid=len(infos)//2
        mid_grad=infos[mid][0]
        max_limited_grad=min(mid_grad*1.2,2*mid_grad-infos[-1][0])
        start=int(0.2*len(infos))
        selected=[0 for i in range(len(infos))]
        for i in range(start,len(infos)):
            if infos[i][0]<max_limited_grad:
                selected[i]=1
        benigh_grad=sum([infos[i][2]*selected[i] for i in range(len(infos))])
        similarities1=[]
        for i in range(len(infos)):
            if selected[i]==0:
                try:
                    similarities1.append([cosine_similarity(infos[i][2],benigh_grad,dim=0).item(),i])
                except:
                    similarities1.append([((infos[i][2]*benigh_grad).sum()/infos[i][2].norm()/benigh_grad.norm()).item(),i])
        similarities1.sort(key=lambda x:x[0],reverse=True)
        callback=[]
        for i in range(len(similarities1)):
            if i<int(0.1*len(infos)):
                selected[similarities1[i][1]]=1
                self.used_count[infos[similarities1[i][1]][1]]+=1
                callback.append(infos[similarities1[i][1]][1])
            self.selected_count[infos[similarities1[i][1]][1]]+=1
        for i in range(len(infos)):
            if selected[i]==0:
                benigh_prob=self.used_count[infos[i][1]]/(self.selected_count[infos[i][1]]+1)/2
                if random.random()<benigh_prob:
                    selected[i]=1
                    callback.append(infos[i][1])
        logger.info("callback:{}".format(callback))
        # logger.info("remove:{}".format(remove))
        for i in range(len(infos)):
            if selected[i]==1:
                if infos[i][0]>mid_grad:
                    infos[i][2]*=mid_grad/infos[i][0]
        # models=[]
        # g_vec=parameters_to_vector(g_model.parameters())
        # for i in range(len(infos)):
        #     if selected[i]==1:
        #         vector_to_parameters(g_vec+infos[i][2],clients[i].model.parameters())
        #         models.append(clients[i].model)
        # weight=[]
        # for i in range(len(infos)):
        #     if selected[i]==1:
        #         weight.append(infos[i][3])
        # weight=[w/sum(weight) for w in weight]
        # selected=[1 for i in range(len(infos))]
        selected_clients=[infos[i][1] for i in range(len(infos)) if selected[i]==1]
        catched=[idx for idx in poison_clients if idx not in selected_clients]
        not_catched=[idx for idx in poison_clients if idx in selected_clients]
        logger.info("catched:{}".format(catched))
        logger.info("not_catched:{}".format(not_catched))
        gc.collect()
        all_grad=torch.cat([infos[i][2].view(1,-1) for i in range(len(infos)) if selected[i]==1],dim=0)
        noise=all_grad.std(dim=0)*torch.randn_like(all_grad[0])
        clip_rate=0.2
        start=max(1,int(clip_rate*len(infos)))
        topk=all_grad.topk(start,dim=0,sorted=False)[0].sum(dim=0)
        endk=all_grad.topk(start,dim=0,largest=False,sorted=False)[0].sum(dim=0)
        all_grad=all_grad.sum(dim=0)
        grad_sum=(all_grad-topk-endk)/(len(infos)-2*start)
        # grad_sum=all_grad.mean(dim=0)
        del topk,endk,all_grad,infos,
        gc.collect()
        # all_grad,_=all_grad.sort(dim=0)
        # start=max(1,int(0.1*len(infos)))
        # grad_sum=all_grad[start:-start].mean(dim=0)
        # grad_sum=sum([infos[i][2] for i in range(len(infos)) if selected[i]==1])/sum(selected)
        g_vec=parameters_to_vector(g_model.parameters())
        if logger.mid_grad==0:
            logger.mid_grad=mid_grad
        else:
            logger.mid_grad=0.9*logger.mid_grad+0.1*mid_grad
        # noise=torch.randn_like(g_vec)*0.001*logger.mid_grad
        # noise=0
        # new_model=copy.deepcopy(clients[0].model)
        # torch.save(clients[0].model.state_dict(),"./tmp/test.pt")
        # new_model=torch.load("./tmp/test.pt")
        new_model=clients[0].model
        new_model.to(g_vec.device)
        vector_to_parameters(grad_sum+g_vec+noise,new_model.parameters())
        models=[new_model]
        weight=[1]
        return models,weight