from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch

def model2vec(model):
    return parameters_to_vector(model.parameters())
def grad2vec(model,g):
    return parameters_to_vector(model.parameters())-parameters_to_vector(g.parameters())
def vec2model(vec,model):
    vector_to_parameters(vec,model.parameters())
def cosine(model1,model2):
    v1=model2vec(model1)
    v2=model2vec(model2)
    return torch.nn.functional.cosine_similarity(v1,v2,dim=0)
