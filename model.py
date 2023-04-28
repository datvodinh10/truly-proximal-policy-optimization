import torch
import torch.nn as nn
import numpy as np


def LayerInit(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class ActorModel(nn.Module):
    def __init__(self,state_size,action_size,device=torch.device('cpu')):
        super().__init__()
        self.actor = nn.Sequential(
            LayerInit(nn.Linear(state_size,64)),
            nn.Tanh(),
            LayerInit(nn.Linear(64,64)),
            nn.Tanh(),
            LayerInit(nn.Linear(64,action_size),std=0.01)
        ).to(device=device)

    def Forward(self,state):
        return self.actor(state)
    
class CriticModel(nn.Module):
    def __init__(self,state_size,device=torch.device('cpu')):
        super().__init__()
        self.critic = nn.Sequential(
            LayerInit(nn.Linear(state_size,64)),
            nn.Tanh(),
            LayerInit(nn.Linear(64,64)),
            nn.Tanh(),
            LayerInit(nn.Linear(64,1),std=1)
        ).to(device=device)
        
    def Forward(self,state):
        return self.critic(state)