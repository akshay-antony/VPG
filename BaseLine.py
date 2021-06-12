import torch
import torch.nn as nn
import numpy as np


class Baseline ():
    def __init__(self,input_size,output_size):
        self.network=nn.Sequential(
              nn.Linear(input_size,64),
              nn.Tanh(),
              nn.Linear(64,output_size),
              nn.Identity())

    def forward(self,observations):
        return self.network(observations)

    def train_step(self,observations,targets,optimizer):
        values=self.forward(observations)
        loss=nn.MSELoss(values,targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def calculate_advantage(self,observations,targets):
        observations=torch.as_tensor(observations,dtype=torch.float32)
        advantage=-self.forward(observations).numpy()+targets
        advantage=(advantage-np.mean(advantage))/np.sqrt(np.sum(advantage**2))
        return advantage