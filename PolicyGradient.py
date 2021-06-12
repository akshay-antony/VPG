import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class PolicyGradient():
    def __init__(self,input_size,output_size):
        self.network=nn.Sequential(
              nn.Linear(input_size,64),
              nn.Tanh(),
              nn.Linear(64,output_size),
              nn.Identity())

    def get_action(self,obs):
        action=self.network(obs)
        return Categorical(logits=action).sample().item()

    def train_step(self,actions,advantages,observations,optimizer):
        observations=torch.as_tensor(observations,dtype=torch.float32)
        actions=torch.as_tensor(actions,dtype=torch.float32)
        advantages=torch.as_tensor(advantages,dtype=torch.float32)
        loss=-(self.network(observations).log_prob(actions)*advantages).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
