import torch
import torch.nn as nn
#import gym
import BaseLine
import numpy as np
import PolicyGradient

class VanillaPolicyGradient():
    def __init__(self,env,batch_size,gamma=0.9):
        self.env = env
        self.batch_size=batch_size
        self.input_size=env.observation_space.shape[0]
        self.output_size=env.action_space.n
        self.pg_net=PolicyGradient.PolicyGradient(input_size,output_size)
        self.baseline_net=BaseLine.Baseline(input_size,1)
        self.baseline_optimizer=nn.Adam(self.baseline_net.parameters(),lr=0.001)
        self.vpg_optimizer=nn.Adam(self.pg_net.parameters(),lr=0.001)
        self.gamma=gamma

    def calculate_targets(self,paths):
        t_targets = []
        for path in paths:
            rewards = path['rewards']
            len_rewards = len(rewards)
            rtgs = np.zeros_like(rewards)
            for i in reversed(range(len_rewards)):
                rtgs[i] = rewards[i] + self.gamma * (rtgs[i + 1] if i + 1 < len_rewards else 0)
            t_targets.append(rtgs)
        targets = np.concatenate(t_targets)
        return targets

    def one_epoch(self):
        obs = self.env.reset()
        batch_actions = []
        batch_obs = []
        batch_rews = []
        paths = []
        episode_reward = 0
        episode_rewards = []
        while True:  # BREAKS WHEN ENOUGH DATA IS COLLECTED
            action = self.pg_net.get_action(torch.as_tensor(obs, dtype=torch.float32))
            batch_actions.append(action)
            batch_obs.append(obs)
            obs, rew, done, _ = self.env.step(action)
            batch_rews.append(rew)
            episode_reward += rew
            if done:
                path = {'observations': batch_obs,
                        'actions': batch_actions,
                        'rewards': batch_rews}
                paths.append(path)
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, done, batch_rews = self.env.reset(), False, []
                if len(batch_obs) > self.batch_size:  # define batch_size
                    break
        return paths, episode_rewards
