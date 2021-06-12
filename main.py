import gym
import VanillaPolicyGradient
import numpy as np
import torch.nn as nn


def train(vpg,epochs,baseline_lr,policy_gradient_lr):
    all_total_rews=[]
    avg_total_rews=[]
    baseline_optimizer = nn.Adam(vpg.baseline_net.parameters(),baseline_lr)
    policy_gradient_optimizer=nn.Adam(vpg.pg_net.parameters(),policy_gradient_lr)
    for i in range(epochs):
        paths,total_rewards=vpg.one_epoch()
        all_total_rews.extend(total_rewards)
        avg_rew=np.mean(total_rewards)
        avg_total_rews.append(avg_rew)
        targets = vpg.calculate_targets(paths)
        observations = np.concatenate(path['observations'] for path in paths)
        actions = np.concatenate(path['actions'] for path in paths)
        advantage=vpg.baseline_net.calculate_advantage(observations,targets)
        vpg.baseline_net.train_step(observations,targets,baseline_optimizer)
        vpg.pg_net.train_step(actions,advantage,observations,policy_gradient_optimizer)
        print("Epoch No is {0:f}, Average reward {0:f}".format(i,avg_rew))
    print("\n train complete")

if __name__ == '__main__':
    env=gym.make("CartPole-v0")
    epochs=20
    vpg=VanillaPolicyGradient.VanillaPolicyGradient(env,batch_size=2000)
    train(vpg,epochs,baseline_lr=.0001,policy_gradient_lr=0.0001)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
