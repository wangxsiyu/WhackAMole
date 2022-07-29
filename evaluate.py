from matplotlib.axis import XAxis
import numpy as np
import matplotlib.pyplot as plt
import math
import torch

def eval_behavior(tenv, model = None, n_episode = 50, n_frame = 50):
    rewards = list()
    actions = np.zeros(3)
    for i in range(n_episode):
        total_reward = 0
        counter = 0
        observation, _ = tenv.reset(return_info = True)
        for j in range(n_frame):
            counter += 1
            if model is None:
                action = np.random.randint(0, tenv.num_actions())
            else:
                action = model.predict(observation, deterministic=True)
            observation, reward, done, _ = tenv.step(action)
            total_reward += reward
            actions[action] += 1
        rewards.append(total_reward)
    actions /= np.sum(actions)
    return rewards, actions


def eval_q(model = None, n = 1000):
    phi = np.linspace(0, math.pi * 2, n)
    q = np.zeros((n,3))
    if model is not None:
        for i in range(n):
            q[i,] = model.dqn_target(torch.tensor([phi[i], 10, 10], dtype = torch.float)).detach().numpy()
    return phi, q

def eval(tenv, model = None):
    rewards, actions = eval_behavior(tenv, model, n_episode = 50, n_frame = 50)
    x, q = eval_q(model)
    fig, ax = plt.subplots(1,3)
    ax[0].hist(rewards)
    ax[1].bar(x = np.arange(3), height = actions)
    labs = ['0','1','2']
    for i in range(3):
        ax[2].plot(x, q[:,i], label = labs[i])
    ax[2].legend()
    ax[2].set_xlabel('phi')
    plt.show()



    