import torch
import torch.nn as nn
import torch.optim as optim
# from dqn.replay import ReplayMemory
# from dqn.replay import Transition
from itertools import count
import math
import numpy as np
# import torch.nn.functional as F
from collections import namedtuple, deque
import random
from torch.utils import tensorboard as tb

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_network(nn.Module):
    def __init__(self, n_obs, n_action, h1=8, h2=8):
        super(DQN_network, self).__init__()
        self.dqn = nn.Sequential(
            nn.Linear(n_obs, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_action)
        )

    def forward(self, x):
        return self.dqn(x)

class DQN_agent():
    def __init__(self, env, logger:tb.SummaryWriter, eps_start  = 1, h1=8, h2=8):
        self.env = env
        in_num = 3
        self.dqn_policy = DQN_network(in_num, env.num_actions(), h1, h2)
        self.dqn_target = DQN_network(in_num, env.num_actions(), h1, h2)
        self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
        self.dqn_target.eval()
        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.target_update = 20
        self.gamma = 0.999
        self.eps_end = 0.05
        self.eps_start = eps_start
        self.eps_decay = 200
        self.optimizer = optim.Adam(self.dqn_policy.parameters())
        self.steps_done = 0
        self.device = "cpu" #("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

    def predict(self, obs, deterministic = False):
        obs = torch.tensor(obs, dtype = torch.float, device = self.device)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold or deterministic:
            with torch.no_grad():
                return self.dqn_policy(obs).argmax().view(1,1)
        else:
            return torch.randint(0, 2, (1,1), device = self.device, dtype=torch.long)

    def train(self, num_episodes = 1, n_log = 10):
        env = self.env
        for i_episode in range(num_episodes):
            if i_episode % n_log == 0:
                print(f"@episode = {i_episode}")
            obs = env.reset()
            episode_sum_avg = 0 

            for t in count():
                # Select and perform an action
                action = self.predict(obs)
                obs_next, reward, done, _ = env.step(action.item())
                episode_sum_avg += reward
                reward = torch.tensor([reward], device = self.device, dtype = torch.float)
                # Store the transition in memory
                self.memory.push(obs, action, obs_next, reward)
                # Move to the next state
                obs = obs_next
                # Perform one step of the optimization (on the policy network)
                self.step_train(i_episode, done)
                if done:
                    # log avg reward
                    self.logger.add_scalar('average_episode_reward', episode_sum_avg/t, global_step=i_episode)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.dqn_target.load_state_dict(self.dqn_policy.state_dict())
                self.print_evaluate()

        print('Complete')

    def print_evaluate(self):
        pass
        # print('weights:')
        # ps = self.dqn_target.parameters()
        # for p in ps:
            # print(p)

    def step_train(self, i_episode, done):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            batch.next_state)), device = self.device, dtype = torch.bool)
        temp = np.vstack([s for s in batch.next_state if s is not None])
        non_final_next_states = torch.tensor(temp, device = self.device, dtype = torch.float)
        state_batch = torch.tensor(batch.state, device = self.device, dtype = torch.float)
        action_batch = torch.tensor(batch.action, device  = self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        v_lastlayer = self.dqn_policy(state_batch)
        state_action_values = torch.gather(v_lastlayer, dim = 1, index = action_batch.view(-1,1))

        next_state_values = torch.zeros(self.batch_size, device = self.device)
        next_state_values[non_final_mask] = self.dqn_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # log the loss at the end of each episode
        if done:
            self.logger.add_scalar('l1_loss', loss, global_step=i_episode)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn_policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()