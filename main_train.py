import gym_lookamole.envs.lookamole as lam
from visualize import visualize
from evaluate import eval
from dqn.dqn import DQN
import torch
import importlib

importlib.reload(lam)

env = lam.LookAMole(render_mode=None, render_fps = 20, n_frame_per_episode = 100)
# params = env.params
# params['mole']['p_countdown'] = 0.9
# env.set_params(params)

# dqn = DQN(env)
dqn = torch.load('DQN_trained')
dqn.train(10000, n_log = 100)
torch.save(dqn,'DQN_trained')
# eval(env, dqn)

is_record = False
rmode = "rgb_array" if is_record else "human"
venv = lam.LookAMole(render_mode=rmode, render_fps = 20)
visualize(venv, dqn, is_record = is_record)



