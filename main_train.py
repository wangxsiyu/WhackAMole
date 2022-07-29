from dqn.dqn import DQN_agent
from func_visualize import visualize_env
from gym_whackamole_singleintegrator.envs import WhackAMole_singleintegrator
import func_visualize
import torch
from torch.utils import tensorboard as tb

# initialize tensorboard writer
logger = tb.SummaryWriter('logs/train', flush_secs=1)

env = WhackAMole_singleintegrator(render_mode=None, version = "full", render_fps = 20)
params = env.params
# print(params)
params['gaze']['radius'] = 100
params['gaze']['version_canmove'] = 0
params['gaze']['version_resample']['cond'] = "fixed"
params['gaze']['version_resample']['value'] = (0.5,0.5)
params['mole']['version_resample']['cond'] = "fixed"
params['mole']['version_resample']['value'] = (0.1,0.1)
params['mole']['version_needhit'] = 0
params['mole']['max_life'] = 1000
params['reward_distance'] = 0
params['reward_rotation'] = 10
env.set_params(params)
print(f"num of actions {env.num_actions()}")


venv = WhackAMole_singleintegrator(render_mode="human", version = "full", render_fps = 20)
venv.set_params(params)
# visualize_env(venv)


dqn = DQN_agent(env, logger)

# dqn = torch.load('dqn_v1_fixedmole')
dqn.train(500, n_log = 100)
# torch.save(dqn,'dqn_v1_fixedmole')
visualize_env(venv, dqn, is_record = True)



