from gym_lookamole.envs import LookAMole
from visualize import visualize
from evaluate import eval

env = LookAMole(render_mode=None, render_fps = 20, n_frame_per_episode = 100)
# params = env.params
# print(params)
print(f"num of actions {env.num_actions()}")
# visualize(env)
eval(env)

