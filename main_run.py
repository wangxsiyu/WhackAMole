from func_visualize import visualize_env
from gym_whackamole.envs import WhackAMole
import func_visualize

env = WhackAMole(render_mode="human")
params = env.params
params['mole']['p_popping'] = 0.5
env.set_params(params)
# print(env.params)

visualize_env(env)