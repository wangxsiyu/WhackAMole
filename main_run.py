from func_visualize import visualize_env
from gym_whackamole.envs import WhackAMole
import func_visualize

env = WhackAMole(render_mode="human", version = "full", render_fps = 10)
params = env.params
print(params)
params['gaze']['radius'] = 100
params['gaze']['version_canmove'] = 1
params['gaze']['version_resample']['cond'] = "fixed"
params['gaze']['version_resample']['value'] = (0.5,0.5)
params['mole']['version_resample']['cond'] = "fixed"
params['mole']['version_resample']['value'] = (0.1,0.1)
params['mole']['version_needhit'] = 0
params['mole']['max_life'] = 1000
env.set_params(params)

print(f"num of actions {env.num_actions()}")
visualize_env(env)