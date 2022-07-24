from gym_whackamole.envs import WhackAMole

env = WhackAMole()
obs, info = env.reset(return_info = True)
print(obs)