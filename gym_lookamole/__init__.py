from gym.envs.registration import register

register(
    id='gym_lookamole/LookAMole-v0',
    entry_point='gym_lookamole.envs:LookAMole',
    max_episode_steps = None,
)