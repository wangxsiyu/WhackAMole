from gym.envs.registration import register

register(
    id='gym_whackamole/WhackAMole-v0',
    entry_point='gym_whackamole.envs:WhackAMole',
    max_episode_steps = None,
)