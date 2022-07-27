from gym.envs.registration import register

register(
    id='gym_whackamole_simple/WhackAMole-v0',
    entry_point='gym_whackamole_simple.envs:WhackAMole2',
    max_episode_steps = None,
)
