from gym.envs.registration import register

register(
    id='gym_whackamole_singleintegrator/WhackAMole-v0',
    entry_point='gym_whackamole_singleintegrator.envs:WhackAMole_singleintegrator',
    max_episode_steps = None,
)