from gym.envs.registration import register

register(
    id='rover-v0',
    entry_point='gym_rover.envs:RoverEnv'
)
