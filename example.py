import gym
import gym_rover

env = gym.make('rover-v0')

for _ in range(100):
    env.render()
