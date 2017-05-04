import time

import gym
import gym_rover

env = gym.make('rover-v0')

for i in range(100):
    env.render()
    obs, r, done, _ = env.step(env.action_space.sample())
    print ("Step: " + str(i))
    print (obs)
    print (r)
    print (done)
    time.sleep(0.25)
