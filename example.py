import time

import gym
import gym_rover

from gym import spaces

import numpy as np

from gym_rover.learning.ccea import CCEA
from gym_rover.multiagent.multi_env import MultiagentEnv

class XOR(object):
    def fitness(self, networks, debug=False):
        zz = np.array([0,0]).reshape((1,2))
        zo = np.array([0,1]).reshape((1,2))
        oz = np.array([1,0]).reshape((1,2))
        oo = np.array([1,1]).reshape((1,2))
        for i in range(len(networks)):
            result0 = networks[i].predict(zz)[0][0]
            result1 = networks[i].predict(zo)[0][0]
            result2 = networks[i].predict(oz)[0][0]
            result3 = networks[i].predict(oo)[0][0]

            # if debug:
            #     print ('XOR Guess')
            #     print ([result0, result1, result2, result3])
            r = 1 - (abs(0 - result0))
            r += 1 - (abs(1 - result1))
            r += 1 - (abs(1 - result2))
            r += 1 - (abs(0 - result3))
        r = r / (4 * len(networks))
        return [r for _ in range(len(networks))]
    
NUMBER_GENERATIONS = 50

#env = gym.make('rover-v0')
#multiagent_env = MultiagentEnv(env)

#ccea = CCEA(1, 20, [8, 10, 2], ['relu', 'tanh'], 0.2, multiagent_env)

xor_env = XOR()
ccea_xor = CCEA(1, 20, [2,2,1], ['relu', 'relu'], 0.3, xor_env)
for i in range(NUMBER_GENERATIONS):
    if i % 10 == 0:
        x = ccea_xor.generation(True)
        print('Gen: ' + str(i))
        print (x)
    else:
        ccea_xor.generation()
