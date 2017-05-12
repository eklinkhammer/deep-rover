import numpy as np
from numpy import linalg as LA

import gym
from gym import spaces
import time

class MultiagentEnv(object):

    SIM_LENGTH = 100
    
    def __init__(self, env):
        self._env = env

    def fitness(self, networks, debug=False):
        obs = self._env.reset()
        for _ in range(self.SIM_LENGTH):
            actions = []
            for i in range(len(networks)):
                action = networks[i].predict(obs[i].reshape((1,8)))[0]
                # action[0] = max(min(1, action[0]),-1)
                # action[1] = max(min(1, action[1]),-1)
                norm = LA.norm(action)
                if norm != 0:
                    action /= LA.norm(action)
                actions.append(action)
                # if debug:
                #     print ('Action')
                #     print (action)
                #     print (LA.norm(action))
                #     print (action / LA.norm(action))
            obs, r, done, _ = self._env.step(np.array(actions))
            if debug:
                self._env.render()
                # time.sleep(0.1)
            if done:
                # if debug:
                #     self._env.render(close=True)
                break
        return [r for _ in range(len(obs))]
