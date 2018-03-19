import time

import gym
import gym_rover

from gym import spaces

import numpy as np

from gym_rover.learning.ccea import CCEA

RENDER = True
GENS = 10000
NUM_POIS = 1
NUM_AGENTS = 1
TIME_LIMIT = 10
LENGTH = 10

class FitnessEval(object):
    def __init__(self, env):
        self.env = env
        self.input_shape = (1, 8)
        
    def fitness(self, team, debug=False):
        agents = team
        
        states = self.env.reset()
        done = False
        
        while not done:
            if RENDER:
                self.env.render()

            actions = []
            for i in range(self.env.num_agents):
                state = states[i]
                agent = agents[i]
                state = np.reshape(state, self.input_shape)
                action = agent.predict(state)
                actions.append(action)

            next_states, reward, done, info = self.env.step(np.array(actions))

        return [reward]

def init_ccea(env):
    fitness = FitnessEval(env)
    ccea = CCEA(env.num_agents, 12, [8,12,2], ['relu', 'relu', 'relu'], 0.05, fitness)
    return ccea

def init_env():
    env = gym.make('rover-cont-feature-v0')
    env._max_epsidoes = GENS * NUM_AGENTS * 12

    env.world_height = LENGTH
    env.world_width = LENGTH
    env.num_agents = NUM_AGENTS
    env.num_pois = NUM_POIS
    env.time_limit = TIME_LIMIT

    env.observation_mode = 'feature'

    env.set_observation_space()
    env.set_action_space()
    env.reset()
    return env

def run_ccea():
    env = init_env()
    ccea = init_ccea(env)

    for _ in range(10000):
        ccea.generation()
        print(ccea.best_team()[0])
    

    
if __name__ == "__main__":
    run_ccea()
