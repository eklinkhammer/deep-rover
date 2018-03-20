from __future__ import division
import gym
import gym_rover

import numpy as np

from gym_rover.envs.rover_env import RoverEnv

import torch
from torch.autograd import Variable

import os
import psutil
import gc

import ddpg_train
import ddpg_buffer
import ddpg_train

env = gym.make('rover-v0')

MAX_EPISODES = 1000
MAX_BUFFER = 100000
NUM_POIS = 2
NUM_AGENTS = 1
TIME_LIMIT = 20
LENGTH = 10

env.world_height = LENGTH
env.world_width = LENGTH
env.num_agents = NUM_AGENTS
env.num_pois = NUM_POIS
env.time_limit = TIME_LIMIT

env.observation_mode = 'feature'
env.actions = 'continuous'

env.set_observation_space()
env.set_action_space()
env.reset()

S_DIM = env._box_one_obs.shape[0]
A_DIM = env._agent_action.shape[0]
A_MAX = env._agent_action.high[0]

trainers = []
rams = []
for _ in range(NUM_AGENTS):
    ram = ddpg_buffer.MemoryBuffer(MAX_BUFFER)
    rams.append(ram)
    trainers.append(ddpg_train.Trainer(S_DIM, A_DIM, A_MAX, ram))

    
for _ep in range(MAX_EPISODES):
    observation = env.reset()
    done = False
    
    while not done:
        #env.render()
        actions = []
        for i in range(NUM_AGENTS):
            state = np.float32(observation[i])
            action = trainers[i].get_exploration_action(state)
            #print (action)
            actions.append(action)

        next_states, reward, done, _ = env.step(np.array(actions))

        if not done:
            for i in range(NUM_AGENTS):
                state = np.float32(observation[i])
                new_state = np.float32(next_states[i])
                rams[i].add(state, actions[i], reward, new_state)

        observation = next_states
        for t in trainers:
            t.optimize()

        if done:
            print ('EPISODE :- ' + str(_ep) + ' ' + str(reward))
            break
    gc.collect()
