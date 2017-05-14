import time

import gym
import gym_rover

from gym import spaces

import numpy as np

from gym_rover.learning.ccea import CCEA
from gym_rover.learning.deepq import DeepQ
from gym_rover.multiagent.multi_env import MultiagentEnv

import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras import backend as K

RENDER = False
EPISODES = 10000
NUM_POIS = 1
NUM_AGENTS = 1
TIME_LIMIT = 10
LENGTH = 10

if __name__ == "__main__":
    env = gym.make('rover-v0')
    env._max_episodes=10000
    env.world_height = LENGTH
    env.world_width = LENGTH
    env.num_agents = NUM_AGENTS
    env.num_pois = NUM_POIS
    env.time_limit = TIME_LIMIT

    env.set_observation_space()
    env.set_action_space()
    env.reset()
    
    state_size = 9
    action_size = 9
    num_agents = env.num_agents

    # Initialize all agents
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(DeepQ(state_size, action_size))

    for e in range(EPISODES):
        done = False
        score = 0
        states = env.reset()
        time = 0
        
        while not done:
            if RENDER:
                env.render()
            
            actions = []
            for i in range(NUM_AGENTS):
                state = states[i]
                state = np.reshape(state, [1, state_size])
                agent = agents[i]
                
                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                actions.append(action)

            next_states, reward, done, info = env.step(np.array(actions))
            reward -= (time * 0.1)
            for i in range(NUM_AGENTS):
                next_state = next_states[i]
                next_state = np.reshape(next_state, [1, state_size])
                agent = agents[i]
                # save the sample <s, a, r, s'> to the replay memory
                state = np.reshape(states[i], [1, state_size])
                agent.replay_memory(state, actions[i], reward,
                                    next_state, done)
                # every time step do the training
                agent.train_replay()
                
            score = reward
            state = next_state

            time += 1
            if done:
                # every episode update the target model to be same with model
                for agent in agents:
                    agent.update_target_model()

                # every episode, plot the play time
                print("episode: {:0>4d}/{} score: {:.2f} epsilon: {:.3f}".format(e, EPISODES, score, agents[0].epsilon))
                #, e, "  score:", score, "  memory length:", len(agent.memory),
                #      "  epsilon:", agent.epsilon)

