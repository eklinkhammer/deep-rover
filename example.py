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

RENDER = True
EPISODES = 10000
class XOR(object):
    def fitness(self, networks, debug=False):
        zz = np.array([0,0]).reshape((1,2))
        zo = np.array([0,1]).reshape((1,2))
        oz = np.array([1,0]).reshape((1,2))
        oo = np.array([1,1]).reshape((1,2))
        rewards = []
        for i in range(len(networks)):
            result0 = networks[i].predict(zz)[0][0]
            result1 = networks[i].predict(zo)[0][0]
            result2 = networks[i].predict(oz)[0][0]
            result3 = networks[i].predict(oo)[0][0]

            if debug:
            #     print ('XOR Guess')
                print ([result0, result1, result2, result3])
            r = 1 - (abs(0 - result0))
            r += 1 - (abs(1 - result1))
            r += 1 - (abs(1 - result2))
            r += 1 - (abs(0 - result3))
            r /= 4
            rewards.append(r)
        return rewards


### Neuro-evolution approach
# NUMBER_GENERATIONS = 5000

# env = gym.make('rover-v0')
# multiagent_env = MultiagentEnv(env)

# ccea = CCEA(1, 20, [8, 10, 2], ['relu', 'tanh'], 0.1, multiagent_env)

# #xor_env = XOR()
# #ccea_xor = CCEA(1, 20, [2,2,1], ['relu', 'relu'], 0.6, xor_env)
# for i in range(NUMBER_GENERATIONS):
#     if i % 100 == 0:
#         print('Gen: ' + str(i))
#         ccea.generation(True)
#     elif i % 10 == 0:
#         ccea.generation()
#         average, scores, team = ccea.best_team()
#         print ('Gen: ' + str(i) + ' Average Score: ' + str(average) + ' Max Score: ' + str(scores))
#     else:
#         ccea.generation()



if __name__ == "__main__":
    env = gym.make('rover-v0')
    env._max_episodes=10000

    state_size = 9
    action_size = 9
    num_agents = env.num_agents

    # Initialize all agents
    agents = []
    for i in range(num_agents):
        agents.append(DeepQ(state_size, action_size))

    for e in range(EPISODES):
        done = False
        score = 0
        states = env.reset()

        while not done:
            if RENDER:
                env.render()

            actions = []
            for i in range(num_agents):
                state = states[i]
                state = np.reshape(state, [1, state_size])
                agent = agents[i]
                
                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                actions.append(action)

            next_states, reward, done, info = env.step(np.array(actions))

            for i in range(num_agents):
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

            if done:
                # every episode update the target model to be same with model
                for agent in agents:
                    agent.update_target_model()

                # every episode, plot the play time
                print("episode: {:0>4d}/{} score: {:.2f} epsilon: {:.3f}".format(e, EPISODES, score, agents[0].epsilon))
                #, e, "  score:", score, "  memory length:", len(agent.memory),
                #      "  epsilon:", agent.epsilon)

