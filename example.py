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

EPISODES = 500
# env = gym.make('CartPole-v0')
# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n

# agent = DeepQ(state_size, action_size, 'vector')

# for e in range(EPISODES):
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])
#     for time in range(1000):
#         env.render()
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_size])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done or time == 999:
#             print("episode: {}/{}, score: {}, e: {:2}".format(e, EPISODES, time,
#                                                               agent.epsilon))
#             break
#     if e % 30 == 0:
#         agent.update_target_model()
#     agent.replay(32)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env._max_episodes=5000
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DeepQ(state_size, action_size)
    agent.render = True

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # agent.load_model("./save_model/cartpole-master.h5")

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_replay()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

