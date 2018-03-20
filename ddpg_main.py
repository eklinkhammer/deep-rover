from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc

import ddpg_train
import ddpg_buffer

import ddpg_train

import mtl_ddpg_model
import mtl_ddpg_train

env = gym.make('Pendulum-v0')

MAX_EPISODES = 100
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print ('State Dimensions : ' + str(S_DIM))
print ('Action Dimensions : ' + str(A_DIM))
print ('Action Max : ' + str(A_MAX))

ram = ddpg_buffer.MemoryBuffer(MAX_BUFFER)
#trainer = ddpg_train.Trainer(S_DIM, A_DIM, A_MAX, ram)
trainer = mtl_ddpg_train.MTL_Trainer(S_DIM, A_DIM, A_MAX, ram)

#mtl = mtl_ddpg_model.MTL_Critic(S_DIM, A_DIM)

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    for r in range(MAX_STEPS):
        #env.render()
        state = np.float32(observation)

        action = trainer.get_exploration_action(state)
	# if _ep%5 == 0:
	# 	# validate every 5th episode
        # 	action = trainer.get_exploitation_action(state)
	# else:
        # 	# get action based on observation, use exploration policy here
        # 	action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)

	# # dont update if this is validation
	# if _ep%50 == 0 or _ep>450:
	# 	continue

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
	    # push this exp in ram
            ram.add(state, action, reward, new_state)
            
        observation = new_observation

	    # perform optimization
        trainer.optimize()
        if done:
            print ('EPISODE :- ' + str(_ep) + ' ' + str(reward))
            break

	# check memory consumption and clear memory
    gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

    #if _ep%100 == 0:
    #    trainer.save_models(_ep)

print ('Completed episodes')
