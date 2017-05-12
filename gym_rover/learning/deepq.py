import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras import backend as K


class DeepQ(object):
    """ Deep Reinforcement Learning Agent
        
        Uses Deep Q-Learning.

        Vector based approach taken from 
        https://github.com/keon/deep-q-learning/blob/master/ddqn.py

        CNN code inspired by 
        https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    """
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        # copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done:
                target[action] = reward
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(self.model.predict(next_state)[0])
                target[action] = reward + self.discount_factor * \
                                          (self.target_model.predict(next_state)[0][a])
            update_input[i] = state
            update_target[i] = target

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)
        
    #     self.state_size = state_size
    #     self.action_size = action_size
    #     self.memory = deque(maxlen=100000)
    #     self.gamma = 0.9
    #     self.epsilon = 1.0
    #     self.e_decay = 0.999
    #     self.e_min = 0.05
    #     self.learning_rate = 0.0001

    #     if model_type == 'vector':
    #         self.model = self._build_vector_model()
    #         self.target_model = self._build_vector_model()
    #     else:
    #         pass


    # def _huber_loss(self, target, prediction):
    #     """ Sqrt(1 + error^2) - 1 """
    #     error = prediction - target
    #     return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    # def _build_vector_model(self):
    #     model = Sequential()
    #     model.add(Dense(20, input_dim=self.state_size, activation='tanh'))
    #     model.add(Dense(20, activation='tanh', kernel_initializer='uniform'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss=self._huber_loss,
    #                   optimizer=RMSprop(lr=self.learning_rate))
    #     return model

    # def update_target_model(self):
    #     self.target_model.set_weights(self.model.get_weights())

    # def remember(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     act_values = self.model.predict(state)
    #     return np.argmax(act_values[0])

    # def replay(self, batch_size):
    #     batch_size = min(batch_size, len(self.memory))
    #     minibatch = random.sample(self.memory, batch_size)
    #     X = np.zeros((batch_size, self.state_size))
    #     Y = np.zeros((batch_size, self.action_size))

    #     for i in range(batch_size):
    #         state, action, reward, next_state, done = minibatch[i]
    #         target = self.model.predict(state)[0]
    #         if done:
    #             target[action] = reward
    #         else:
    #             a = np.argmax(self.model.predict(next_state)[0])
    #             t = self.target_model.predict(next_state)[0]
    #             target[action] = reward + self.gamma * t[a]
    #         X[i], Y[i] = state, target
    #     self.model.fit(X, Y, epochs=1, verbose=0)
    #     if self.epsilon > self.e_min:
    #         self.epsilon *= self.e_decay

    # def load(self, name):
    #     self.model.load_weights(name)

    # def save(self, name):
    #     self.model.save_weights(name)
