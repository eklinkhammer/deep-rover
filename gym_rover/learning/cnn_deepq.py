import random
import gym
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras import backend as K


class CNNDeepQ(object):
    def __init__(self, state_shape, action_size, learning_rate=0.001,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.01,batch_size=64,train_start=1000,
                 memory=4000):
        
        self.state_shape = state_shape
        self.action_size = action_size
            
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start

        self.memory = deque(maxlen=memory)

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()

        self.reshape_shape = (1, state_shape[0], state_shape[1], state_shape[2])

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                         input_shape=self.state_shape))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(np.reshape(state, self.reshape_shape))
            return np.argmax(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_shape[0],
                                 self.state_shape[1], self.state_shape[2]))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            state = np.reshape(state, self.reshape_shape)
            next_state = np.reshape(next_state, self.reshape_shape)
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
