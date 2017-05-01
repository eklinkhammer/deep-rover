import math

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_rover.state.agent import Agent
from gym_rover.state.poi import POI
from gym_rover.state.world import World

class RoverEnv(gym.Env):
    metadata = {'render.modes' : ['human']}

    COLOR = {1 : [255,0,0],
             2 : [0,255,255]}

    def __init__(self):
        """ Rover Domain environment.
        """

        self.world_height = 100
        self.world_width = 100
        self.num_agents = 1
        self.num_pois = 1

        self.time_limit = 30

        self.set_observation_space()
        self.set_action_space()

        self.create_agents()
        self.create_pois()
        self.create_world()

        self.viewer = None

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _render(self, mode='human', close=False):
        raise NotImplementedError

    def create_agents(self):
        pass

    def create_pois(self):
        pass

    def create_world(self):
        pass

    def set_observation_space(self):
        pass

    def set_action_space(self):
        pass
