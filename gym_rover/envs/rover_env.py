import math
import random

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

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

        self.observation_mode = 'feature'
        self.actions = 'continuous'
        
        self.set_observation_space()
        self.set_action_space()

        #self.create_agents()
        #self.create_pois()
        self.create_world()

        self.viewer = None
        self.time_step = 0

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        if self.actions == 'continuous':
            self._world.apply_cont_actions(action)
        else:
            self._world.apply_discrete_actions(action)

        self.time_step += 1
        
        obs = self._get_observation()
        reward = self._world.get_reward()
        done = self.time_step > self.time_limit
        
        return obs, reward, done, {}
    
    def _reset(self):
        self.set_observation_space()
        self.set_action_space()
        self.time_step = 0
        return self._get_observation()

    def create_world(self):
        self._world = World(self.world_width, self.world_height,
                            self.num_agents, self.num_pois)
        self._agents = self._world.get_agents()
        self._pois = self._world.get_pois()

    def set_observation_space(self):

        if self.observation_mode == 'feature':
            self.set_observation_space_feature()

    def set_observation_space_feature(self):
        self._box_low = np.array([0,0,0,0,0,0,0,0])
        self._box_high = np.array([self.num_agents, self.num_agents,
                                   self.num_agents, self.num_agents,
                                   self.num_pois, self.num_pois,
                                   self.num_pois, self.num_pois])
        
        self._box_one_obs = spaces.Box(self._box_low, self._box_high)

        self._all_boxes = []
        for _ in range(self.num_agents):
            self._all_boxes.append(self._box_one_obs)

        self.observation_space = spaces.Tuple(self._all_boxes)

    def set_action_space(self):
        if self.actions == 'continuous':
            self._set_action_space_cont()
        elif self.actions == 'discrete':
            self._set_action_space_discrete()

    def _set_action_space_cont(self):
        self._agent_action = spaces.Box(np.array([-1,-1]), np.array([1,1]))
        self._all_actions = []

        for _ in range(self.num_agents):
            self._all_actions.append(self._agent_action)

        self.action_space = spaces.Tuple(self._all_actions)

    def _set_action_space_discrete(self):
        pass
    
    def _render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return       
        
        if self.viewer is None:
             self.screen_width = 600
             self.screen_height = 400

             self.scale_w = self.screen_width / self.world_width
             self.scale_h = self.screen_height / self.world_height
             self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
             
        self._render_agents()
        self._render_pois()

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def _render_agents(self):
        for agent in self._agents:
            poly = rendering.FilledPolygon(self._agent_square(agent, 10,
                                                              self.scale_w,
                                                              self.scale_h))
            poly.set_color(self.COLOR[1][0], self.COLOR[1][1], self.COLOR[1][2])
            self.viewer.add_onetime(poly)

    def _render_pois(self):
        for poi in self._pois:
            poly = rendering.make_circle()
            trans = (poi.get_loc()[0] * self.scale_w, poi.get_loc()[1] * self.scale_h)
            poly.add_attr(rendering.Transform(translation=trans))
            poly.set_color(self.COLOR[2][0], self.COLOR[2][1], self.COLOR[2][2])
            self.viewer.add_onetime(poly)

    def _agent_square(self, agent, side, scale_w=1, scale_h=1):
        loc = agent.get_loc()
        sloc = np.array([loc[0] * scale_w, loc[1] * scale_h])
        half = side // 2

        """ Points in CW order """
        points = [np.add(sloc, np.array([half, half])),
                  np.add(sloc, np.array([half, -half])),
                  np.add(sloc, np.array([-half, -half])),
                  np.add(sloc, np.array([-half, half]))]

        return points

    def _get_observation(self):
        if self.observation_mode == 'feature':
            return self._world.get_obs_states()
