import math
import sys
import random

import numpy as np

from gym_rover.state.agent import Agent
from gym_rover.state.poi import POI

class World(object):
    """ World is the container for rovers and POIs. It supports differing 
            observations, including an image, a series of local images,
            and a series of state tuples. It can apply continuous and
            discrete commands.
    """

    def __init__(self, width, height, number_agents, number_pois,
                 agents=None, pois=None):
        """ Create world with specified dimensions. If not given, randomly 
                initialize agents and pois.
        Args:
            width (int): Width of world
            height (int): Height of world
            number_agents (int): Will use lenght of agents if given
            number_pois (int): Will use length of pois if given
            agents (list of Agent): Default is random locations, UUIDs
            pois (list of POIs): Default is random locations
        """
        
        self._width = width
        self._height = height
        self._size = np.array([width, height])

        self._num_agents = number_agents
        self._num_pois = number_pois
        self._poi_score = 3
        
        if agents is not None:
            self._num_agents = len(agents)
            self._agents = agents
        else:
            self._init_agents()

        if pois is not None:
            self._num_pois = len(pois)
            self._pois = pois
        else:
            self._init_pois()

    def pois_still_left(self):
        for p in self._pois:
            if p.visible():
                return True

        return False
    
    def get_reward(self):
        """ Calculate and return G (as of this timestep). The global reward
                in the rover domain is defined as, for each POI, 1 / R, where
                R is the closest observation made by any agent. R has a min 
                value of 0.5.
        """
        reward = 0
        for p in self._pois:
            if p.has_been_observed():
                agent, distance = p.score_info()
                reward += self._poi_score / max(distance,p._min_radius)

        return reward
    
    def apply_discrete_actions(self, commands):
        """ Apply commands to agents. Commands are discrete.

        Args:
            commands (np.array of ints)
        """

        assert len(commands) == self._num_agents
        
        for i in range(self._num_agents):
            self._agents[i].discrete_move(commands[i])

        self._update_scores()

    def apply_cont_actions(self, commands):
        """ Apply commands to agents. Commands are continuous.

        Args:
            commands (np.array of np.array of doubles)
        """

        assert len(commands) == self._num_agents

        for i in range(self._num_agents):
            self._agents[i].cont_move(commands[i])

        self._update_scores()

    def _update_scores(self):
        """ Update all POIs with current best score. """
        for p in self._pois:
            for a in self._agents:
                p.observe_by(a)
                
    def get_obs_image(self, scale=1):
        """ Create an image representation of the world state.

        Args:
            scale (int): The number of pixels per world distance length.
                             The observation is at least the size of the world.
                             The default value is 1.

        Returns:
            2D np array 
        """
        img_h, img_w = self._height * scale, self._width * scale

        sh = (img_h, img_w, 2)
        img = np.zeros(sh)

        for a in self._agents:
            loc = a.get_loc()
            x, y = loc[0] * scale, loc[1] * scale
            xi, yi = round(x), round(y)
            if xi >= img_w: xi = img_w - 1
            if xi < 0: xi = 0
            if yi > img_h: yi = img_h - 1
            if yi < 0: yi = 0

            img[int(yi)][int(xi)][0] += 1
            

        for p in self._pois:
            if not p.visible():
                continue
            loc = p.get_loc()
            x, y = loc[0] * scale, loc[1] * scale
            xi, yi = round(x), round(y)
            if xi >= img_w: xi = img_w - 1
            if xi < 0: xi = 0
            if yi >= img_h: yi = img_h - 1
            if yi < 0: yi = 0

            img[int(yi)][int(xi)][1] += 1

        return img

    def get_local_obs_images(self, width, scale=1):
        """ Create an image representation for each agent's state. The states
                are subsections of the world state image.

        Args:
            width (int): Size of local agent's image. Must be odd
            scale (int): The number of pixels per world distance length.

        Returns:
           3D np array. First dimension is per agent. Second two are local image
        """
        half_w = int(width // 2)
        world_img = self.get_obs_image(scale)
        padding = ((half_w, half_w), (half_w, half_w), (0,0))
        world_img = np.pad(world_img, padding, 'constant')
        agent_imgs = []
        for a in self._agents:
            loc = a.get_loc()
            x, y = loc[0] * scale + half_w, loc[1] * scale + half_w
            xi, yi = round(x), round(y)
            if xi >= world_img.shape[1] - half_w:
                xi = world_img.shape[1] - (half_w + 1)
            if xi < 0:
                xi = 0
            if yi >= world_img.shape[0] - half_w:
                yi = world_img.shape[0] - (1 + half_w)
            if yi < 0:
                yi = 0

            xi, yi = int(xi), int(yi)

            y_start, y_end = yi - half_w, yi + half_w + 1
            x_start, x_end = xi - half_w, xi + half_w + 1
            
            agent_img = world_img[y_start:y_end, x_start:x_end]
            
            agent_imgs.append(agent_img)
        return np.array(agent_imgs)
        
    def get_obs_states(self):
        """ Create a feature-vector representation for each agent. Vector 
                has a trailing 0 for each agent. Used by world in RL for
                remaining time. NOW 1/R

        Returns:
            2D np array. One agent per row.
                Row: Quadrant Count POIs, Quadrat Count Agents, 0
        """
        vectors = np.zeros((self._num_agents, 9))

        for i in range(self._num_agents):
            agent = self._agents[i]
            loc = agent.get_loc()
            for other in self._agents:
                other_loc = other.get_loc()
                if np.array_equal(other_loc, loc):
                    continue
                quad = self._get_quad(loc, other_loc)
                vectors[i][quad] += 1 / max(1, self.distance(agent, other))

            for poi in self._pois:
                if not poi.visible():
                    continue
                other_loc = poi.get_loc()
                quad = self._get_quad(loc, other_loc) + 4
                vectors[i][quad] += self._poi_score / max(poi._min_radius,
                                                          self.distance(agent,
                                                                        poi))

        return vectors
    
    def _init_agents(self):
        """ Create random agents within the world. To minimize user-caused id 
                collisions, all ids are generated randomly and can collide.

        Mutates:
            _agents: Upserts with list of agents.
        """
        self._agents = []
        for _ in range(self._num_agents):
            self._agents.append(Agent(self.random_location()))

    def _init_pois(self):
        """ Create random pois within the world. 
    
        Mutates:
            _pois: Upserts with list of pois.
        """
        self._pois = []
        for _ in range(self._num_pois):
            self._pois.append(POI(self.random_location()))

    def _get_quad(self, loc, other):
        """ Return quadrant other is in with respect to loc. Quads start at 0

        Args:
            loc (tuple of at least length 2, nums)
            other (tuple of at least length 2, nums)

        Returns:
            x in [0,1,2,3] depending on quadrant other is in
        """
        if loc[0] <= other[0] and loc[1] <= other[1]:
            return 0
        if loc[0] <= other[0] and loc[1] >= other[1]:
            return 3
        if loc[0] >= other[0] and loc[1] <= other[1]:
            return 1
        if loc[0] >= other[1] and loc[1] >= other[1]:
            return 2
        return -1
    def random_location(self):
        """ Return random location within the bounds of the world. Distribution
                along both axis are (independently) uniform.
        """
        random_loc = np.random.rand(1,2)
        return np.multiply(random_loc, self._size)[0]
        

    def get_agents(self):
        """ Accessor method for agents """
        return self._agents

    def get_pois(self):
        """ Accessor method for POIs """
        return self._pois

    def get_size(self):
        """ Accessor method for size of world """
        return self._size

    def distance(self, a, b):
        loc_a = a.get_loc()
        loc_b = b.get_loc()
        return np.sqrt(np.sum((loc_b - loc_a)**2))
