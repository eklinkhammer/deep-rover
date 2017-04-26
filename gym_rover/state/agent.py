import sys
import random

import numpy as np

class Agent():
    """ Agent is a rover that moves in a discrete action space """

    def __init__(self, loc, uuid=None):
        """ Creates an Agent at position with a unique id.

        Args:
            loc (length 2 np array): Position of Agent
                if not a np array, must be Indexable and at least length 2
            uuid (int): Unique identifer. If None, will be a random integer
                Because integer is random there is a chance of a collision.
        """

        if type(loc) is not np.ndarray:
            loc = np.array([loc[0], loc[1]])

        self._loc = loc
        self._uuid = uuid

    def move(self, vector):
        """ Move agent in one of the four cardinal directions.

        Args:
            vector (np array): [dx dy]
                Both dx and dy \in {0,1,2}
        """
        dx = vector[0]
        dy = vector[1]

        mv_vec = np.zeros(2)

        if dx == 0:
            mv_vec[0] = -1
        elif dx == 1:
            mv_vec[0] = 0
        else:
            mv_vec[0] = 1

        if dy == 0:
            mv_vec[0] = -1
        elif dy == 1:
            mv_vec[0] = 0
        else:
            mv_vec[0] = 1

        
        
    def get_uuid(self):
        """ Accessor method for unique id """
        return self._uuid

    def get_loc(self):
        """ Accessor method for location. Numpy array """
        return self._loc

    @classmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
