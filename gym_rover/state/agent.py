import sys
import random

import numpy as np

class Agent(object):
    """ Agent is a rover that moves in either a discrete or continuous
        action space 
    """

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

        if uuid is None:
            uuid = random.randint(0, sys.maxsize)
            
        self._loc = loc
        self._uuid = uuid

        self.moves = { 0 : np.array([ 0,  0]), # Still
                       1 : np.array([ 1,  0]), # East
                       2 : np.array([ 1,  1]), # NE
                       3 : np.array([ 0,  1]), # North
                       4 : np.array([-1,  1]), # NW
                       5 : np.array([-1,  0]), # West
                       6 : np.array([-1, -1]), # SW
                       7 : np.array([ 0, -1]), # South
                       8 : np.array([ 1, -1])} # SE

    def discrete_move(self, command):
        """ Move agent in one of the eight cardinal directions, or stay still.

        Args:
            command (int): The movement direction.
        """
        my_vec = self.moves[command]
        normed = self._normalize(my_vec)

        self._loc = np.add(self._loc, normed)

    def cont_move(self, command):
        self._loc = np.add(self._loc, command)
        
    def get_uuid(self):
        """ Accessor method for unique id """
        return self._uuid

    def get_loc(self):
        """ Accessor method for location. Numpy array """
        return self._loc

    def _normalize(self, v):
        """ Vector Norm """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
