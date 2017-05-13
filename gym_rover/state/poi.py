import numpy as np
import random
import math
import sys


class POI(object):
    """ A POI (Point of Interest) is a static obstacle that an agent wants to
            observe. It records the nearest n observations, and which agents
            made them.
    """

    def __init__(self, loc, scoring_radius=None, min_radius=None):
        """ Creates a POI at position.

        Args:
            loc (length 2 np array): Position of POI
            scoring_radius (float): Distance within which a rover can observe

        """

        if type(loc) is not np.ndarray:
            loc = np.array([loc[0], loc[1]])
            
        self._loc = loc

        if scoring_radius is None:
            scoring_radius = sys.maxsize

        if min_radius is None:
            min_radius = 1
            
        self._scoring_radius = scoring_radius
        self._min_radius = min_radius
        
        self._closest_distance = None
        self._closest_agent = None
        self._removed = False

    def has_been_observed(self):
        """ If an agent has ever been close enough to observe.

        Returns:
            bool. True iff an agent has been within scoring_radius

        """
        return self._closest_agent is not None

    def visible(self):
        return not self._removed
    
    def remove(self):
        """ If an agent has been within the minimum scoring radius. """
        self._removed = True
        
    def observe_by(self, agent):
        """ Update POI based on agent observing the POI

        Args:
            agent (Agent): Must have a get_loc and get_uuid function

        Returns:
            None:
        
        Mutates:
            _closest_distance: Set to distance between poi and agent if shorter
            _closest_agent: Set to agent if agent is now closest agent
            _removed: If agent is within minimum distance, marks POI
        """

        if not self.visible():
            return
        
        dist = self._distance(agent.get_loc())
        if dist < self._scoring_radius:
            if not self.has_been_observed() or dist < self._closest_distance:
                self._closest_distance = max(self._min_radius, dist)
                self._closest_agent = agent.get_uuid()

        if dist < self._min_radius:
            self.remove()

    def reset(self):
        """ Reset POI by forgetting previous scoring agent. """
        self._closest_distance = None
        self._closest_agent = None
        self._removed = False
        
    def get_loc(self):
        """ Accessor method for location tuple """
        return self._loc

    def score_info(self):
        """ Accessor method for closet agent and corresponding distance """
        return self._closest_agent, self._closest_distance
    
    def _distance(self, other_loc):
        """ Private method for vector distance from POI to another point.

        Args:
            other_loc (numpy array): Location of other point, must be np array.

        Returns:
            distance (double): Euclidean distance between point and self.
        """
        return np.sqrt(np.sum((self._loc - other_loc)**2))
    
