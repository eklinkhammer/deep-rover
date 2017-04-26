import numpy as np

from gym_rover.state.poi import POI

class DummyAgent():
    def __init__(self, loc, uuid):
        self.uuid = uuid
        self.loc = loc

    def get_loc(self):
        return self.loc
    
    def get_uuid(self):
        return self.uuid

def test_reset():
    poi = POI(np.array([0,0]), 5.1)
    agent = DummyAgent(np.array([3,4]), 1)

    poi.observe_by(agent)
    poi.reset()

    assert not poi.has_been_observed()

def test_observe_by():
    poi = POI(np.array([0,0]), 5.1)
    agent = DummyAgent(np.array([3,4]), 1)

    poi.observe_by(agent)
    uuid, dist = poi.score_info()

    assert uuid == 1
    assert dist == 5

def test_observe_by_better_before():
    poi = POI(np.array([0,0]), 10)
    agent1 = DummyAgent(np.array([3,4]), 1)
    agent2 = DummyAgent(np.array([3,5]), 2)

    poi.observe_by(agent1)
    poi.observe_by(agent2)
    uuid, dist = poi.score_info()

    assert uuid == 1
    assert dist == 5

def test_observe_by_new_agent():
    poi = POI(np.array([0,0]), 10)
    agent1 = DummyAgent(np.array([4,5]), 1)
    agent2 = DummyAgent(np.array([3,4]), 2)

    poi.observe_by(agent1)
    poi.observe_by(agent2)
    uuid, dist = poi.score_info()

    assert uuid == 2
    assert dist == 5

def test_observe_by_too_far():
    poi = POI(np.array([0,0]), 1.0)
    agent = DummyAgent(np.array([1,1]), 1)

    poi.observe_by(agent)

    assert not poi.has_been_observed()
    
def test_has_been_observed():
    poi = POI(np.array([0,0]), 10)
    agent = DummyAgent(np.array([4,5]), 1)
    
    poi.observe_by(agent)

    assert poi.has_been_observed()

def test_get_loc():
    poi = POI(np.array([0,0]), 1)

    loc = poi.get_loc()

    assert type(loc) is np.ndarray

def test_get_loc_np():
    poi = POI((2,3),1)

    loc = poi.get_loc()

    assert type(loc) is np.ndarray
    assert loc[0] == 2 and loc[1] == 3
