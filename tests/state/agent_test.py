import numpy as np
import math

from gym_rover.state.agent import Agent

def test_init_uuid():
    x = Agent((2,2))
    y = Agent((2,2))

    assert not x.get_uuid() == y.get_uuid()

def test_discrete_move_still():
    agent = Agent(np.array([0,0]), 1)
    agent.discrete_move(0)

    assert agent.get_loc().all() == np.array([0,0]).all()

def test_discrete_move_cardinal():
    agent = Agent(np.array([0,0]), 1)
    agent.discrete_move(1)

    assert agent.get_loc().all() == np.array([1,0]).all()

def test_discrete_move_nw():
    agent = Agent(np.array([0,0]), 1)
    agent.discrete_move(4)

    r2 = math.sqrt(2)/2
    assert agent.get_loc().all() == np.array([r2, r2]).all()

def test_cont_move():
    agent = Agent(np.array([1,1]), 1)

    agent.cont_move(np.array([1.5, 2.3]))

    assert agent.get_loc().all() == np.array([2.5, 3.3]).all()

def test_get_loc():
    agent = Agent(np.array([0,0]), 1)

    loc = agent.get_loc()

    assert type(loc) is np.ndarray

def test_get_loc_np():
    agent = Agent((2,3),1)

    loc = agent.get_loc()

    assert type(loc) is np.ndarray
    assert loc[0] == 2 and loc[1] == 3
