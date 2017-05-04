import numpy as np
import math
import sys

import pytest

from gym_rover.state.agent import Agent
from gym_rover.state.poi import POI
from gym_rover.state.world import World

def test_init_creates_valid_agents():
    world = World(10,10, 2, 0)
    valid = True
    for a in world.get_agents():
        loc = a.get_loc()
        x, y = loc[0], loc[1]
        valid = valid and x >= 0 and x < 10 and y >= 0 and y < 10

    assert valid
    assert len(world.get_agents()) == 2

def test_init_creates_valid_pois():
    world = World(10,10, 0, 2)
    valid = True
    for p in world.get_pois():
        loc = p.get_loc()
        x, y = loc[0], loc[1]
        valid = valid and x >= 0 and x < 10 and y >= 0 and y < 10

    assert valid
    assert len(world.get_pois()) == 2

def test_apply_discrete_actions():
    agents = [Agent((0,0))]
    world = World(10,10,1,0,agents)
    commands = np.array([1])

    world.apply_discrete_actions(commands)

    assert np.array_equal(np.array([1,0]), world.get_agents()[0].get_loc())

def test_apply_discrete_actions_mismatch():
    world = World(10,10,1,0)
    commands = np.array([1,5])
    
    with pytest.raises(AssertionError):
        world.apply_discrete_actions(commands)


def test_apply_cont_actions():
    agents = [Agent((0,0))]
    world = World(10,10,1,0,agents)
    commands = np.array([np.array([1,2])])

    world.apply_cont_actions(commands)

    assert np.array_equal(np.array([1,2]), world.get_agents()[0].get_loc())

def test_apply_cont_actions_mismatch():
    world = World(10,10,1,0)
    commands = np.array([np.array([1,2]), np.array([4,3])])

    with pytest.raises(AssertionError):
        world.apply_cont_actions(commands)

def test_get_obs_image():
    agents = [Agent((1.1,1))]
    pois = [POI((2.5,1.2))] # Even numbers round up, odd round down
    world = World(10,10,1,1,agents,pois)

    img = world.get_obs_image()

    assert img[1][1][0] == 1
    assert img[1][2][1] == 1

def test_get_obs_image_scale():
    agents = [Agent((1.1,1))]
    pois = [POI((2.5,1.2))]
    world = World(10,10,1,1,agents,pois)

    img = world.get_obs_image(10)

    assert img[10][11][0] == 1
    assert img[12][25][1] == 1
    
def test_get_local_obs_images():
    agents = [Agent((1.1,1))]
    pois = [POI((2.5,1.2))]
    world = World(10,10,1,1,agents,pois)

    imgs = world.get_local_obs_images(3)
    agent_img = imgs[0]

    assert agent_img[1][1][0] == 1 # Observe self
    assert agent_img[1][2][1] == 1 # Observe POI
    


def test_get_local_obs_images_border_agent():
    agents = [Agent((1.1,1))]
    pois = [POI((2.5,1.2))]
    world = World(10,10,1,1,agents,pois)

    imgs = world.get_local_obs_images(11)
    agent_img = imgs[0]

    assert agent_img[5][5][0] == 1 # Observe self
    assert agent_img[5][6][1] == 1 # Observe POI


def test_get_obs_states():
    agents = [Agent((1,  1)),
              Agent((0,  1.5)),
              Agent((1.5,0)),
              Agent((0,  0)),
              Agent((1.5,1.5))]
    pois = [POI((0,0))]

    world = World(5, 5, 5, 1, agents, pois)

    vec = world.get_obs_states()

    assert vec.shape[0] == 5
    assert vec[0][0] == 1
    assert vec[0][6] == 1 / np.sqrt(2)
    

