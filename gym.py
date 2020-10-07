from collections import defaultdict

import numpy as np

from environment import TreasureCube

ALL_POSSIBLE_ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
DISCOUNT_FACTOR = 0.99
TRANSITION = 0.6
THETA = 0.0001

env = TreasureCube(max_step=500)
V = defaultdict(lambda: 0.0)
env.reset()
