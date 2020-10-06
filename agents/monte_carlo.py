from collections import defaultdict
from random import choice

import numpy as np

ALL_POSSIBLE_ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
dimension = 4


def gen_policy():
    policy = {}

    for i in range(dimension):
        for j in range(dimension):
            policy[(i, j)] = choice(ALL_POSSIBLE_ACTIONS)

    return policy


class MonteCarloAgent:
    def __init__(self):
        self.action_space = ALL_POSSIBLE_ACTIONS
        # TODO: Change Q to a dict with all states as key and dictionary with value as each action as key and 0 as value
        # Example
        # {(0, 1): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (1, 2): {'L': 0, 'D': 0, 'R': 0, 'U': 0},
        #  (0, 0): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (2, 3): {'L': 0, 'D': 0, 'R': 0, 'U': 0},
        #  (2, 0): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (1, 0): {'L': 0, 'D': 0, 'R': 0, 'U': 0},
        #  (2, 2): {'L': 0, 'D': 0, 'R': 0, 'U': 0}, (0, 2): {'L': 0, 'D': 0, 'R': 0, 'U': 0},
        #  (2, 1): {'L': 0, 'D': 0, 'R': 0, 'U': 0}}
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.returns = defaultdict(list)

    def take_action_using_policy(self):
        # Returns a random action using epsilon-soft policy
        pass

    def take_action(self, state):
        action = choice(self.action_space)
        return action

    def train(self, state, action, next_state, reward):
        # TODO: Implement policy evaluation
        # TODO: Implement action base on state? Base on policy improvement
        pass
