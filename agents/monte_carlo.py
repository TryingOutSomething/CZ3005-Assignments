from collections import defaultdict
from random import choice

import numpy as np

ALL_POSSIBLE_ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
dimension = 4


class MonteCarloAgent:
    def __init__(self):
        self.action_space = ALL_POSSIBLE_ACTIONS
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.returns = defaultdict(list)
        self.policy = self.generate_random_policy()

    def generate_random_policy(self):
        policy = {}

        for i in range(dimension):
            for j in range(dimension):
                for k in range(dimension):
                    policy[(i, j, k)] = choice(self.action_space)

        return policy

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
