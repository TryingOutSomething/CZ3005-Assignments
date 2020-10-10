from collections import defaultdict

import numpy as np

epsilon = 0.1


class MonteCarloAgent:
    def __init__(self):
        self.action_space = ['left', 'right', 'forward', 'backward', 'up', 'down']
        self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.returns_sum = defaultdict(float)
        self.states_count = defaultdict(float)

    def get_epsilon_greedy_action_policy(self, q_table, observation):
        length_of_action_space = len(self.action_space)

        q_table_actions = np.ones(length_of_action_space, dtype=float) * epsilon / length_of_action_space
        best_action = np.argmax(q_table[observation])
        q_table_actions[best_action] += (1.0 - epsilon)

        return q_table_actions

    def take_action(self, current_state):
        prob_scores = self.get_epsilon_greedy_action_policy(self.q_table, current_state)
        return np.random.choice(self.action_space, p=prob_scores)

    def get_action_index_from_action_space(self, action):
        return self.action_space.index(action)

    def train(self, episode):
        state_actions_in_episode = list(set([(sar[0], sar[1]) for sar in episode]))

        for i, sa_pair in enumerate(state_actions_in_episode):
            state, action = sa_pair
            action_index = self.get_action_index_from_action_space(action)

            G = sum([sar[2] for i, sar in enumerate(episode[i:])])

            self.returns_sum[sa_pair] += G
            self.states_count[sa_pair] += 1.0
            self.q_table[state][action_index] = self.returns_sum[sa_pair] / self.states_count[sa_pair]
