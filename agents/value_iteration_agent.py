from collections import defaultdict
from copy import deepcopy

import utils

ACTION_SPACE = ['left', 'right', 'forward', 'backward', 'up', 'down']
THETA = 0.001


class ValueIterationAgent:
    def __init__(self, environment_dimension):
        self.v_table = defaultdict(lambda: 0.0)
        self.q_table = defaultdict(lambda: 0.0)
        self.policy = defaultdict(str)
        self.state_list = utils.generate_all_possible_states(environment_dimension)
        self.discount_factor = 0.99
        self.set_reward_for_goal_state_in_v_table()

    def set_reward_for_goal_state_in_v_table(self):
        self.v_table['333'] = 1.0

    def value_iteration(self):
        for state in self.state_list:
            self.calculate_q_table_action_values_for_state(state)

            max_q_value = utils.get_max_q_value_for_state_action_pair(self.q_table, state)
            self.v_table[state] = max_q_value
            self.policy[state] = self.get_best_action_for_state(state, max_q_value)

    def calculate_q_table_action_values_for_state(self, current_state):
        for action in ACTION_SPACE:
            q_value = self.calculate_q_value_for_state(current_state, action)
            self.q_table[current_state, action] = q_value

    def calculate_q_value_for_state(self, current_state, current_action):
        all_possible_actions = utils.get_list_of_possible_actions(current_action, ACTION_SPACE)

        q_value = 0

        for action in all_possible_actions:
            transition_probability = utils.get_transition_probability(action, current_action)
            next_state = utils.determine_next_state(current_state, current_action)
            reward = utils.get_current_state_reward(current_state)
            q_value += transition_probability * (reward + self.discount_factor * self.v_table[next_state])

        return q_value

    def get_best_action_for_state(self, current_state, max_q_value):
        for (state, action), q_value in self.q_table.items():
            if state != current_state or q_value != max_q_value:
                continue

            return action

    def has_converged(self, old_v_table):
        for key, value in self.v_table.items():
            delta = abs(value - old_v_table.get(key))

            if delta > THETA:
                return False

        return True

    def train(self):
        converges = False

        while not converges:
            old_v_table = deepcopy(self.v_table)
            self.value_iteration()
            converges = self.has_converged(old_v_table)
