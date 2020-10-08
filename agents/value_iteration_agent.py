from collections import defaultdict

import numpy as np

from utils import get_list_of_possible_actions, get_transition_probability, generate_all_possible_states, \
    convert_string_state_to_list

ACTION_SPACE = ['left', 'right', 'forward', 'backward', 'up', 'down']
THETA = 0.1


class ValueIterationAgent:
    def __init__(self, environment):
        self.env = environment
        self.v_table = defaultdict(lambda: 0.0)
        self.policy = defaultdict(str)
        self.state_list = generate_all_possible_states(environment.dim)
        self.discount_factor = 0.99

    def train(self):
        self.generate_v_table()
        self.generate_optimal_policy()

    def generate_v_table(self):
        self.env.reset()

        while True:
            delta = 0

            for state in self.state_list:
                q_table_row = self.generate_q_table_row_for_state(state)

                best_action_value_from_q_table = np.max(q_table_row)
                difference = np.abs(best_action_value_from_q_table - self.v_table[state])
                delta = max(delta, difference)

                self.v_table[state] = best_action_value_from_q_table

            if delta < THETA:
                break

    def generate_q_table_row_for_state(self, current_state):
        action_index = 0
        q_table_of_actions = np.zeros(len(ACTION_SPACE))

        for action in ACTION_SPACE:
            all_possible_actions = get_list_of_possible_actions(action, ACTION_SPACE)

            q_table_of_actions[action_index] = self.calculate_q_value_for_action(all_possible_actions,
                                                                                 action,
                                                                                 current_state)
            action_index += 1

        return q_table_of_actions

    def calculate_q_value_for_action(self, all_possible_actions, current_state_action, current_state):
        action_value = 0

        for action in all_possible_actions:
            # self.env.curr_pos = deepcopy(current_state)
            self.env.curr_pos = convert_string_state_to_list(current_state)

            reward, _, next_state = self.env.step(action)

            transition_probability = get_transition_probability(action, current_state_action)
            action_value += transition_probability * (reward + self.discount_factor * self.v_table[next_state])

        return action_value

    def generate_optimal_policy(self):
        self.env.reset()

        for state in self.state_list:
            q_table_row = self.generate_q_table_row_for_state(state)
            best_action_index = np.argmax(q_table_row)
            self.policy[state] = ACTION_SPACE[best_action_index]
