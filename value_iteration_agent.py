from collections import defaultdict

import numpy as np

import utils

ACTION_SPACE = ['left', 'right', 'forward', 'backward', 'up', 'down']
THETA = 0.001


class ValueIterationAgent:
    def __init__(self, environment_dimension):
        """
        Creates the Agent with the environment's dimension. It also instantiates the required variables to compute the
        optimal V values for each state in the environment

        :parameter environment_dimension: Generates all possible states that can be made by the agent.
        :type environment_dimension: int
        """
        self.v_table = defaultdict(lambda: 0.0)
        self.policy = defaultdict(str)
        self.state_list = utils.generate_all_possible_states(environment_dimension)
        self.discount_factor = 0.95
        self.set_reward_for_goal_state_in_v_table()

    def set_reward_for_goal_state_in_v_table(self):
        """ Sets the reward of the goal state in the environment """
        self.v_table['333'] = 1.0

    def value_iteration(self):
        """
        Finds the best action to take to reach the goal state for each state in the environment.
        This process is repeated continuously till the values of the V function converges.
        """
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
        """
        Calculates every Q function value from the current state to the neighbouring states base on all possible actions
        available

        :param current_state: str
        :return: An array of Q function values from the current state to the neighbouring states
        """
        action_index = 0
        q_table_of_actions = np.zeros(len(ACTION_SPACE))

        for action in ACTION_SPACE:
            all_possible_actions = utils.get_list_of_possible_actions(action, ACTION_SPACE)

            q_table_of_actions[action_index] = self.calculate_q_value_for_action(all_possible_actions,
                                                                                 action,
                                                                                 current_state)
            action_index += 1

        return q_table_of_actions

    def calculate_q_value_for_action(self, all_possible_actions, current_state_action, current_state):
        """
        Calculates the Q function value for the current intended action in the current state

        :param all_possible_actions: A list of all possible actions that the agent can take in the current state
        :type all_possible_actions: list

        :param current_state_action: The indented action of the Agent in the current state
        :type current_state_action: str

        :param current_state: The current state of the Agent
        :type current_state: str

        :return: The Q function cost of the current intended action
        """
        action_value = 0

        for action in all_possible_actions:
            transition_probability = utils.get_transition_probability(action, current_state_action)
            reward = utils.get_current_state_reward(current_state)
            next_state = utils.determine_next_state(current_state, current_state_action)
            action_value += transition_probability * (reward + self.discount_factor * self.v_table[next_state])

        return action_value

    def generate_optimal_policy(self):
        """
        Generates the optimal action to take for each state base on the V function values

        :return: A list of optimal actions for each state in the environment
        """
        for state in self.state_list:
            q_table_row = self.generate_q_table_row_for_state(state)
            best_action_index = np.argmax(q_table_row)
            self.policy[state] = ACTION_SPACE[best_action_index]

    def train(self):
        """
        Explores the environment to find the optimal path from the start state to the goal state.
        The Agent records the best actions along the way as the optimal policy.
        """
        self.value_iteration()
        self.generate_optimal_policy()

    def take_action(self, state):
        """
        Returns the optimal action in the current state from the optimal policy table

        :param state: The current state of the Agent in the environment
        :type state: str

        :return: The optimal action from the policy table
        """
        return self.policy[state]
