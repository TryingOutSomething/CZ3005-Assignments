from collections import defaultdict

import numpy as np

from environment import TreasureCube

ALL_POSSIBLE_ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
DISCOUNT_FACTOR = 0.99
TRANSITION = 0.6
THETA = 0.0001


def get_all_states_in_environment(dimension):
    return list(
        dict.fromkeys(
            [f"{i}{j}{k}"
             for i in range(dimension)
             for j in range(dimension)
             for k in range(dimension) if i != 0 or j != 0 or k != 0]
        )
    )


def get_transition_probability(old_state, new_state, action):
    if action == 'left' or action == 'right':
        if old_state[0] == new_state[0] and old_state[2] == new_state[2]:
            return 0.6
        else:
            return 0.1
    if action == 'up' or action == 'down':
        if old_state[0] == new_state[0] and old_state[1] == new_state[1]:
            return 0.6
        else:
            return 0.1

    # up or down
    if old_state[1] == new_state[1] and old_state[2] == new_state[2]:
        return 0.6
    else:
        return 0.1


def one_step_lookahead(v):
    list_of_action_values = defaultdict(lambda: 0.0)

    current_env_state = env.curr_pos

    for action in ALL_POSSIBLE_ACTIONS:
        reward, _, next_state = env.step(action)
        probability = get_transition_probability(''.join(map(str, current_env_state)), next_state, action)
        list_of_action_values[action] += probability * (reward + DISCOUNT_FACTOR * v[(next_state, action)])

        env.curr_pos = current_env_state

    return list_of_action_values


env = TreasureCube(max_step=500)
V = defaultdict(lambda: 0.0)
all_possible_states = get_all_states_in_environment(env.dim)
env.reset()

while True:
    delta = 0

    for state in all_possible_states:
        action_values = one_step_lookahead(V)
        best_action_value = np.max(list(action_values.values()))

        old_V_value = V[state]
        delta = max(delta, np.abs(best_action_value - old_V_value))

        V[state] = best_action_value

    if delta < THETA:
        break

# policy = defaultdict(lambda: 0.0)
#
# for state in all_possible_states:
#     action_values = one_step_lookahead(state, V)
#     best_action = np.argmax(list(action_values.values()))
#
#     policy[(state, best_action)] = 1.0
#
# pprint(V)
# pprint(policy)
