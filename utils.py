from copy import deepcopy


def get_list_of_possible_actions(action, action_space):
    copy_of_action_space = deepcopy(action_space)

    if action == 'left':
        copy_of_action_space.remove('right')
        return copy_of_action_space
    if action == 'right':
        copy_of_action_space.remove('left')
        return copy_of_action_space
    if action == 'up':
        copy_of_action_space.remove('down')
        return copy_of_action_space
    if action == 'down':
        copy_of_action_space.remove('up')
        return copy_of_action_space
    if action == 'forward':
        copy_of_action_space.remove('backward')
        return copy_of_action_space

    # if action is backward
    copy_of_action_space.remove('forward')
    return copy_of_action_space


def get_transition_probability(action, current_state_action):
    return 0.6 if current_state_action == action else 0.1


def determine_next_state(state, action):
    if (state[0] == "3" and action == "forward") or (state[0] == "0" and action == "backward"):
        return state

    if (state[1] == "3" and action == "right") or (state[1] == "0" and action == "left"):
        return state

    if (state[2] == "3" and action == "up") or (state[2] == "0" and action == "down"):
        return state

    if action == "forward":
        new_z_state = int(state[0]) + 1
        return f"{new_z_state}{state[1]}{state[2]}"
    if action == "backward":
        new_z_state = int(state[0]) - 1
        return f"{new_z_state}{state[1]}{state[2]}"
    if action == "right":
        new_x_state = int(state[1]) + 1
        return f"{state[0]}{new_x_state}{state[2]}"
    if action == "left":
        new_x_state = int(state[1]) - 1
        return f"{state[0]}{new_x_state}{state[2]}"
    if action == "up":
        new_y_state = int(state[2]) + 1
        return f"{state[0]}{state[1]}{new_y_state}"
    if action == "down":
        new_y_state = int(state[2]) - 1
        return f"{state[0]}{state[1]}{new_y_state}"


def get_current_state_reward(state):
    return 1 if state == "333" else -0.1


def get_max_q_value_for_state_action_pair(q_table, state):
    return max([value for key, value in q_table.items() if key[0] == state])


# deprecated
def convert_string_state_to_list(state):
    return list(map(int, state))


# deprecated
def generate_all_possible_states(dimension):
    return list(
        dict.fromkeys(
            [f"{i}{j}{k}"
             for i in range(dimension)
             for j in range(dimension)
             for k in range(dimension)]
        )
    )
