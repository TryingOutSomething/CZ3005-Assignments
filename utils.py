from copy import deepcopy


def generate_all_possible_states(dimension):
    return list(
        dict.fromkeys(
            [f"{i}{j}{k}"
             for i in range(dimension)
             for j in range(dimension)
             for k in range(dimension)]
        )
    )


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


def convert_string_state_to_list(state):
    return list(map(int, state))
