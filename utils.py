from copy import deepcopy


def generate_all_possible_states(dimension):
    """
    Generates a list of all possible states in the environment

    :param dimension:
    :type dimension: int

    :return: A list of all possible states in the environment
    """
    return list(
        dict.fromkeys(
            [f"{i}{j}{k}"
             for i in range(dimension)
             for j in range(dimension)
             for k in range(dimension)]
        )
    )


def get_list_of_possible_actions(action, action_space):
    """
    Generates a list of possible actions the agent can take currently.

    :param action: The intended action of the agent
    :type action: str

    :param action_space: The base action space defined in the lab document
    :type action_space: list

    :return: A list of all possible actions that the agent can take from its intended action
    """
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
    """
    Determines the probability that the Agent carries out the intended action in the current state

    :param action: The intended action of the Agent in the current state
    :type action: str

    :param current_state_action:
    :type current_state_action: str

    :return: The probability of the agent carrying out the intended action in the current state
    """
    return 0.6 if current_state_action == action else 0.1


def determine_next_state(state, action):
    """
    Determines the next state of the Agent when the action is carried out

    :param state: The current state of the Agent
    :type state: str

    :param action: The action that the Agent is taking
    :type action: str

    :return: The next state that the Agent will transition to with the action is taken
    """
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
    """
    Determines the current reward of the current state

    :param state: The current state of the Agent
    :type state: str

    :return: The reward of the current state
    """
    return 1 if state == "333" else -0.1
