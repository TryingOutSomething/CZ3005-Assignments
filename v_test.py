import argparse

from environment import TreasureCube
from value_iteration_agent import ValueIterationAgent


def test_cube(max_episode, max_step):
    env = TreasureCube(max_step=max_step)
    agent = ValueIterationAgent(env.dim)

    agent.train()
    # pprint(agent.v_table)
    for episode_num in range(0, max_episode):
        state = env.reset()

        terminate = False
        no_of_steps = 0
        episode_reward = 0

        while not terminate:
            action = agent.take_action(state)
            reward, terminate, next_state = env.step(action)
            episode_reward += reward
            # you can comment the following two lines, if the output is too much
            # env.render()  # comment
            # print(f'step: {no_of_steps}, action: {action}, reward: {reward}')  # comment
            no_of_steps += 1
            state = next_state
        print(f'episode: {episode_num}, total_steps: {no_of_steps} episode reward: {episode_reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--max_episode', type=int, default=500)
    parser.add_argument('--max_step', type=int, default=500)
    args = parser.parse_args()

    test_cube(args.max_episode, args.max_step)
