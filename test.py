import argparse

# from agents.random_agent import RandomAgent
from agents.monte_carlo import MonteCarloAgent
from environment import TreasureCube


def test_cube(max_episode, max_step):
    env = TreasureCube(max_step=max_step)
    agent = MonteCarloAgent()

    for episode_num in range(0, max_episode):
        """Generate an episode using PI"""

        state = env.reset()
        terminate = False
        no_of_steps = 0
        episode_reward = 0

        # Start exploration
        while not terminate:
            action = agent.take_action(state)
            reward, terminate, next_state = env.step(action)
            episode_reward += reward
            # you can comment the following two lines, if the output is too much
            # env.render()  # comment
            # print(f'step: {t}, action: {action}, reward: {reward}')  # comment
            no_of_steps += 1
            agent.train(state, action, next_state, reward)
            state = next_state
        print(f'episode: {episode_num}, total_steps: {no_of_steps} episode reward: {episode_reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--max_episode', type=int, default=500)
    parser.add_argument('--max_step', type=int, default=500)
    args = parser.parse_args()

    test_cube(args.max_episode, args.max_step)
