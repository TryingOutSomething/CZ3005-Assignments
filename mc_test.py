import argparse

# from agents.random_agent import RandomAgent
from agents.monte_carlo import MonteCarloAgent
from environment import TreasureCube


def test_cube(max_episode, max_step):
    env = TreasureCube(max_step=max_step)
    agent = MonteCarloAgent()

    episode_reward_progress = []

    for episode_num in range(max_episode):
        episode, episode_reward, no_of_steps = agent.generate_episode(env)

        print(f'episode: {episode_num}, total_steps: {no_of_steps} episode reward: {episode_reward}')
        episode_reward_progress.append(episode_reward)

        state_actions_in_episode = list(set([(sar[0], sar[1]) for sar in episode]))
        agent.train(state_actions_in_episode, episode)

    # print(agent.q_table)
    print(max(episode_reward_progress))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--max_episode', type=int, default=500)
    parser.add_argument('--max_step', type=int, default=500)
    args = parser.parse_args()

    test_cube(args.max_episode, args.max_step)
