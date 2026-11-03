import argparse

import gymnasium as gym
import numpy as np
import rl_sensors.envs


def main():
    parser = argparse.ArgumentParser(description="Run an environment with random actions.")
    parser.add_argument(
        "--env",
        choices=("BeamOptimization-v0", "MobileSurveillance-v0"),
        default="BeamOptimization-v0",
        help="Registered Gymnasium environment ID (default: BeamOptimization-v0)",
    )
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed)

    env = gym.make(args.env)
    env.reset(seed=seed)

    n_episode_steps = 1000
    n_episodes = 100

    for _ in range(n_episodes):
        total_episode_reward = 0

        for _ in range(n_episode_steps):
            env.render()
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_episode_reward += reward

            if terminated or truncated:
                break

        print("Episode reward:", total_episode_reward)
        env.reset()

    env.close()


if __name__ == "__main__":
    main()
