from typing import Dict

import numpy as np

from src.agent import QLearningAgent
from src.environment import DynamicGridWorld


def evaluate_agent(
    env: DynamicGridWorld,
    agent: QLearningAgent,
    num_episodes: int = 500,
    max_steps: int = 200,
) -> Dict[str, float]:
    rewards = []
    lengths = []
    goals = 0
    collisions = 0
    wall_hits = 0

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, info = env.step(action)
            total_reward += reward
            steps += 1
            if info["event"] == "wall":
                wall_hits += 1
            state = next_state
            if terminated:
                if info["event"] == "goal":
                    goals += 1
                elif info["event"] == "collision":
                    collisions += 1
                break

        rewards.append(total_reward)
        lengths.append(steps)

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)

    return {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_length": float(np.mean(lengths_arr)),
        "goal_rate": goals / num_episodes,
        "collision_rate": collisions / num_episodes,
        "mean_wall_hits": wall_hits / num_episodes,
    }