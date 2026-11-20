from typing import Any, Dict, List, Optional, Set

from src.agent import QLearningAgent
from src.config import (
    EAST,
    GOAL_POS,
    NORTH,
    NUM_ACTIONS,
    NUM_PERCEPTION_STATES,
    SOUTH,
    TrainingConfig,
    WEST,
)
from src.environment import DynamicGridWorld, PerceptionState

THETA_MIRROR: Dict[int, int] = {NORTH: WEST, EAST: SOUTH, SOUTH: EAST, WEST: NORTH}
ACTION_MIRROR: Dict[int, int] = {0: 1, 1: 0, 2: 2}


def _mirror_perception(state: PerceptionState) -> PerceptionState:
    x, y, theta = state
    return (y, x, THETA_MIRROR[theta])


def symmetrize_q_table(agent: QLearningAgent) -> None:
    for idx in range(NUM_PERCEPTION_STATES):
        state = DynamicGridWorld.decode_perception(idx)
        m_state = _mirror_perception(state)
        m_idx = DynamicGridWorld.encode_perception(m_state)
        if m_idx < idx:
            continue
        for a in range(NUM_ACTIONS):
            ma = ACTION_MIRROR[a]
            if m_idx == idx and ma < a:
                continue
            v1 = float(agent.visit_counts[idx, a])
            v2 = float(agent.visit_counts[m_idx, ma])
            if v1 + v2 > 0.0:
                q1 = float(agent.q_table[idx, a])
                q2 = float(agent.q_table[m_idx, ma])
                merged = (v1 * q1 + v2 * q2) / (v1 + v2)
                agent.q_table[idx, a] = merged
                agent.q_table[m_idx, ma] = merged


def train_agent(
    env: DynamicGridWorld,
    agent: QLearningAgent,
    config: TrainingConfig,
    snapshot_episodes: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    episode_rewards: List[float] = []
    episode_lengths: List[float] = []
    success_flags: List[float] = []
    collision_flags: List[float] = []
    snapshots: List[Dict[str, Any]] = []

    if snapshot_episodes is None:
        snapshot_episodes = {
            1,
            max(1, config.episodes // 20),
            max(1, config.episodes // 5),
            max(1, config.episodes // 2),
            config.episodes,
        }

    for ep_idx in range(1, config.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        final_event = "timeout"
        record_this = ep_idx in snapshot_episodes
        frames: List[Dict[str, Any]] = []

        if record_this:
            frames.append(
                {
                    "step": 0,
                    "agent_pos": list(env.agent_pos),
                    "agent_theta": env.agent_theta,
                    "obstacles": [list(o) for o in env.obstacles],
                    "perception": list(state),
                    "action": None,
                    "reward": 0.0,
                    "total_reward": 0.0,
                    "event": "start",
                }
            )

        for _ in range(config.max_steps_per_episode):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, info = env.step(action)
            dist_prev = (GOAL_POS[0] - state[0]) + (GOAL_POS[1] - state[1])
            dist_next = (GOAL_POS[0] - next_state[0]) + (GOAL_POS[1] - next_state[1])
            phi_prev = -2.0 * float(dist_prev)
            phi_next = 0.0 if terminated else -2.0 * float(dist_next)
            shaped_reward = reward + agent.gamma * phi_next - phi_prev

            agent.update(state, action, shaped_reward, next_state, terminated)

            m_state = _mirror_perception(state)
            m_next = _mirror_perception(next_state)
            m_action = ACTION_MIRROR[action]
            agent.update(m_state, m_action, shaped_reward, m_next, terminated)

            state = next_state
            total_reward += reward
            steps += 1

            if record_this:
                frames.append(
                    {
                        "step": steps,
                        "agent_pos": list(env.agent_pos),
                        "agent_theta": env.agent_theta,
                        "obstacles": [list(o) for o in env.obstacles],
                        "perception": list(state),
                        "action": action,
                        "reward": reward,
                        "total_reward": total_reward,
                        "event": str(info["event"]),
                    }
                )

            if terminated:
                final_event = str(info["event"])
                break

        if record_this:
            snapshots.append(
                {
                    "episode": ep_idx,
                    "epsilon": round(agent.epsilon, 4),
                    "total_reward": round(total_reward, 2),
                    "steps": steps,
                    "final_event": final_event,
                    "frames": frames,
                }
            )

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(float(steps))
        success_flags.append(1.0 if final_event == "goal" else 0.0)
        collision_flags.append(1.0 if final_event == "collision" else 0.0)

    symmetrize_q_table(agent)

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "successes": success_flags,
        "collisions": collision_flags,
        "snapshots": snapshots,
    }
