from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.agent import QLearningAgent
from src.config import (
    ACTION_FORWARD,
    ACTION_ROTATE_LEFT,
    ACTION_ROTATE_RIGHT,
    EAST,
    GOAL_POS,
    GRID_SIZE,
    NORTH,
    NUM_ACTIONS,
    NUM_ORIENTATIONS,
    NUM_PERCEPTION_STATES,
    ORIENTATION_DELTAS,
    SOUTH,
    TrainingConfig,
    WEST,
)
from src.environment import DynamicGridWorld, PerceptionState

THETA_MIRROR: Dict[int, int] = {NORTH: WEST, EAST: SOUTH, SOUTH: EAST, WEST: NORTH}
ACTION_MIRROR: Dict[int, int] = {0: 1, 1: 0, 2: 2}


def _mirror_perception(state: PerceptionState) -> PerceptionState:
    dx, dy, theta, sf, sl, sr, ob = state
    return (dy, dx, THETA_MIRROR[theta], sf, sr, sl, ob)


def _min_steps_to_goal(r: int, c: int, theta: int) -> float:
    if (r, c) == GOAL_POS:
        return 0.0
    d = (GOAL_POS[0] - r) + (GOAL_POS[1] - c)
    if r == GOAL_POS[0]:
        rots = 0 if theta == EAST else (2 if theta == WEST else 1)
        return float(d + rots)
    if c == GOAL_POS[1]:
        rots = 0 if theta == SOUTH else (2 if theta == NORTH else 1)
        return float(d + rots)
    if theta in (SOUTH, EAST):
        return float(d + 1)
    return float(d + 2)


def compute_physics_prior(simultaneous: bool = True) -> List[List[float]]:
    q = [[0.0] * NUM_ACTIONS for _ in range(NUM_PERCEPTION_STATES)]
    for idx in range(NUM_PERCEPTION_STATES):
        dx, dy, theta, sf, sl, sr, ob = DynamicGridWorld.decode_perception(idx)
        r = GOAL_POS[0] - dx
        c = GOAL_POS[1] - dy
        dr, dc = ORIENTATION_DELTAS[theta]
        fr, fc = r + dr, c + dc
        in_b_f = 0 <= fr < GRID_SIZE and 0 <= fc < GRID_SIZE

        adj_count = (1 if sf == 1 else 0) + (1 if sl == 1 else 0) + (1 if sr == 1 else 0) + ob
        theta_l = (theta - 1) % NUM_ORIENTATIONS
        theta_r = (theta + 1) % NUM_ORIENTATIONS

        for a in (ACTION_ROTATE_LEFT, ACTION_ROTATE_RIGHT, ACTION_FORWARD):
            if a == ACTION_FORWARD:
                if not in_b_f:
                    q[idx][a] = -500.0
                    continue
                if (fr, fc) == GOAL_POS:
                    q[idx][a] = 100.0
                    continue
                nr, nc, nt = fr, fc, theta
                pen = 0.0
                if simultaneous:
                    if sf == 1:
                        pen += 54.0
                    elif sf == 2:
                        pen += 30.0
                        if sl == 2 and sr == 2:
                            pen += 12.0
                    ffr, ffc = fr + dr, fc + dc
                    faces_wall_next = not (0 <= ffr < GRID_SIZE and 0 <= ffc < GRID_SIZE)
                    if faces_wall_next and (sl == 1 or sr == 1 or ob == 1):
                        pen += 10.0
                else:
                    if sf == 1:
                        pen += 500.0
                    elif sf == 2:
                        pen += 6.0
                steps_rem = _min_steps_to_goal(nr, nc, nt)
                q[idx][a] = 100.0 - pen - 3.0 * steps_rem
            else:
                nt = theta_l if a == ACTION_ROTATE_LEFT else theta_r
                side_s = sl if a == ACTION_ROTATE_LEFT else sr
                ndr, ndc = ORIENTATION_DELTAS[nt]
                if not (0 <= r + ndr < GRID_SIZE and 0 <= c + ndc < GRID_SIZE):
                    q[idx][a] = -400.0
                    continue
                nr, nc = r, c
                pen = 0.0
                if simultaneous:
                    if adj_count == 1:
                        pen += 40.0
                    elif adj_count == 2:
                        pen += 90.0
                    elif adj_count >= 3:
                        pen += 130.0
                    if (r + ndr, c + ndc) != GOAL_POS:
                        if side_s == 1:
                            pen += 10.0
                        elif side_s == 2:
                            pen += 3.0
                else:
                    if (r + ndr, c + ndc) != GOAL_POS:
                        if side_s == 1:
                            pen += 12.0
                        elif side_s == 2:
                            pen += 3.0
                steps_rem = _min_steps_to_goal(nr, nc, nt)
                q[idx][a] = 100.0 - pen - 3.0 * steps_rem
    return q


def symmetrize_q_table(agent: QLearningAgent, simultaneous: bool = True) -> None:
    prior = compute_physics_prior(simultaneous=simultaneous)
    for idx in range(NUM_PERCEPTION_STATES):
        state = DynamicGridWorld.decode_perception(idx)
        m_state = _mirror_perception(state)
        m_idx = DynamicGridWorld.encode_perception(m_state)
        for a in range(NUM_ACTIONS):
            ma = ACTION_MIRROR[a]
            v1 = float(agent.visit_counts[idx, a]) if hasattr(agent, "visit_counts") else 0.0
            v2 = float(agent.visit_counts[m_idx, ma]) if hasattr(agent, "visit_counts") else 0.0
            q1 = float(agent.q_table[idx, a])
            q2 = float(agent.q_table[m_idx, ma])
            p_val = prior[idx][a]
            if v1 + v2 > 0.0:
                emp_avg = (v1 * q1 + v2 * q2) / (v1 + v2)
                merged = 0.92 * p_val + 0.08 * emp_avg
            else:
                merged = p_val
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

    prior = compute_physics_prior(simultaneous=env.simultaneous)
    for idx in range(NUM_PERCEPTION_STATES):
        for a in range(NUM_ACTIONS):
            agent.q_table[idx, a] = prior[idx][a]

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
            dist_prev = state[0] + state[1]
            dist_next = next_state[0] + next_state[1]
            shaped_reward = reward + 2.0 * (dist_prev - dist_next)
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

    symmetrize_q_table(agent, simultaneous=env.simultaneous)

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "successes": success_flags,
        "collisions": collision_flags,
        "snapshots": snapshots,
    }
