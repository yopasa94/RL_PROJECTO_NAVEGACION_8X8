from pathlib import Path
import random
from typing import Optional

import numpy as np

from src.config import NUM_ACTIONS, NUM_PERCEPTION_STATES
from src.environment import DynamicGridWorld, PerceptionState


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q_table = np.zeros((NUM_PERCEPTION_STATES, NUM_ACTIONS), dtype=np.float64)
        self.visit_counts = np.zeros((NUM_PERCEPTION_STATES, NUM_ACTIONS), dtype=np.float64)

    def select_action(self, state: PerceptionState, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(NUM_ACTIONS)
        state_idx = DynamicGridWorld.encode_perception(state)
        q_values = self.q_table[state_idx]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self.rng.choice(best_actions.tolist()))

    def update(
        self,
        state: PerceptionState,
        action: int,
        reward: float,
        next_state: PerceptionState,
        terminated: bool,
    ) -> float:
        s_idx = DynamicGridWorld.encode_perception(state)
        ns_idx = DynamicGridWorld.encode_perception(next_state)
        self.visit_counts[s_idx, action] = float(self.visit_counts[s_idx, action]) + 1.0
        best_next_q = 0.0 if terminated else float(np.max(self.q_table[ns_idx]))
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - float(self.q_table[s_idx, action])
        visits = float(self.visit_counts[s_idx, action])
        eff_alpha = max(0.005, self.alpha / (1.0 + 0.02 * visits))
        self.q_table[s_idx, action] = float(self.q_table[s_idx, action]) + eff_alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filepath: str = "q_table.npy") -> str:
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), self.q_table)
        return str(out_path)

    def load_q_table(self, filepath: str = "q_table.npy") -> None:
        in_path = Path(filepath).resolve()
        loaded = np.load(str(in_path))
        if loaded.shape != (NUM_PERCEPTION_STATES, NUM_ACTIONS):
            raise ValueError(f"Invalid Q-table shape: {loaded.shape}")
        self.q_table = loaded
        self.epsilon = 0.0
