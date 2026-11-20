import random
from typing import Dict, List, Optional, Tuple

from src.config import (
    ACTION_FORWARD,
    ACTION_ROTATE_LEFT,
    ACTION_ROTATE_RIGHT,
    EAST,
    GOAL_POS,
    GRID_SIZE,
    NUM_OBSTACLES,
    NUM_ORIENTATIONS,
    ORIENTATION_DELTAS,
    REWARD_COLLISION,
    REWARD_GOAL,
    REWARD_STEP,
    SIMULTANEOUS_MOVES,
    REWARD_WALL,
    SAFE_SPAWN_DISTANCE,
    SOUTH,
    START_POS,
)

FullState = Tuple[int, int, int, Tuple[Tuple[int, int], ...]]
PerceptionState = Tuple[int, int, int]


class DynamicGridWorld:
    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        num_obstacles: int = NUM_OBSTACLES,
        seed: Optional[int] = None,
        simultaneous: bool = SIMULTANEOUS_MOVES,
    ) -> None:
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.simultaneous = simultaneous
        self.rng = random.Random(seed)
        self.agent_pos: Tuple[int, int] = START_POS
        self.agent_theta: int = EAST
        self.obstacles: List[Tuple[int, int]] = []

    def reset(
        self,
        agent_pos: Tuple[int, int] = START_POS,
        agent_theta: Optional[int] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
    ) -> PerceptionState:
        if agent_theta is None:
            agent_theta = self.rng.choice([EAST, SOUTH])
        self.agent_pos = agent_pos
        self.agent_theta = agent_theta
        if obstacles is not None:
            if len(obstacles) != self.num_obstacles:
                raise ValueError(f"Expected {self.num_obstacles} obstacles.")
            self.obstacles = list(obstacles)
        else:
            available = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if (r, c) != GOAL_POS
                and abs(r - self.agent_pos[0]) + abs(c - self.agent_pos[1])
                >= SAFE_SPAWN_DISTANCE
            ]
            self.obstacles = self.rng.sample(available, self.num_obstacles)
        return self.get_perception_state()

    def get_full_state(self) -> FullState:
        return (
            self.agent_pos[0],
            self.agent_pos[1],
            self.agent_theta,
            tuple(self.obstacles),
        )

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _adjacent_cell(self, row: int, col: int, theta: int) -> Tuple[int, int]:
        dr, dc = ORIENTATION_DELTAS[theta]
        return row + dr, col + dc

    def get_perception_state(self) -> PerceptionState:
        return (self.agent_pos[0], self.agent_pos[1], self.agent_theta)

    @staticmethod
    def encode_perception(state: PerceptionState) -> int:
        x, y, theta = state
        return (x * GRID_SIZE + y) * NUM_ORIENTATIONS + theta

    @staticmethod
    def decode_perception(idx: int) -> PerceptionState:
        theta = idx % NUM_ORIENTATIONS
        idx //= NUM_ORIENTATIONS
        y = idx % GRID_SIZE
        x = idx // GRID_SIZE
        return (x, y, theta)

    def get_free_neighbors(
        self, pos: Tuple[int, int], occupied: set
    ) -> List[Tuple[int, int]]:
        r, c = pos
        neighbors: List[Tuple[int, int]] = []
        for dr, dc in ORIENTATION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if (
                self._in_bounds(nr, nc)
                and (nr, nc) not in occupied
                and (nr, nc) != GOAL_POS
            ):
                neighbors.append((nr, nc))
        return neighbors

    def _move_obstacles(self) -> None:
        initial_occupied = set(self.obstacles)
        claimants_by_cell: Dict[Tuple[int, int], List[int]] = {}

        for idx, obs_pos in enumerate(self.obstacles):
            free_neighbors = self.get_free_neighbors(obs_pos, initial_occupied)
            if free_neighbors:
                target = self.rng.choice(free_neighbors)
            else:
                target = obs_pos
            claimants_by_cell.setdefault(target, []).append(idx)

        new_obstacles = list(self.obstacles)
        for target, claimants in claimants_by_cell.items():
            winner = (
                claimants[0]
                if len(claimants) == 1
                else self.rng.choice(claimants)
            )
            new_obstacles[winner] = target

        self.obstacles = new_obstacles

    def is_terminal(self) -> Tuple[bool, str]:
        if self.agent_pos == GOAL_POS:
            return True, "goal"
        if self.agent_pos in self.obstacles:
            return True, "collision"
        return False, "non_terminal"

    def step(
        self, action: int
    ) -> Tuple[PerceptionState, float, bool, Dict[str, object]]:
        prev_pos = self.agent_pos
        hit_wall = False

        if action == ACTION_ROTATE_LEFT:
            self.agent_theta = (self.agent_theta - 1) % NUM_ORIENTATIONS
        elif action == ACTION_ROTATE_RIGHT:
            self.agent_theta = (self.agent_theta + 1) % NUM_ORIENTATIONS
        elif action == ACTION_FORWARD:
            cand_r, cand_c = self._adjacent_cell(
                prev_pos[0], prev_pos[1], self.agent_theta
            )
            if self._in_bounds(cand_r, cand_c):
                self.agent_pos = (cand_r, cand_c)
            else:
                self.agent_pos = prev_pos
                hit_wall = True
        else:
            raise ValueError(f"Invalid action: {action}")

        if self.agent_pos == GOAL_POS:
            reward = REWARD_GOAL
            terminated = True
            event = "goal"
        elif self.simultaneous:
            prev_obstacles = list(self.obstacles)
            self._move_obstacles()
            crossed_head_on = (
                self.agent_pos != prev_pos
                and any(
                    old_o == self.agent_pos and new_o == prev_pos
                    for old_o, new_o in zip(prev_obstacles, self.obstacles)
                )
            )
            if self.agent_pos in self.obstacles or crossed_head_on:
                reward = REWARD_COLLISION
                terminated = True
                event = "collision"
            elif action == ACTION_FORWARD and hit_wall:
                reward = REWARD_WALL
                terminated = False
                event = "wall"
            else:
                reward = REWARD_STEP
                terminated = False
                event = "step"
        else:
            if self.agent_pos in self.obstacles:
                reward = REWARD_COLLISION
                terminated = True
                event = "collision"
            else:
                self._move_obstacles()
                if self.agent_pos in self.obstacles:
                    reward = REWARD_COLLISION
                    terminated = True
                    event = "collision"
                elif action == ACTION_FORWARD and hit_wall:
                    reward = REWARD_WALL
                    terminated = False
                    event = "wall"
                else:
                    reward = REWARD_STEP
                    terminated = False
                    event = "step"

        next_perception = self.get_perception_state()
        info: Dict[str, object] = {
            "event": event,
            "hit_wall": hit_wall,
            "full_state": self.get_full_state(),
        }
        return next_perception, reward, terminated, info