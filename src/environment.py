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
    SENSOR_FREE,
    SENSOR_OCCUPIED,
    SENSOR_THREAT,
    SOUTH,
    START_POS,
)

FullState = Tuple[int, int, int, Tuple[Tuple[int, int], ...]]
PerceptionState = Tuple[int, int, int, int, int, int, int]


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

    def _sensor_code(self, cell: Tuple[int, int], obstacle_set: set) -> int:
        if not self._in_bounds(*cell):
            return SENSOR_FREE
        if cell in obstacle_set:
            return SENSOR_OCCUPIED
        if cell == GOAL_POS:
            return SENSOR_FREE
        r, c = cell
        for dr, dc in ORIENTATION_DELTAS.values():
            if (r + dr, c + dc) in obstacle_set:
                return SENSOR_THREAT
        return SENSOR_FREE

    def get_perception_state(self) -> PerceptionState:
        xa, ya = self.agent_pos
        dx_goal = GOAL_POS[0] - xa
        dy_goal = GOAL_POS[1] - ya
        obstacle_set = set(self.obstacles)

        front_cell = self._adjacent_cell(xa, ya, self.agent_theta)
        left_theta = (self.agent_theta - 1) % NUM_ORIENTATIONS
        right_theta = (self.agent_theta + 1) % NUM_ORIENTATIONS
        back_theta = (self.agent_theta + 2) % NUM_ORIENTATIONS
        left_cell = self._adjacent_cell(xa, ya, left_theta)
        right_cell = self._adjacent_cell(xa, ya, right_theta)
        back_cell = self._adjacent_cell(xa, ya, back_theta)

        sens_front = self._sensor_code(front_cell, obstacle_set)
        sens_left = self._sensor_code(left_cell, obstacle_set)
        sens_right = self._sensor_code(right_cell, obstacle_set)
        obs_back = 1 if back_cell in obstacle_set else 0

        return (
            dx_goal,
            dy_goal,
            self.agent_theta,
            sens_front,
            sens_left,
            sens_right,
            obs_back,
        )

    @staticmethod
    def encode_perception(state: PerceptionState) -> int:
        dx, dy, theta, sens_f, sens_l, sens_r, obs_b = state
        idx = dx
        idx = idx * 8 + dy
        idx = idx * 4 + theta
        idx = idx * 3 + sens_f
        idx = idx * 3 + sens_l
        idx = idx * 3 + sens_r
        idx = idx * 2 + obs_b
        return idx

    @staticmethod
    def decode_perception(idx: int) -> PerceptionState:
        obs_b = idx % 2
        idx //= 2
        sens_r = idx % 3
        idx //= 3
        sens_l = idx % 3
        idx //= 3
        sens_f = idx % 3
        idx //= 3
        theta = idx % 4
        idx //= 4
        dy = idx % 8
        dx = idx // 8
        return (dx, dy, theta, sens_f, sens_l, sens_r, obs_b)

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
                and (self.simultaneous or (nr, nc) != self.agent_pos)
            ):
                neighbors.append((nr, nc))
        return neighbors

    def _move_obstacles(self) -> None:
        occupied = set(self.obstacles)
        new_obstacles: List[Tuple[int, int]] = []
        for obs_pos in self.obstacles:
            occupied.remove(obs_pos)
            free_neighbors = self.get_free_neighbors(obs_pos, occupied)
            if free_neighbors:
                next_pos = self.rng.choice(free_neighbors)
            else:
                next_pos = obs_pos
            occupied.add(next_pos)
            new_obstacles.append(next_pos)
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
        else:
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
            if not terminated:
                self._move_obstacles()

        next_perception = self.get_perception_state()
        info: Dict[str, object] = {
            "event": event,
            "hit_wall": hit_wall,
            "full_state": self.get_full_state(),
        }
        return next_perception, reward, terminated, info