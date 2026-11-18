
from dataclasses import dataclass
from typing import Dict, Tuple

GRID_SIZE: int = 8
NUM_OBSTACLES: int = 5
START_POS: Tuple[int, int] = (0, 0)
GOAL_POS: Tuple[int, int] = (7, 7)

NORTH: int = 0
EAST: int = 1
SOUTH: int = 2
WEST: int = 3
NUM_ORIENTATIONS: int = 4

ORIENTATION_DELTAS: Dict[int, Tuple[int, int]] = {
    NORTH: (-1, 0),
    EAST: (0, 1),
    SOUTH: (1, 0),
    WEST: (0, -1),
}

ACTION_ROTATE_LEFT: int = 0
ACTION_ROTATE_RIGHT: int = 1
ACTION_FORWARD: int = 2
NUM_ACTIONS: int = 3

SENSOR_FREE: int = 0
SENSOR_OCCUPIED: int = 1
SENSOR_THREAT: int = 2
NUM_SENSOR_LEVELS: int = 3
NUM_PERCEPTION_STATES: int = 8 * 8 * 4 * (NUM_SENSOR_LEVELS ** 3) * 2
Q_TABLE_ENTRIES: int = NUM_PERCEPTION_STATES * NUM_ACTIONS

REWARD_GOAL: float = 100.0
REWARD_COLLISION: float = -100.0
REWARD_WALL: float = -5.0
REWARD_STEP: float = -1.0
SIMULTANEOUS_MOVES: bool = True
SAFE_SPAWN_DISTANCE: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 50000
    max_steps_per_episode: int = 200
    alpha: float = 0.08
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    seed: int = 42