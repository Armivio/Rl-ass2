"""
continuous_env.py — Continuous-state wrapper for the provided grid Environment.

This is the "substrate" for Assignment 2. It does NOT modify the provided
world/environment.py. Instead it wraps it and adds the layer Deep RL needs:

    state                      = env.reset(curriculum_level)
    state, reward, done, info  = env.step(action)

WHY A WRAPPER (and not editing environment.py):
  * The provided environment already does movement, stochasticity (sigma),
    rewards, and termination correctly. 
  * All we add is (a) a continuous STATE made from sensor rays, and
    (b) a curriculum start position. Both belong in a separate layer.

STATE (continuous, normalized to [0, 1]):
  * 8 "obstacle" rays: distance to the nearest wall/obstacle in 8 directions.

The state deliberately contains NO (col,row), NO distance-to-target, and NO
"this cell is the target" flag — it satisfies the assignment constraint.

ACTIONS (unchanged from the grid environment):
  0 = down, 1 = up, 2 = left, 3 = right
"""

import random
from pathlib import Path

import numpy as np

from world import Environment


# ── Shared constants — import these in the agent files ──────────────────────────
ACTION_DIM = 4   # 0 down, 1 up, 2 left, 3 right

# 8 ray directions as (dcol, drow). Grid is indexed grid[col, row].
_DIRECTIONS = [
    (0, -1),   # up
    (0,  1),   # down
    (-1, 0),   # left
    (1,  0),   # right
    (-1, -1),  # up-left
    (1,  -1),  # up-right
    (-1,  1),  # down-left
    (1,   1),  # down-right
]

# Cell codes in the grid array.
_WALL, _OBSTACLE, _TARGET = 1, 2, 3
_TRAVERSABLE = (0, _TARGET)   # rays pass through empty cells and the target


class ContinuousGridEnv:
    """Continuous-state wrapper around the provided grid `Environment`."""

    def __init__(self,
                 grid_fp,
                 sigma: float = 0.0,
                 random_seed: int = 0,
                 max_steps: int = 200):
        self.grid_fp = Path(grid_fp)
        self.sigma = sigma
        self.random_seed = random_seed
        self.max_steps = max_steps

        # Build the provided environment once, headless.
        self.env = Environment(self.grid_fp, no_gui=True, sigma=sigma,
                               random_seed=random_seed)
        # One reset to load the grid so we can read its shape / find targets.
        self.env.reset()
        self._n_cols, self._n_rows = self.env.grid.shape

        # The agent files read these instead of hard-coding numbers.
        self.state_dim = 8
        self.action_dim = ACTION_DIM

        self.steps = 0
        self._state = None

    # ── helpers to read the grid ─────────────────────────────────────────────
    def _cells_with_value(self, value: int):
        cols, rows = np.where(self.env.grid == value)
        return list(zip(cols.tolist(), rows.tolist()))

    def _curriculum_start(self, level: int):
        """Pick an empty cell a controlled Manhattan distance from a target.

        Level 0 spawns 1-3 cells away (easy), then the band widens. Beyond the
        defined bands the robot may spawn anywhere empty.
        """
        targets = self._cells_with_value(_TARGET)
        free = self._cells_with_value(0)
        if not targets or not free:
            return None

        tx, ty = targets[0]
        bands = {0: (1, 3), 1: (3, 6), 2: (6, 12)}
        d_min, d_max = bands.get(level, (1, max(self._n_cols, self._n_rows)))

        candidates = [(c, r) for (c, r) in free
                      if d_min <= abs(c - tx) + abs(r - ty) <= d_max]
        if not candidates:
            candidates = free
        return random.choice(candidates)

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, curriculum_level: int = 0):
        """Start a new episode and return the first continuous state."""
        # Reload the grid (this restores a target that was consumed last episode).
        self.env.reset()
        # Override the spawn with a curriculum-controlled start cell.
        start = self._curriculum_start(curriculum_level)
        if start is not None:
            self.env.agent_pos = (int(start[0]), int(start[1]))
        self.steps = 0
        self._state = self.get_state()
        return self._state

    # ── state construction ──────────────────────────────────────────────────
    def get_state(self):
        """Convert the grid + agent position into the continuous sensor vector."""
        grid = self.env.grid
        col, row = self.env.agent_pos
        max_len = max(self._n_cols, self._n_rows)

        # 8 obstacle rays: walk outward until a wall/obstacle or the grid edge.
        # Diagonal rays accumulate Euclidean distance (sqrt(2) per diagonal step).
        obstacle_rays = []
        for dcol, drow in _DIRECTIONS:
            dist = 0
            step_len = float(np.hypot(dcol, drow))
            c, r = col + dcol, row + drow
            while (0 <= c < self._n_cols and 0 <= r < self._n_rows
                   and grid[c, r] in _TRAVERSABLE):
                dist += step_len
                c += dcol
                r += drow
            obstacle_rays.append(dist / max_len)

        return list(obstacle_rays)

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        """Apply one action; return (state, reward, done, info)."""
        _pos, reward, terminated, info = self.env.step(action)
        self.steps += 1
        done = bool(terminated) or (self.steps >= self.max_steps)
        self._state = self.get_state()
        return self._state, reward, done, info
