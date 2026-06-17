"""Continuous-observation wrapper around the discrete grid Environment.

Observation vector (per frame)
-------------------------------
[0:8]  8 untyped LiDAR rays -- normalised distance to nearest blocking cell
       (wall, obstacle, or target are indistinguishable). 
[8:10] Orientation (optional) -- unit direction (dcol, drow) of the last
       action actually executed by the environment.  Encodes heading without
       revealing the target.  Replaces frame-stacking as the primary way to
       break perceptual aliasing.

Reward shaping (optional, additive on top of the base reward)
-------------------------------------------------------------
progress   : weight * (prev_manhattan - new_manhattan)  -- positive when the
             agent moves closer to the target; zero when it hits a wall and
             stays put.  The agent still cannot read the target position from
             its sensors; shaping only comes through the scalar reward signal.
proximity  : -weight * max(0, 1 - min_ray / threshold) -- small penalty when
             the nearest obstacle is within ``threshold`` normalised units,
             computed from the current ray observations.
"""
from __future__ import annotations

import math
from pathlib import Path
from collections import deque

import numpy as np

from world.environment import Environment
from world.helpers import ACTIONS_TO_DIRECTIONS


# Eight ray directions as (delta_col, delta_row), matching the environment's
# (col, row) position convention.  Euclidean step-lengths so diagonal and
# orthogonal readings are on the same metric scale.
RAY_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),    # down
    (1, 1),    # down-right
    (1, 0),    # right
    (1, -1),   # up-right
    (0, -1),   # up
    (-1, -1),  # up-left
    (-1, 0),   # left
    (-1, 1),   # down-left
)


class ContinuousEnvironment:
    """Wraps the discrete grid ``Environment`` with a continuous ray observation.

    Args:
        grid_fp: Path to the ``.npy`` grid file.
        n_rays: Number of range sensors. Must equal ``len(RAY_DIRECTIONS)``.
        max_ray_len: Normalisation length in grid units.  Defaults to the grid
            diagonal so every reading lies in [0, 1].
        sigma: Transition stochasticity passed to the underlying environment.
        max_steps: Episode truncation horizon.
        frame_stack: Consecutive frames concatenated into one observation.
            Set to 1 when ``use_orientation=True`` (default).
        use_orientation: Append a 2-D heading vector (dcol, drow) of the last
            executed action to each frame.  Breaks aliasing without leaking
            target information.
        progress_reward_weight: Scale for the Manhattan-distance progress bonus.
            0.0 disables it.
        obstacle_proximity_weight: Scale for the obstacle-proximity penalty.
            0.0 disables it.
        obstacle_proximity_threshold: Normalised ray distance below which the
            proximity penalty activates (e.g. 0.15 = within 15 % of max_ray_len).
        random_seed: Seed for start-position sampling.
        reward_fn: Base reward function (step/collision/target).  Defaults to
            the environment's built-in reward.
    """

    def __init__(self,
                 grid_fp: Path,
                 n_rays: int = 8,
                 max_ray_len: float | None = None,
                 sigma: float = 0.0,
                 max_steps: int = 200,
                 frame_stack: int = 1,
                 use_orientation: bool = True,
                 progress_reward_weight: float = 0.0,
                 obstacle_proximity_weight: float = 0.0,
                 obstacle_proximity_threshold: float = 0.15,
                 random_seed: int = 0,
                 reward_fn: callable | None = None):
        if n_rays != len(RAY_DIRECTIONS):
            raise ValueError(
                f"n_rays={n_rays} but {len(RAY_DIRECTIONS)} ray directions "
                f"are defined.")

        self.grid_fp = Path(grid_fp)
        self.n_rays = n_rays
        self.sigma = sigma
        self.max_steps = max_steps
        self.frame_stack = max(1, int(frame_stack))
        self.use_orientation = use_orientation
        self.progress_reward_weight = progress_reward_weight
        self.obstacle_proximity_weight = obstacle_proximity_weight
        self.obstacle_proximity_threshold = obstacle_proximity_threshold
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        reward_fn = reward_fn or Environment._default_reward_function
        self.env = Environment(grid_fp=self.grid_fp,
                               no_gui=True,
                               sigma=sigma,
                               reward_fn=reward_fn,
                               target_fps=-1,
                               random_seed=random_seed)

        from world.grid import Grid
        cells = Grid.load_grid(self.grid_fp).cells
        self.grid_shape = cells.shape
        n_cols, n_rows = self.grid_shape
        self.max_ray_len = (max_ray_len if max_ray_len is not None
                            else math.hypot(n_cols, n_rows))

        target_cells = np.argwhere(cells == 3)
        if len(target_cells) == 0:
            raise ValueError(f"Grid {self.grid_fp} has no target cell (value 3).")
        self._target_pos = tuple(int(x) for x in target_cells[0])
        self._empty_cells = np.argwhere(cells == 0)

        # orientation: 2-D direction of last executed action; (0,0) at start.
        self._orientation = np.zeros(2, dtype=np.float32)

        # single-frame size: rays + optional heading
        self.single_obs_size = n_rays + (2 if use_orientation else 0)
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._step_count = 0

    # ------------------------------------------------------------------ specs
    @property
    def obs_size(self) -> int:
        return self.single_obs_size * self.frame_stack

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def max_manhattan_from_target(self) -> int:
        tgt = np.asarray(self._target_pos)
        return int(np.abs(self._empty_cells - tgt).sum(axis=1).max())

    # -------------------------------------------------------------- internals
    def _cast_ray(self, grid: np.ndarray, start: tuple[int, int],
                  direction: tuple[int, int]) -> float:
        """Normalised Euclidean distance to nearest blocker along ``direction``."""
        dcol, drow = direction
        step_len = math.hypot(dcol, drow)
        c, r = start
        dist = 0.0
        n_cols, n_rows = grid.shape
        while True:
            c += dcol
            r += drow
            dist += step_len
            if not (0 <= c < n_cols and 0 <= r < n_rows):
                break
            if grid[c, r] != 0:
                break
        return min(dist, self.max_ray_len) / self.max_ray_len

    def _raw_observation(self) -> np.ndarray:
        """Single-frame vector: [ray_0, ..., ray_7, dcol, drow] (if orientation)."""
        grid = self.env.grid
        pos = self.env.agent_pos
        rays = np.empty(self.n_rays, dtype=np.float32)
        for i, direction in enumerate(RAY_DIRECTIONS):
            rays[i] = self._cast_ray(grid, pos, direction)
        if self.use_orientation:
            return np.concatenate([rays, self._orientation])
        return rays

    def _stacked_observation(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0).astype(np.float32)

    def _manhattan_to_target(self, pos: tuple[int, int]) -> int:
        return abs(pos[0] - self._target_pos[0]) + abs(pos[1] - self._target_pos[1])

    def _sample_start_pos(self, curriculum_radius: int | None,
                          min_dist: int | None = None) -> tuple[int, int]:
        empties = self._empty_cells
        tgt = np.asarray(self._target_pos)
        manhattan = np.abs(empties - tgt).sum(axis=1)

        if curriculum_radius is not None:
            mask = manhattan <= curriculum_radius
            if mask.any():
                empties = empties[mask]
                manhattan = manhattan[mask]

        if min_dist is not None:
            far = empties[manhattan >= min_dist]
            if len(far) > 0:
                empties = far

        idx = int(self._rng.integers(len(empties)))
        col, row = empties[idx]
        return int(col), int(row)

    # ------------------------------------------------------------------- API
    def reset(self,
              curriculum_radius: int | None = None,
              agent_start_pos: tuple[int, int] | None = None,
              min_start_dist: int | None = None) -> np.ndarray:
        """Reset and return the initial stacked observation."""
        if agent_start_pos is None:
            agent_start_pos = self._sample_start_pos(curriculum_radius,
                                                      min_dist=min_start_dist)

        self.env.reset(agent_start_pos=agent_start_pos, no_gui=True)
        self._step_count = 0
        self._orientation = np.zeros(2, dtype=np.float32)

        frame = self._raw_observation()
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        return self._stacked_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one step; returns (obs, shaped_reward, done, info).

        Reward = base_reward + progress_shaping + proximity_penalty.
        ``done`` is True on target-reach (terminal) or step-budget exhaustion
        (truncation).  Only target-reach should zero the bootstrap value.
        """
        prev_dist = self._manhattan_to_target(self.env.agent_pos)

        _, base_reward, terminated, env_info = self.env.step(action)
        self._step_count += 1

        # Update heading from the action the environment actually executed.
        if self.use_orientation:
            actual = env_info.get("actual_action")
            if actual is not None:
                d = ACTIONS_TO_DIRECTIONS[actual]
                self._orientation[0] = float(d[0])
                self._orientation[1] = float(d[1])

        raw_obs = self._raw_observation()
        self._frames.append(raw_obs)
        obs = self._stacked_observation()

        # ---- reward shaping -----------------------------------------------
        shaped_reward = float(base_reward)

        if self.progress_reward_weight != 0.0:
            new_dist = self._manhattan_to_target(self.env.agent_pos)
            # Positive when closer, negative when further, zero when stuck.
            shaped_reward += self.progress_reward_weight * (prev_dist - new_dist)

        if self.obstacle_proximity_weight != 0.0:
            min_ray = float(raw_obs[:self.n_rays].min())
            if min_ray < self.obstacle_proximity_threshold:
                shaped_reward -= self.obstacle_proximity_weight * (
                    1.0 - min_ray / self.obstacle_proximity_threshold)

        target_reached = bool(env_info.get("target_reached", False)) or terminated
        truncated = (not target_reached) and (self._step_count >= self.max_steps)
        done = target_reached or truncated

        info = {
            "target_reached": target_reached,
            "truncated": truncated,
            "agent_pos": self.env.agent_pos,
            "actual_action": env_info.get("actual_action"),
        }
        return obs, shaped_reward, done, info

    @property
    def agent_pos(self) -> tuple[int, int]:
        return self.env.agent_pos

    @property
    def target_pos(self) -> tuple[int, int]:
        return self._target_pos
