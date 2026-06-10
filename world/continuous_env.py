"""Continuous-observation wrapper around the discrete grid Environment.

The underlying :class:`world.environment.Environment` is a discrete grid world
(integer cell positions, four discrete actions). For Assignment 2 we keep the
*dynamics* discrete but expose a **continuous observation vector** to the agent,
turning the task into a continuous-state / discrete-action MDP suitable for Deep
RL.

Observation design (rubric-compliant)
-------------------------------------
The agent perceives the world through ``n_rays`` range sensors (LiDAR-style),
evenly spaced around it. Each ray reports the *normalised distance* to the
nearest blocking cell in that direction. A "blocking cell" is any non-empty
cell: a boundary wall, an obstacle, *or the target*. Crucially the rays are
**untyped** -- the reading is identical whether the ray hits an obstacle or the
target, and there is no displacement / direction / distance-to-target feature.
The agent therefore cannot read off where the target is; it must *learn* a
navigation policy from local perception, satisfying the assignment constraint
that the state must not trivially reveal the optimal action.

Because a single set of untyped readings is perceptually aliased (different
cells can produce identical readings), the wrapper optionally stacks the last
``frame_stack`` observations so the network can use short-horizon context.

A simple distance-banded *curriculum* spawns the agent at a controlled
Manhattan distance from the target so early episodes contain easy successes that
seed learning, then widens as the agent improves.
"""
from __future__ import annotations

import math
from pathlib import Path
from collections import deque

import numpy as np

from world.environment import Environment


# Eight ray directions as (delta_col, delta_row), matching the environment's
# (col, row) position convention. Orthogonal rays have length 1 per step,
# diagonal rays sqrt(2) per step (Euclidean), so distances are comparable.
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
        n_rays: Number of range sensors (must match ``len(RAY_DIRECTIONS)`` for
            now; kept as an arg for clarity / future extension).
        max_ray_len: Distance (in grid units) that normalises a ray reading to
            1.0. Defaults to the grid diagonal, so every reading lies in [0, 1].
        sigma: Transition stochasticity passed to the underlying environment.
        max_steps: Episode step budget; the episode is *truncated* (not treated
            as a terminal absorbing state) when it is exceeded.
        frame_stack: Number of consecutive observations concatenated together.
        random_seed: Seed for the wrapper's own RNG (start-position sampling).
            The underlying environment is seeded with the same value.
        reward_fn: Optional custom reward function; defaults to the environment's
            built-in reward (-1 step, -5 collision, +10 target).
    """

    def __init__(self,
                 grid_fp: Path,
                 n_rays: int = 8,
                 max_ray_len: float | None = None,
                 sigma: float = 0.0,
                 max_steps: int = 200,
                 frame_stack: int = 1,
                 random_seed: int = 0,
                 reward_fn: callable | None = None):
        if n_rays != len(RAY_DIRECTIONS):
            raise ValueError(
                f"n_rays={n_rays} but {len(RAY_DIRECTIONS)} ray directions are "
                f"defined. Adjust RAY_DIRECTIONS to change the sensor count.")

        self.grid_fp = Path(grid_fp)
        self.n_rays = n_rays
        self.sigma = sigma
        self.max_steps = max_steps
        self.frame_stack = max(1, int(frame_stack))
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        # Underlying discrete environment (head-less; we never use its GUI here).
        reward_fn = reward_fn or Environment._default_reward_function
        self.env = Environment(grid_fp=self.grid_fp,
                               no_gui=True,
                               sigma=sigma,
                               reward_fn=reward_fn,
                               target_fps=-1,
                               random_seed=random_seed)

        # The layout is static, so we can precompute geometry once from a fresh
        # load of the grid (target location, empty cells, normalisation length).
        from world.grid import Grid
        cells = Grid.load_grid(self.grid_fp).cells
        self.grid_shape = cells.shape
        n_cols, n_rows = self.grid_shape
        self.max_ray_len = (max_ray_len if max_ray_len is not None
                            else math.hypot(n_cols, n_rows))

        target_cells = np.argwhere(cells == 3)
        if len(target_cells) == 0:
            raise ValueError(f"Grid {self.grid_fp} has no target (cell value 3).")
        # Use the first target as the curriculum anchor.
        self._target_pos = tuple(int(x) for x in target_cells[0])
        self._empty_cells = np.argwhere(cells == 0)  # (col, row) rows

        # Single-frame observation size and the (possibly stacked) agent view.
        self.single_obs_size = n_rays
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._step_count = 0

    # ------------------------------------------------------------------ specs
    @property
    def obs_size(self) -> int:
        """Length of the (stacked) observation vector returned to the agent."""
        return self.single_obs_size * self.frame_stack

    @property
    def n_actions(self) -> int:
        return 4

    # -------------------------------------------------------------- internals
    def _cast_ray(self, grid: np.ndarray, start: tuple[int, int],
                  direction: tuple[int, int]) -> float:
        """Return the normalised distance from ``start`` to the nearest blocker.

        Marches one grid cell at a time along ``direction`` until a non-empty
        cell is hit. The boundary wall guarantees termination. The reading is
        untyped: walls, obstacles and the target all count identically.
        """
        dcol, drow = direction
        step_len = math.hypot(dcol, drow)
        c, r = start
        dist = 0.0
        n_cols, n_rows = grid.shape
        while True:
            c += dcol
            r += drow
            dist += step_len
            # Out-of-bounds safety (shouldn't trigger: borders are walls).
            if not (0 <= c < n_cols and 0 <= r < n_rows):
                break
            if grid[c, r] != 0:  # wall / obstacle / target -> blocker
                break
        return min(dist, self.max_ray_len) / self.max_ray_len

    def _raw_observation(self) -> np.ndarray:
        """Single-frame observation: one normalised distance per ray."""
        grid = self.env.grid
        pos = self.env.agent_pos
        obs = np.empty(self.n_rays, dtype=np.float32)
        for i, direction in enumerate(RAY_DIRECTIONS):
            obs[i] = self._cast_ray(grid, pos, direction)
        return obs

    def _stacked_observation(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0).astype(np.float32)

    def _sample_start_pos(self, curriculum_radius: int | None) -> tuple[int, int]:
        """Sample an empty start cell within ``curriculum_radius`` (Manhattan)
        of the target. ``None`` means anywhere on the grid (full difficulty)."""
        empties = self._empty_cells
        if curriculum_radius is not None:
            tgt = np.asarray(self._target_pos)
            manhattan = np.abs(empties - tgt).sum(axis=1)
            within = empties[manhattan <= curriculum_radius]
            if len(within) > 0:
                empties = within
        idx = int(self._rng.integers(len(empties)))
        col, row = empties[idx]
        return int(col), int(row)

    # ------------------------------------------------------------------- API
    def reset(self,
              curriculum_radius: int | None = None,
              agent_start_pos: tuple[int, int] | None = None) -> np.ndarray:
        """Reset the episode and return the initial (stacked) observation.

        Args:
            curriculum_radius: If given (and ``agent_start_pos`` is None), the
                agent spawns within this Manhattan distance of the target.
            agent_start_pos: Explicit ``(col, row)`` start; overrides curriculum.
        """
        if agent_start_pos is None:
            agent_start_pos = self._sample_start_pos(curriculum_radius)

        # The underlying env reloads the grid (restoring the target) each reset.
        self.env.reset(agent_start_pos=agent_start_pos, no_gui=True)
        self._step_count = 0

        frame = self._raw_observation()
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        return self._stacked_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one step.

        Returns ``(obs, reward, done, info)`` where ``done`` is True on either
        target reach (terminal) or step-budget exhaustion (truncation). ``info``
        carries ``target_reached`` (use this as the bootstrap-terminal flag) and
        ``truncated`` so the learner does not zero-bootstrap on truncation.
        """
        _, reward, terminated, env_info = self.env.step(action)
        self._step_count += 1

        target_reached = bool(env_info.get("target_reached", False)) or terminated
        truncated = (not target_reached) and (self._step_count >= self.max_steps)
        done = target_reached or truncated

        self._frames.append(self._raw_observation())
        obs = self._stacked_observation()

        info = {
            "target_reached": target_reached,
            "truncated": truncated,
            "agent_pos": self.env.agent_pos,
            "actual_action": env_info.get("actual_action"),
        }
        return obs, float(reward), done, info

    @property
    def agent_pos(self) -> tuple[int, int]:
        return self.env.agent_pos

    @property
    def target_pos(self) -> tuple[int, int]:
        return self._target_pos
