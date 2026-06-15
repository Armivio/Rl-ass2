"""Deterministic evaluation protocol shared by both agents.

The single biggest comparability fix: a **fixed set of evaluation start positions
per grid**, derived from a hash of the grid name (NOT the training seed). Every
algorithm and every run-seed is therefore evaluated from the *same* starts, so
success rates are directly comparable and reproducible. This removes the
random-start eval noise that made the original Dueling-DQN numbers unstable.

Greedy (exploitation) rollouts only; raw (undiscounted) return is reported so the
return scale is identical for DQN and Dueling DQN.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from world.grid import Grid
from experiments.grid_analysis import find_target


def _grid_seed(grid_fp: Path) -> int:
    """Stable per-grid seed from the file stem (process-independent)."""
    return zlib.crc32(Path(grid_fp).stem.encode("utf-8")) & 0xFFFFFFFF


def sample_eval_starts(grid_fp: str | Path,
                       n_starts: int,
                       min_start_dist: int | None = None) -> list[tuple[int, int]]:
    """Deterministic list of ``(col, row)`` empty start cells for a grid.

    Args:
        grid_fp: Path to the ``.npy`` grid.
        n_starts: Number of start positions to return (capped at #eligible cells).
        min_start_dist: If set, only sample empty cells at least this Manhattan
            distance from the target (forces non-trivial navigation).
    """
    cells = Grid.load_grid(Path(grid_fp)).cells
    target = np.asarray(find_target(cells))
    empties = np.argwhere(cells == 0)                       # rows of [col, row]
    if len(empties) == 0:
        raise ValueError(f"Grid {grid_fp} has no empty cells.")

    dists = np.abs(empties - target).sum(axis=1)
    if min_start_dist is not None:
        keep = dists >= min_start_dist
        if keep.any():
            empties = empties[keep]

    # Deterministic, grid-specific ordering.
    rng = np.random.default_rng(_grid_seed(grid_fp))
    n = min(int(n_starts), len(empties))
    idx = rng.choice(len(empties), size=n, replace=False)
    idx.sort()
    return [(int(empties[i, 0]), int(empties[i, 1])) for i in idx]


def greedy_rollout(adapter, env, start_pos: tuple[int, int], max_steps: int) -> dict:
    """One greedy episode from a fixed start. Returns per-episode stats."""
    obs = env.reset(agent_start_pos=start_pos)
    adapter.set_eval_mode(True)

    total_return = 0.0
    steps = 0
    reached = False
    for steps in range(1, max_steps + 1):
        action = adapter.select_action(obs, greedy=True)
        obs, reward, done, info = env.step(action)
        total_return += float(reward)
        if done:
            reached = bool(info.get("target_reached", False))
            break

    adapter.set_eval_mode(False)
    return {"success": reached, "steps": steps, "ret": total_return, "reached": reached}


def evaluate(adapter, env, eval_starts: list[tuple[int, int]], max_steps: int) -> dict:
    """Aggregate greedy rollouts over the fixed eval start set.

    Returns ``{success_rate, mean_return, mean_steps_to_goal, n_eval}``.
    ``mean_steps_to_goal`` is averaged over successful episodes only (NaN if none).
    """
    successes, returns, success_steps = [], [], []
    for start in eval_starts:
        r = greedy_rollout(adapter, env, start, max_steps)
        successes.append(1.0 if r["success"] else 0.0)
        returns.append(r["ret"])
        if r["success"]:
            success_steps.append(r["steps"])

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_steps_to_goal": (float(np.mean(success_steps))
                               if success_steps else float("nan")),
        "n_eval": len(eval_starts),
    }
