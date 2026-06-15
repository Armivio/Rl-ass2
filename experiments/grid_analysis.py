"""Grid analysis utilities: reachability, Manhattan stats, greedy-trap check, render.

Grid convention (from ``world.grid.Grid``):
    cells : np.int8 array of shape (n_cols, n_rows), indexed ``cells[col, row]``.
    values: 0=empty, 1=wall/boundary, 2=obstacle, 3=target, 4=start-marker.

Action convention (from ``world.helpers.ACTIONS_TO_DIRECTIONS``):
    0=Down (0,+1), 1=Up (0,-1), 2=Left (-1,0), 3=Right (+1,0)   as (dcol, drow).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


# (dcol, drow) for the four discrete actions, matching world.helpers.
ACTION_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),    # 0 Down
    (0, -1),   # 1 Up
    (-1, 0),   # 2 Left
    (1, 0),    # 3 Right
)

# Cells the agent may occupy / traverse.
_WALKABLE = (0, 3, 4)   # empty, target, start-marker
_BLOCKING = (1, 2)      # wall, obstacle


def find_target(cells: np.ndarray) -> tuple[int, int]:
    """Return the (col, row) of the single target cell (value 3)."""
    targets = np.argwhere(cells == 3)
    if len(targets) == 0:
        raise ValueError("Grid has no target cell (value 3).")
    if len(targets) > 1:
        raise ValueError(f"Grid has {len(targets)} targets; expected exactly one.")
    col, row = targets[0]
    return int(col), int(row)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _walkable(cells: np.ndarray, pos: tuple[int, int]) -> bool:
    c, r = pos
    n_cols, n_rows = cells.shape
    if not (0 <= c < n_cols and 0 <= r < n_rows):
        return False
    return int(cells[c, r]) in _WALKABLE


def bfs_reachable(cells: np.ndarray, target_pos: tuple[int, int] | None = None) -> np.ndarray:
    """4-connected BFS from the target over walkable cells.

    Returns a boolean mask (same shape as ``cells``) that is True for every cell
    reachable from the target (the target itself included). Walls/obstacles block.
    """
    if target_pos is None:
        target_pos = find_target(cells)

    reached = np.zeros(cells.shape, dtype=bool)
    if not _walkable(cells, target_pos):
        return reached

    q: deque[tuple[int, int]] = deque([target_pos])
    reached[target_pos] = True
    while q:
        c, r = q.popleft()
        for dc, dr in ACTION_DIRECTIONS:
            nc, nr = c + dc, r + dr
            if _walkable(cells, (nc, nr)) and not reached[nc, nr]:
                reached[nc, nr] = True
                q.append((nc, nr))
    return reached


def unreachable_empty_cells(cells: np.ndarray,
                            target_pos: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Empty cells (value 0) from which the target cannot be reached."""
    reached = bfs_reachable(cells, target_pos)
    out = []
    for c, r in map(tuple, np.argwhere(cells == 0)):
        if not reached[c, r]:
            out.append((int(c), int(r)))
    return out


def greedy_descent(cells: np.ndarray,
                   start: tuple[int, int],
                   target_pos: tuple[int, int] | None = None,
                   max_steps: int = 1000) -> dict:
    """Simulate a deterministic Manhattan-greedy oracle from ``start``.

    At each step it picks the action that *strictly* reduces Manhattan distance
    to the target (blocked moves count as staying put, i.e. no reduction). Ties
    break on lowest action index. If no action strictly reduces the distance the
    oracle is declared *stuck* (a local optimum). This models the failure mode a
    distance-greedy policy / naive shaping would exhibit.

    Returns ``{reached, stuck, steps, path, final_pos, final_dist}``.
    """
    if target_pos is None:
        target_pos = find_target(cells)

    pos = (int(start[0]), int(start[1]))
    path = [pos]
    for step in range(max_steps):
        if pos == target_pos:
            return {"reached": True, "stuck": False, "steps": step,
                    "path": path, "final_pos": pos, "final_dist": 0}

        cur = manhattan(pos, target_pos)
        best_action, best_dist = None, cur
        for action, (dc, dr) in enumerate(ACTION_DIRECTIONS):
            nxt = (pos[0] + dc, pos[1] + dr)
            reachable = _walkable(cells, nxt)
            d = manhattan(nxt, target_pos) if reachable else cur  # blocked => stay
            if d < best_dist:
                best_dist, best_action = d, action

        if best_action is None:                      # no strictly-improving move
            return {"reached": False, "stuck": True, "steps": step,
                    "path": path, "final_pos": pos, "final_dist": cur}

        dc, dr = ACTION_DIRECTIONS[best_action]
        pos = (pos[0] + dc, pos[1] + dr)
        path.append(pos)

    return {"reached": pos == target_pos, "stuck": False, "steps": max_steps,
            "path": path, "final_pos": pos, "final_dist": manhattan(pos, target_pos)}


def analyze_grid(cells: np.ndarray) -> dict:
    """Summary statistics for a grid array."""
    target_pos = find_target(cells)
    reached = bfs_reachable(cells, target_pos)
    empties = np.argwhere(cells == 0)
    n_empty = len(empties)
    reachable_empty = sum(1 for c, r in map(tuple, empties) if reached[c, r])

    if n_empty:
        dists = np.abs(empties - np.asarray(target_pos)).sum(axis=1)
        max_manhattan = int(dists.max())
    else:
        max_manhattan = 0

    return {
        "shape": tuple(int(x) for x in cells.shape),
        "n_empty": int(n_empty),
        "n_walls": int(np.sum(cells == 1)),
        "n_obstacles": int(np.sum(cells == 2)),
        "n_targets": int(np.sum(cells == 3)),
        "target_pos": target_pos,
        "max_manhattan_from_target": max_manhattan,
        "reachable_empty_frac": (reachable_empty / n_empty) if n_empty else 0.0,
    }


def render_grid(cells: np.ndarray,
                out_path: str | Path,
                agent_pos: tuple[int, int] | None = None,
                path: list[tuple[int, int]] | None = None,
                title: str | None = None) -> Path:
    """Render a grid to PNG. x increases right, y(row) increases downward."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 0 empty(white) 1 wall(black) 2 obstacle(gray) 3 target(green) 4 start(blue)
    cmap = ListedColormap(["#ffffff", "#000000", "#888888", "#2ca02c", "#1f77b4"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    # cells is (col, row); transpose so imshow rows=row(y, down), cols=col(x, right).
    ax.imshow(cells.T, cmap=cmap, norm=norm, origin="upper")

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="#d62728", linewidth=1.5, alpha=0.8)
    if agent_pos is not None:
        ax.plot(agent_pos[0], agent_pos[1], marker="o", color="#ff7f0e", markersize=8)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or "Grid")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
