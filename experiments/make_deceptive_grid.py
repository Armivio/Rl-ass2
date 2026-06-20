"""Build a *deceptive* / local-optima grid for the DQN vs Dueling-DQN comparison.

Layout concept ("trap pocket")
------------------------------
The target sits just **east** of a U-shaped wall pocket whose mouth opens **west**
toward the start region. An agent that simply reduces its Manhattan distance to
the target marches east straight into the pocket and gets stuck against the back
wall -- the target is only ~2 cells away but unreachable from inside. The single
real route detours **north** over the pocket's top arm, which a distance-greedy
policy never takes because the first detour steps *increase* the distance.

What this grid is for (scope -- read before citing it)
------------------------------------------------------
The trap is defined w.r.t. a *Manhattan-greedy* policy. DQN and Dueling DQN never
observe Manhattan distance (it is not in the state), so the pocket is NOT expected
to discriminate the two architectures -- and empirically it does not (both solve it
about equally). Its real purpose is a **reward-shaping safety / exploration
testbed**: naive progress shaping (``w*(prev-new)``) is itself a distance-greedy
signal, so it should *lure the reward-shaped agent into the trap*, whereas sparse
reward and true potential-based shaping (PBRS, ``w*(prev-gamma*new)``) should
escape. Run the grid under {sparse, naive, PBRS} (see ``run_comparison.py``'s
``--progress-reward`` / ``--progress-gamma``) to demonstrate that. The rays remain
untyped (wall / obstacle / target indistinguishable), so the layout adds difficulty
without leaking the target location into the state -- rubric-safe.

Run
---
    python -m experiments.make_deceptive_grid
    # -> grid_configs/deceptive_grid.npy  (+ a PNG preview next to it)
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from world.grid import Grid
from experiments.grid_analysis import (
    analyze_grid, greedy_descent, unreachable_empty_cells, render_grid, find_target,
)


def build_deceptive_grid(n_cols: int = 20, n_rows: int = 20) -> Grid:
    """Construct the deceptive grid. Geometry is tuned for the default 20x20."""
    if n_cols < 20 or n_rows < 20:
        raise ValueError("Deceptive layout needs at least a 20x20 grid.")

    grid = Grid(n_cols, n_rows)          # boundary walls already laid (value 1)
    cells = grid.cells

    # Centre of the playable area; target a few cells right of centre.
    cx = n_cols // 2          # 10 for 20
    cy = n_rows // 2          # 10 for 20
    tx, ty = cx + 5, cy       # (15, 10) target

    # --- U-shaped trap pocket -------------------------------------------------
    # Back wall: vertical, one cell west of the target, spanning the pocket rows.
    back_x = tx - 1                       # 14
    top_y, bot_y = ty - 3, ty + 3         # 7 .. 13  (pocket spans these rows)
    cells[back_x, top_y:bot_y + 1] = 1    # back wall (rows top_y..bot_y)

    # Two arms reaching west from the back wall (top and bottom of the U).
    mouth_x = cx - 1                       # 9  (west end of the arms = mouth edge)
    cells[mouth_x:back_x + 1, top_y] = 1   # top arm (cols mouth_x..back_x)
    cells[mouth_x:back_x + 1, bot_y] = 1   # bottom arm

    # --- decoy obstacles framing the mouth (look like a doorway) --------------
    # They narrow the visible entrance but leave the centre rows open, and they
    # do NOT block the greedy lure (row ty) nor the northern detour (row <= top_y-1).
    for (ox, oy) in [(mouth_x + 1, top_y + 1), (mouth_x + 1, bot_y - 1)]:
        cells[ox, oy] = 2

    # --- target ---------------------------------------------------------------
    grid.place_object(tx, ty, "target")

    # --- start marker in the far-west start band ------------------------------
    grid.place_object(2, cy, "start")

    return grid


def validate(cells: np.ndarray) -> dict:
    """Assert the deceptive grid has the discriminating properties.

    Raises AssertionError if any property fails; returns the analysis dict.
    """
    info = analyze_grid(cells)
    tgt = info["target_pos"]

    assert info["n_targets"] == 1, f"expected 1 target, got {info['n_targets']}"

    # Every empty cell must be able to reach the target: the trap is a *lure*,
    # not a true prison -- the optimal policy can always escape.
    orphans = unreachable_empty_cells(cells, tgt)
    assert not orphans, f"{len(orphans)} empty cells cannot reach the target: {orphans[:5]}..."

    # From the intended western start region, the Manhattan-greedy oracle must
    # FAIL to reach the target (it falls into the pocket). This is the property
    # that makes the grid discriminate greedy/under-explored policies.
    starts = [(2, cells.shape[1] // 2),          # the start marker row
              (3, cells.shape[1] // 2 - 4),       # higher west cell
              (3, cells.shape[1] // 2 + 4)]       # lower west cell
    trapped = 0
    for s in starts:
        if cells[s] != 0 and cells[s] != 4:
            continue
        res = greedy_descent(cells, s, tgt)
        if not res["reached"]:
            trapped += 1
    assert trapped >= 1, "greedy oracle was not trapped from any western start " \
                         "-- the layout is not deceptive."

    info["greedy_trapped_starts"] = trapped
    return info


def main(out_path: str | Path = "grid_configs/deceptive_grid.npy",
         n_cols: int = 20, n_rows: int = 20, render: bool = True) -> Path:
    out_path = Path(out_path)
    grid = build_deceptive_grid(n_cols, n_rows)

    info = validate(grid.cells)
    print("Deceptive grid analysis:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    grid.save_grid_file(out_path)
    saved = out_path.with_suffix(".npy")
    print(f"Saved grid to {saved}")

    if render:
        # Render the layout, overlaying the greedy oracle's trapped path.
        start = (2, n_rows // 2)
        res = greedy_descent(grid.cells, start, find_target(grid.cells))
        png = saved.with_name(saved.stem + "_preview.png")
        render_grid(grid.cells, png, agent_pos=start, path=res["path"],
                    title=f"deceptive_grid (greedy trapped: {not res['reached']})")
        print(f"Saved preview to {png}")

    return saved


def parse_args():
    p = ArgumentParser(description="Build the deceptive local-optima grid.")
    p.add_argument("--out", type=Path, default=Path("grid_configs/deceptive_grid.npy"))
    p.add_argument("--n_cols", type=int, default=20)
    p.add_argument("--n_rows", type=int, default=20)
    p.add_argument("--no_render", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.out, args.n_cols, args.n_rows, render=not args.no_render)
