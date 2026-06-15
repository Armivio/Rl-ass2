"""Visualisations for the DQN vs Dueling DQN comparison (matplotlib only).

Replaces the original tables with figures: learning curves with 95% CI bands
(DQN vs Dueling overlaid, one panel per grid), and grouped bar charts for the
aggregate metrics. Every function takes tidy DataFrames + an output path, saves a
PNG, and returns the path.

    python -m experiments.plots [out_dir]
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import metrics
from experiments.metrics import ci95, _t_crit


COLORS = {"DQN": "#1f77b4", "DuelingDQN": "#ff7f0e"}
_ALGO_ORDER = ["DQN", "DuelingDQN"]


def _algos(df: pd.DataFrame) -> list[str]:
    return [a for a in _ALGO_ORDER if a in set(df["algo"])] or sorted(df["algo"].unique())


def _grids(df: pd.DataFrame) -> list[str]:
    return sorted(df["grid"].unique())


def _grid_axes(n: int):
    # Prefer square-ish layouts: 4 grids -> 2x2 (not 2x3 with empty panels).
    ncols = n if n <= 3 else (2 if n == 4 else 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows),
                             squeeze=False)
    return fig, axes.flatten(), nrows * ncols


# ---------------------------------------------------------------- learning curves
def _checkpoint_ci(eval_df: pd.DataFrame, algo: str, grid: str, value="success_rate"):
    g = eval_df[(eval_df.algo == algo) & (eval_df.grid == grid)]
    eps = sorted(g["episode"].unique())
    means, los, his = [], [], []
    for e in eps:
        m, lo, hi = ci95(g[g.episode == e][value].values)
        means.append(m); los.append(lo); his.append(hi)
    return np.array(eps), np.array(means), np.array(los), np.array(his)


def plot_learning_curves_success(eval_df: pd.DataFrame, out_path: str | Path) -> Path:
    grids = _grids(eval_df)
    fig, axes, total = _grid_axes(len(grids))
    for ax, grid in zip(axes, grids):
        for algo in _algos(eval_df):
            eps, mean, lo, hi = _checkpoint_ci(eval_df, algo, grid)
            if len(eps) == 0:
                continue
            c = COLORS.get(algo, None)
            ax.plot(eps, mean, color=c, label=algo, linewidth=2)
            ax.fill_between(eps, lo, hi, color=c, alpha=0.2)
        ax.set_title(grid)
        ax.set_xlabel("training episode")
        ax.set_ylabel("eval success rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes[len(grids):]:
        ax.axis("off")
    fig.suptitle("Evaluation success rate over training (mean ± 95% CI)")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_learning_curves_return(train_df: pd.DataFrame, out_path: str | Path,
                                smooth: int = 20) -> Path:
    grids = _grids(train_df)
    fig, axes, total = _grid_axes(len(grids))
    for ax, grid in zip(axes, grids):
        for algo in _algos(train_df):
            g = train_df[(train_df.algo == algo) & (train_df.grid == grid)]
            piv = g.pivot_table(index="episode", columns="seed", values="episode_return")
            sm = piv.rolling(smooth, min_periods=1).mean()
            vals = sm.values
            mean = np.nanmean(vals, axis=1)
            n = vals.shape[1]
            if n > 1:
                sd = np.nanstd(vals, axis=1, ddof=1)
                h = _t_crit(n - 1) * sd / math.sqrt(n)
            else:
                h = np.zeros_like(mean)
            x = sm.index.values
            c = COLORS.get(algo, None)
            ax.plot(x, mean, color=c, label=algo, linewidth=2)
            ax.fill_between(x, mean - h, mean + h, color=c, alpha=0.2)
        ax.set_title(grid)
        ax.set_xlabel("training episode")
        ax.set_ylabel(f"episode return (MA{smooth})")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes[len(grids):]:
        ax.axis("off")
    fig.suptitle("Training return over time (mean ± 95% CI)")
    fig.tight_layout()
    return _save(fig, out_path)


# --------------------------------------------------------------------- bar charts
def _grouped_bars(agg: pd.DataFrame, ax, ylabel: str, title: str,
                  clip01: bool = False):
    grids = sorted(agg["grid"].unique())
    algos = [a for a in _ALGO_ORDER if a in set(agg["algo"])]
    x = np.arange(len(grids))
    width = 0.8 / max(1, len(algos))
    for i, algo in enumerate(algos):
        sub = agg[agg.algo == algo].set_index("grid").reindex(grids)
        means = sub["mean"].values
        lo, hi = sub["lo"].values, sub["hi"].values
        if clip01:
            # A CI for a [0,1] rate is meaningless outside [0,1]; clip whiskers.
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        yerr = np.vstack([means - lo, hi - means])
        ax.bar(x + i * width, means, width, label=algo,
               color=COLORS.get(algo), yerr=yerr, capsize=4)
    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels(grids, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()


def plot_final_success_bars(eval_df: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(_grids(eval_df)), 4.5))
    _grouped_bars(metrics.final_success(eval_df), ax,
                  "final eval success rate", "Final success rate (mean ± 95% CI)",
                  clip01=True)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_auc_bars(eval_df: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(_grids(eval_df)), 4.5))
    _grouped_bars(metrics.learning_curve_auc(eval_df), ax,
                  "normalized learning-curve AUC",
                  "Convergence AUC (mean ± 95% CI)", clip01=True)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_composite_bars(eval_df: pd.DataFrame, train_df: pd.DataFrame,
                        out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(_grids(eval_df)), 4.5))
    _grouped_bars(metrics.composite_score(eval_df, train_df), ax,
                  "composite score", "Composite score (mean ± 95% CI)", clip01=True)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_steps_to_goal_dist(eval_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Boxplot of steps-to-goal at final eval, grouped by algo, per grid."""
    f = eval_df[eval_df.eval_kind == "final"].dropna(subset=["mean_steps_to_goal"])
    grids = _grids(eval_df)
    algos = _algos(eval_df)
    fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(grids), 4.5))
    positions, data, colors, ticks = [], [], [], []
    for gi, grid in enumerate(grids):
        for ai, algo in enumerate(algos):
            vals = f[(f.grid == grid) & (f.algo == algo)]["mean_steps_to_goal"].values
            positions.append(gi * (len(algos) + 1) + ai)
            data.append(vals if len(vals) else [np.nan])
            colors.append(COLORS.get(algo))
        ticks.append((gi * (len(algos) + 1) + (len(algos) - 1) / 2, grid))
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks],
                                                             rotation=20, ha="right")
    ax.set_ylabel("steps to goal (successful eval episodes)")
    ax.set_title("Path efficiency (lower is better)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS.get(a), alpha=0.6) for a in algos]
    ax.legend(handles, algos)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, out_path)


# ------------------------------------------------------------------------ helpers
def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def make_all_plots(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    train_df, eval_df = metrics.load_results(out_dir)
    pdir = out_dir / "plots"
    paths = [
        plot_learning_curves_success(eval_df, pdir / "learning_curves_success.png"),
        plot_learning_curves_return(train_df, pdir / "learning_curves_return.png"),
        plot_final_success_bars(eval_df, pdir / "final_success_bars.png"),
        plot_auc_bars(eval_df, pdir / "auc_bars.png"),
        plot_composite_bars(eval_df, train_df, pdir / "composite_bars.png"),
        plot_steps_to_goal_dist(eval_df, pdir / "steps_to_goal.png"),
    ]
    print("Saved plots:")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    import sys
    make_all_plots(sys.argv[1] if len(sys.argv) > 1 else "experiments/results")
