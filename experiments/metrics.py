"""Comparison metrics computed from the tidy CSVs written by ``run_comparison``.

All functions take/return pandas DataFrames and aggregate across seeds with a 95%
confidence interval. Metrics (per ``(algo, grid)``):

* **final eval success rate** -- greedy success on the fixed eval start set.
* **learning-curve AUC** -- ``(1/E) Σ_e SR(e)`` over eval checkpoints, in [0,1];
  a normalized convergence / sample-efficiency measure comparable across grids.
* **episodes-to-threshold** -- first episode whose rolling(success, w) ≥ thr.
* **normalized final return** -- per-grid min-max of episode return (pooled across
  both algos) so the return scale is shared and comparable.
* **mean steps-to-goal** -- on successful eval episodes.
* **composite score** -- ``0.6·final_SR + 0.4·sample_efficiency`` -> one ranking number.

No SciPy dependency: 95% CIs use a small embedded Student-t table (bootstrap not
needed for the default 3-5 seeds); a Welch t-test helper flags significance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Two-sided 95% Student-t critical values by degrees of freedom (n-1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.96
    # nearest lower tabulated df (conservative)
    return _T95[max(k for k in _T95 if k <= df)]


def ci95(values) -> tuple[float, float, float]:
    """Return (mean, lo, hi) 95% CI via Student-t. n<2 -> zero-width."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return (float("nan"),) * 3
    mean = float(arr.mean())
    if n < 2:
        return (mean, mean, mean)
    sem = float(arr.std(ddof=1)) / np.sqrt(n)
    h = _t_crit(n - 1) * sem
    return (mean, mean - h, mean + h)


def _agg_ci(df: pd.DataFrame, value_col: str,
            group_cols=("algo", "grid")) -> pd.DataFrame:
    """Aggregate a per-seed value to mean/lo/hi/std/n per group."""
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        mean, lo, hi = ci95(g[value_col].values)
        vals = g[value_col].dropna().values
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(mean=mean, lo=lo, hi=hi,
                   std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                   n=len(vals))
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- loading
def load_results(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    train = pd.read_csv(out_dir / "train_metrics.csv")
    eval_ = pd.read_csv(out_dir / "eval_metrics.csv")
    return train, eval_


# ------------------------------------------------------------- per-seed metrics
def _final_success_per_seed(eval_df: pd.DataFrame) -> pd.DataFrame:
    f = eval_df[eval_df["eval_kind"] == "final"]
    return f[["algo", "grid", "seed", "success_rate"]].copy()


def _auc_per_seed(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Normalized learning-curve AUC = mean success_rate over eval checkpoints."""
    rows = []
    for (algo, grid, seed), g in eval_df.groupby(["algo", "grid", "seed"]):
        g = g.sort_values("episode")
        rows.append({"algo": algo, "grid": grid, "seed": seed,
                     "auc": float(g["success_rate"].mean())})
    return pd.DataFrame(rows)


def _ep_to_threshold_per_seed(train_df: pd.DataFrame, thr: float = 0.8,
                              window: int = 20) -> pd.DataFrame:
    rows = []
    for (algo, grid, seed), g in train_df.groupby(["algo", "grid", "seed"]):
        g = g.sort_values("episode")
        roll = g["success"].rolling(window, min_periods=window).mean()
        hit = g.loc[roll.values >= thr, "episode"]
        budget = int(g["episode"].max())
        ep = int(hit.iloc[0]) if len(hit) else np.nan
        rows.append({"algo": algo, "grid": grid, "seed": seed,
                     "ep_to_thr": ep, "budget": budget})
    return pd.DataFrame(rows)


def _norm_return_per_seed(train_df: pd.DataFrame, tail_frac: float = 0.1) -> pd.DataFrame:
    """Mean final-phase return per seed, min-max normalized per grid (pooled algos)."""
    rows = []
    grid_range = {}
    for grid, g in train_df.groupby("grid"):
        grid_range[grid] = (g["episode_return"].min(), g["episode_return"].max())
    for (algo, grid, seed), g in train_df.groupby(["algo", "grid", "seed"]):
        g = g.sort_values("episode")
        tail = max(1, int(len(g) * tail_frac))
        raw = float(g["episode_return"].iloc[-tail:].mean())
        lo, hi = grid_range[grid]
        norm = (raw - lo) / (hi - lo) if hi > lo else 0.0
        rows.append({"algo": algo, "grid": grid, "seed": seed,
                     "raw_return": raw, "norm_return": norm})
    return pd.DataFrame(rows)


def _steps_to_goal_per_seed(eval_df: pd.DataFrame) -> pd.DataFrame:
    f = eval_df[eval_df["eval_kind"] == "final"]
    return f[["algo", "grid", "seed", "mean_steps_to_goal"]].copy()


def _composite_per_seed(eval_df: pd.DataFrame, train_df: pd.DataFrame,
                        thr: float = 0.8, w_final: float = 0.6,
                        w_eff: float = 0.4) -> pd.DataFrame:
    fs = _final_success_per_seed(eval_df).rename(columns={"success_rate": "final_sr"})
    et = _ep_to_threshold_per_seed(train_df, thr=thr)
    m = fs.merge(et, on=["algo", "grid", "seed"], how="left")
    eff = 1.0 - (m["ep_to_thr"] / m["budget"])
    eff = eff.clip(lower=0.0, upper=1.0).fillna(0.0)  # never reached -> 0
    m["efficiency"] = eff
    m["composite"] = w_final * m["final_sr"] + w_eff * m["efficiency"]
    return m[["algo", "grid", "seed", "final_sr", "efficiency", "composite"]]


# ----------------------------------------------------------- aggregated metrics
def final_success(eval_df: pd.DataFrame) -> pd.DataFrame:
    return _agg_ci(_final_success_per_seed(eval_df), "success_rate")


def learning_curve_auc(eval_df: pd.DataFrame) -> pd.DataFrame:
    return _agg_ci(_auc_per_seed(eval_df), "auc")


def episodes_to_threshold(train_df: pd.DataFrame, thr: float = 0.8,
                          window: int = 20) -> pd.DataFrame:
    per = _ep_to_threshold_per_seed(train_df, thr=thr, window=window)
    agg = _agg_ci(per, "ep_to_thr")
    reached = per.groupby(["algo", "grid"])["ep_to_thr"].apply(
        lambda s: float(np.mean(~np.isnan(s)))).rename("reached_frac")
    return agg.merge(reached.reset_index(), on=["algo", "grid"], how="left")


def normalized_return(train_df: pd.DataFrame) -> pd.DataFrame:
    return _agg_ci(_norm_return_per_seed(train_df), "norm_return")


def steps_to_goal(eval_df: pd.DataFrame) -> pd.DataFrame:
    return _agg_ci(_steps_to_goal_per_seed(eval_df), "mean_steps_to_goal")


def composite_score(eval_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    return _agg_ci(_composite_per_seed(eval_df, train_df), "composite")


# ------------------------------------------------------------ significance test
def welch_ttest(a, b) -> dict:
    """Welch's two-sample t-test (unequal variance). Returns t, df, significant@95%."""
    a = np.asarray(list(a), float); a = a[~np.isnan(a)]
    b = np.asarray(list(b), float); b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan"), "significant": False}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": float("inf"), "df": float(na + nb - 2),
                "significant": a.mean() != b.mean()}
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return {"t": float(t), "df": float(df),
            "significant": bool(abs(t) >= _t_crit(int(round(df))))}


def paired_ttest(diffs) -> dict:
    """Paired t-test on per-seed differences (Dueling - DQN).

    Both algorithms share the same seeds (same environment randomness per seed),
    so a paired/blocked test removes between-seed variance and is more powerful
    and more appropriate than the unpaired Welch test.
    """
    d = np.asarray(list(diffs), float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"t": float("nan"), "df": float("nan"), "significant": False,
                "mean_diff": float(d.mean()) if n else float("nan"), "n": n}
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0:
        return {"t": float("inf"), "df": n - 1, "significant": md != 0,
                "mean_diff": md, "n": n}
    t = md / (sd / np.sqrt(n))
    return {"t": float(t), "df": n - 1,
            "significant": bool(abs(t) >= _t_crit(n - 1)), "mean_diff": md, "n": n}


def _paired_by_seed(per_seed: pd.DataFrame, grid: str, value: str,
                    a: str = "DuelingDQN", b: str = "DQN"):
    """Return aligned (a - b) per-seed differences for a grid (matched on seed)."""
    sub = per_seed[per_seed.grid == grid]
    piv = sub.pivot_table(index="seed", columns="algo", values=value)
    if a not in piv or b not in piv:
        return None
    return (piv[a] - piv[b]).dropna()


# ------------------------------------------------------------------ summary api
def summary_table(out_dir: str | Path, save: bool = True) -> pd.DataFrame:
    """One row per (algo, grid) with all metrics (mean ± 95% CI)."""
    train_df, eval_df = load_results(out_dir)

    parts = {
        "final_success": final_success(eval_df),
        "auc": learning_curve_auc(eval_df),
        "ep_to_thr": episodes_to_threshold(train_df),
        "norm_return": normalized_return(train_df),
        "steps_to_goal": steps_to_goal(eval_df),
        "composite": composite_score(eval_df, train_df),
    }

    base = None
    for name, df in parts.items():
        cols = {"mean": f"{name}_mean", "lo": f"{name}_lo", "hi": f"{name}_hi"}
        sel = df[["algo", "grid", "mean", "lo", "hi"]].rename(columns=cols)
        base = sel if base is None else base.merge(sel, on=["algo", "grid"], how="outer")

    base = base.sort_values(["grid", "algo"]).reset_index(drop=True)
    if save:
        out = Path(out_dir) / "summary_table.csv"
        base.to_csv(out, index=False)
    return base


def print_summary(out_dir: str | Path) -> None:
    """Human-readable console summary + pairwise significance on final success."""
    train_df, eval_df = load_results(out_dir)
    tbl = summary_table(out_dir, save=True)
    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("\n=== Summary (mean per algo/grid) ===")
    show = tbl[["grid", "algo", "final_success_mean", "auc_mean",
                "ep_to_thr_mean", "steps_to_goal_mean", "composite_mean"]]
    print(show.round(3).to_string(index=False))

    # Paired test is primary (same seeds => matched design); Welch shown for ref.
    fs = _final_success_per_seed(eval_df)
    auc = _auc_per_seed(eval_df)
    print("\n=== DuelingDQN vs DQN significance: PAIRED t-test (same seeds), per grid ===")
    print(f"  {'grid':16s} {'metric':14s} {'mean diff':>9s}  {'t':>6s} {'df':>3s}  verdict")
    for grid in sorted(fs["grid"].unique()):
        for label, per_seed, value in [("final_success", fs, "success_rate"),
                                       ("auc", auc, "auc")]:
            diffs = _paired_by_seed(per_seed, grid, value)
            if diffs is None or len(diffs) < 2:
                continue
            r = paired_ttest(diffs)
            verdict = "SIGNIFICANT" if r["significant"] else "n.s."
            print(f"  {grid:16s} {label:14s} {r['mean_diff']:+9.3f}  "
                  f"{r['t']:6.2f} {int(r['df']):3d}  {verdict} (95%)")

    print("\n=== (reference) Welch unpaired t-test on final success ===")
    for grid in sorted(fs["grid"].unique()):
        a = fs[(fs.grid == grid) & (fs.algo == "DQN")]["success_rate"]
        b = fs[(fs.grid == grid) & (fs.algo == "DuelingDQN")]["success_rate"]
        if len(a) and len(b):
            r = welch_ttest(b, a)
            print(f"  {grid:16s} Dueling {b.mean():.3f} vs DQN {a.mean():.3f} | "
                  f"t={r['t']:.2f} df={r['df']:.1f} "
                  f"{'SIGNIFICANT' if r['significant'] else 'n.s.'} (95%)")


if __name__ == "__main__":
    import sys
    print_summary(sys.argv[1] if len(sys.argv) > 1 else "experiments/results")
