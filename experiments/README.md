# `experiments/` — unified DQN vs Dueling DQN comparison

Additive harness for Assignment 2. It **calls** the continuous environment
(`world/continuous_env.py`) and the from-scratch agents (`agents/dqn_agent.py`,
`agents/DuelingDQN_agentv2.py`) — it does **not** modify them. The goal is to make
the DQN baseline and the Dueling-DQN main method directly comparable: both run
under one protocol (same env + shaping, seeds, normalized hyper-parameters,
epsilon annealing, and a fixed deterministic evaluation start set), so any gap is
attributable to the network architecture.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch above is the CUDA wheel; for a smaller CPU-only build instead:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 1. Build the deceptive grid

A local-optima ("trap pocket") grid that deviates from Assignment 1: a
distance-greedy agent marches east into a pocket and gets stuck one cell from the
target, while the real route detours around it. Validates reachability and that
the greedy oracle is trapped, then renders a preview PNG.

```bash
python -m experiments.make_deceptive_grid          # -> grid_configs/deceptive_grid.npy (+ _preview.png)
```

## 2. Run the comparison (multi-seed)

```bash
# quick
python -m experiments.run_comparison --grids grid_configs/small_grid.npy \
    --seeds 0 --episodes 50 --eval-every 25

# full: deceptive + standard grids, 3 seeds
python -m experiments.run_comparison \
    --grids grid_configs/deceptive_grid.npy grid_configs/small_grid.npy \
            grid_configs/A1_grid.npy grid_configs/large_grid.npy \
    --seeds 0 1 2 --episodes 600
```

Writes two tidy long-format CSVs under `--out-dir` (default `experiments/results/`):

| file | one row per | key columns |
|---|---|---|
| `train_metrics.csv` | (algo, grid, seed, episode) | `episode_return, episode_length, success, epsilon, mean_loss` |
| `eval_metrics.csv` | (algo, grid, seed, eval-checkpoint) | `success_rate, mean_return, mean_steps_to_goal, eval_kind` |

Both algorithms (`DQN`, `DuelingDQN`) and reward shaping (`--progress-reward`,
`--obstacle-penalty`) are applied identically. Ablations: `--double-dqn`,
`--sigma`, `--progress-reward 0.5`, `--eval-min-dist N`.

## 3. Metrics + significance

```bash
python -m experiments.metrics experiments/results
```

Per `(algo, grid)`, aggregated across seeds with **95% CIs**: final eval success
rate, normalized **learning-curve AUC** (convergence), **episodes-to-threshold**
(sample efficiency), normalized return, mean steps-to-goal, and a **composite
score** (`0.6·final_success + 0.4·efficiency`). Also a Welch t-test of final
success (Dueling vs DQN) per grid. Writes `summary_table.csv`.

## 4. Plots (replace the old tables)

```bash
python -m experiments.plots experiments/results        # -> experiments/results/plots/
```

- `learning_curves_success.png` — eval success vs episode, DQN vs Dueling, **±95% CI band**, one panel per grid.
- `learning_curves_return.png` — smoothed training return, ±95% CI band.
- `final_success_bars.png`, `auc_bars.png`, `composite_bars.png` — grouped bars with CI error bars.
- `steps_to_goal.png` — path-efficiency boxplots.

## Module map

| file | role |
|---|---|
| `grid_analysis.py` | BFS reachability, Manhattan stats, greedy-trap check, grid render |
| `make_deceptive_grid.py` | build + validate + render the deceptive grid |
| `adapters.py` | unify the two agents' APIs + normalize hyper-parameters |
| `eval_protocol.py` | deterministic per-grid eval start set + greedy evaluation |
| `run_comparison.py` | multi-seed training loop → two CSVs |
| `metrics.py` | aggregate metrics + 95% CIs + Welch t-test |
| `plots.py` | all figures |
