# Reinforcement Learning on a Grid-World

A delivery-robot grid-world solved with three families of tabular
reinforcement-learning algorithms, for the 2AMC15 Data Intelligence Challenge. 
The agent must navigate a discrete grid of walls and obstacles to
reach a target cell under a stochastic transition model, while maximising its
cumulative reward.

| Family | Algorithm | Implementation |
|---|---|---|
| Dynamic programming | Value iteration | `agents/value_iteration_agent.py` |
| Temporal difference | Tabular Q-learning | `agents/QLearning_agent.py` |
| Monte Carlo | On-policy first-visit MC control | `agents/on_policy_mc.py` |
| Monte Carlo | Off-policy MC control (weighted / ordinary IS) | `agents/off_policy_mc.py` |

## Setup

Requires Python >= 3.10.

```bash
python -m venv venv
# Windows:        venv\Scripts\activate
# Linux / macOS:  source venv/bin/activate
pip install -r requirements.txt
```

## Repository structure

```
agents/                      RL agents (all implement agents/base_agent.py)
  value_iteration_agent.py     value iteration (DP)
  QLearning_agent.py           tabular Q-learning (TD)
  on_policy_mc.py              on-policy first-visit MC control
  off_policy_mc.py             off-policy MC control (weighted + ordinary IS)
  null_agent.py, random_agent.py   benchmark baselines
world/                       grid-world simulator (environment, grid, GUI, grid creator)
grid_configs/                grid layouts (.npy), including the mandatory A1_grid.npy
results/                     Q-learning and Monte-Carlo outputs
results/VI/                  multi-seed value-iteration outputs and report figures
results/VI/demo_runs/        outputs and path figures from single GUI/evaluation runs

run_experiments.py           runs the six value-iteration experiments
QLearning_train.py           Q-learning hyper-parameter sweep
train_mc.py                  Monte-Carlo hyper-parameter sweep
train_VI.py                  single-run trainer / GUI demo
aggregate_results.py         aggregates multi-seed runs into summary.csv + plots
make_vi_figures.py           renders value / policy / convergence figures
create_custom_grid.py        builds grid_configs/custom_grid.npy (VI experiment 4)
evaluate.py                  plotting helpers (imported by the scripts above)
report.tex, ref.bib          report sources (LaTeX)
```

## Reproducing the report

Each results table in the report is produced by one script. First build the
custom grid used by value-iteration experiment 4:

```bash
python create_custom_grid.py
```

### Value iteration — "VI results" table

```bash
python run_experiments.py --experiment all --seeds 42 43 44 45 46 --n_eval_episodes 200 --max_steps 1000
python aggregate_results.py
python make_vi_figures.py
```

`aggregate_results.py` writes `results/VI/summary_VI.csv` (the mean ± std
numbers in the table); `make_vi_figures.py` writes the value-function, policy
and convergence figures used in the report to `results/VI/figures/`.

### Q-learning — "Q-learning results" table

```bash
python QLearning_train.py grid_configs/A1_grid.npy grid_configs/large_grid.npy --no_gui --start_pos 1,12 --gammas 0.9 0.6 --sigmas 0.02 0.5 --epsilons 0.1 0.3 --max_steps 200 1000 --iter 1000 --eval_iter 100 --alpha 0.1
```

Runs all 32 configurations (16 per grid). The combined results land in
`results/q_learning/summary/q_learning_eval_summary.csv`, with per-setup
training logs and convergence plots under `results/q_learning/setup_*/`.

### Monte Carlo — "MC results" table

```bash
python train_mc.py --grids grid_configs/A1_grid.npy grid_configs/large_grid.npy --episodes 1000 --out results/mc_results.csv
```

Trains the on-policy and off-policy MC agents over all 32 configurations and
writes one row per (grid × algorithm × hyper-parameters) to
`results/mc_results.csv`. The off-policy agent reports both its weighted-IS
and ordinary-IS estimators.

## Single run and GUI demo

`train_VI.py` rolls out one agent on one grid and renders the visited path. It
supports value iteration and the `random` / `null` baselines (Q-learning and
the Monte-Carlo agents are episodic and use the dedicated scripts above).

```bash
# watch value iteration solve A1_grid in the GUI
python train_VI.py grid_configs/A1_grid.npy --agent vi --start_pos 1,12 --sigma 0.02

# same, without the GUI (much faster)
python train_VI.py grid_configs/A1_grid.npy --agent vi --start_pos 1,12 --sigma 0.02 --no_gui
```

## Creating grids

```bash
python world/grid_creator.py    # interactive web editor at http://127.0.0.1:5000
python create_custom_grid.py    # the 8x8 bottleneck grid used by VI experiment 4
```

## Environment conventions

- **Grid array.** A grid is a NumPy array of shape `(n_cols, n_rows)` indexed
  `grid[col, row]`; agent positions are `(col, row)` tuples.
- **Cell codes.** `0` empty, `1` wall, `2` obstacle, `3` target, `4` start.
- **Actions.** `0` down, `1` up, `2` left, `3` right.
- **Stochasticity.** With probability `1 - sigma` the chosen action runs;
  otherwise a uniformly random action runs instead.
- **Rewards.** `-1` per step, `-5` for hitting a wall or obstacle, `+10` for reaching the target. An episode ends when the target is
  reached or the step limit is hit.
