# Reinforcement Learning on a Grid-World

A delivery-robot grid-world solved with three families of tabular
reinforcement-learning algorithms, for the 2AMC15 Data Intelligence Challenge. 
The agent must navigate a discrete grid of walls and obstacles to
reach a target cell under a stochastic transition model, while maximising its
cumulative reward.


The continuous environment is `world/continuous_env.py`; the comparison harness,
metrics, plots and the deceptive grid are in `experiments/` (see
`experiments/README.md` for details).

## Setup (Python >= 3.10)
\`\`\`bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# torch above is the CUDA wheel; for a smaller CPU-only build:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
\`\`\`

## Quick demo
\`\`\`bash
python -m experiments.make_deceptive_grid                      # build the deceptive grid
python -m experiments.run_comparison --grids grid_configs/small_grid.npy \
    --seeds 0 --episodes 200 --out-dir experiments/results/demo
python -m experiments.metrics experiments/results/demo         # tables + significance
python -m experiments.plots   experiments/results/demo         # figures
\`\`\`

## Reproduce the report results
\`\`\`bash
python -m experiments.run_comparison \
    --grids grid_configs/small_grid.npy grid_configs/A1_grid.npy \
            grid_configs/large_grid.npy grid_configs/deceptive_grid.npy \
    --seeds 0 1 2 3 4 --episodes 800 --eval-every 50 --out-dir experiments/results/main
python -m experiments.metrics experiments/results/main
python -m experiments.plots   experiments/results/main

# Fourier ablation
python -m experiments.run_comparison --algos DuelingDQN --no-fourier \
    --grids grid_configs/A1_grid.npy grid_configs/large_grid.npy \
    --seeds 0 1 2 3 4 --episodes 800 --out-dir experiments/results/ablation
\`\`\`

All CLI flags: \`python -m experiments.run_comparison --help\`.

## Layout
- `agents/` — DQN and Dueling DQN (from scratch; only NN primitives from PyTorch)
- `world/continuous_env.py` — continuous LiDAR observation wrapper
- `experiments/` — comparison runner, metrics, plots, deceptive grid
- `grid_configs/` — grid layouts (.npy); `requirements.txt` — all dependencies
