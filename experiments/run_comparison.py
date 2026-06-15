"""Unified multi-seed runner: train DQN and Dueling DQN under ONE protocol.

Both agents share: the same environment + reward-shaping config, the same grids,
the same seeds, the same normalized hyper-parameters, the same epsilon annealing
(calibrated so both reach ``eps_end`` over the same fraction of training), and the
same deterministic evaluation start set. The ONLY intended difference is the
network architecture (plain MLP Q vs duelling value/advantage streams), so any
performance gap is attributable to that.

Outputs two tidy long-format CSVs under ``--out-dir``:

* ``train_metrics.csv`` -- one row per (algo, grid, seed, episode).
* ``eval_metrics.csv``  -- one row per (algo, grid, seed, eval-checkpoint).

Example
-------
    # quick smoke test
    python -m experiments.run_comparison --grids grid_configs/small_grid.npy \\
        --seeds 0 --episodes 50 --eval-every 25

    # full comparison on the deceptive grid + standard grids, 3 seeds
    python -m experiments.run_comparison \\
        --grids grid_configs/deceptive_grid.npy grid_configs/small_grid.npy \\
                grid_configs/A1_grid.npy grid_configs/large_grid.npy \\
        --seeds 0 1 2 --episodes 600
"""
from __future__ import annotations

import csv
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from world.continuous_env import ContinuousEnvironment
from experiments.adapters import build_adapter, available_algos
from experiments.eval_protocol import sample_eval_starts, evaluate


TRAIN_FIELDS = ["algo", "grid", "seed", "episode", "global_step",
                "episode_return", "episode_length", "success", "epsilon", "mean_loss"]
EVAL_FIELDS = ["algo", "grid", "seed", "episode", "global_step", "n_eval",
               "success_rate", "mean_return", "mean_steps_to_goal", "eval_kind"]


class CsvLogger:
    """Append-as-you-go CSV writer with a fixed header."""

    def __init__(self, path: Path, fields: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = fields
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fields)
        self._w.writeheader()

    def write(self, **row):
        self._w.writerow({k: row.get(k, "") for k in self.fields})
        self._f.flush()

    def close(self):
        self._f.close()


def build_hp(args, episodes: int, max_steps: int) -> dict:
    """Normalized hyper-parameters + per-run epsilon schedule (shared by both algos)."""
    frac = args.eps_decay_frac
    eps_decay_steps = max(1, int(frac * episodes * max_steps))
    # multiplicative per-episode decay reaching eps_end after frac*episodes episodes
    n_decay_eps = max(1.0, frac * episodes)
    eps_decay_mult = (args.eps_end / args.eps_start) ** (1.0 / n_decay_eps)
    return {
        "lr": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "learning_starts": args.learning_starts,
        "target_update": args.target_update,
        "double_dqn": args.double_dqn,
        "train_freq": args.train_freq,
        "max_grad_norm": args.max_grad_norm,
        "state_scale": args.state_scale,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_decay_steps": eps_decay_steps,
        "eps_decay_mult": eps_decay_mult,
        "device": args.device,
    }


def make_env(grid_fp: Path, sigma: float, seed: int, args) -> ContinuousEnvironment:
    return ContinuousEnvironment(
        grid_fp=grid_fp,
        n_rays=args.n_rays,
        sigma=sigma,
        max_steps=args.max_steps,
        use_orientation=not args.no_orientation,
        progress_reward_weight=args.progress_reward,
        obstacle_proximity_weight=args.obstacle_penalty,
        obstacle_proximity_threshold=args.obstacle_threshold,
        random_seed=seed,
    )


def run_one(algo: str, grid_fp: Path, seed: int, args, hp: dict,
            train_log: CsvLogger, eval_log: CsvLogger, models_dir: Path):
    grid_name = grid_fp.stem
    env = make_env(grid_fp, args.sigma, seed, args)
    eval_env = make_env(grid_fp, args.eval_sigma, seed + 9973, args)
    eval_starts = sample_eval_starts(grid_fp, args.eval_episodes,
                                     min_start_dist=args.eval_min_dist)

    adapter = build_adapter(algo, env.obs_size, env.n_actions, seed, hp)

    print(f"\n[{algo} | {grid_name} | seed {seed}] obs={env.obs_size} "
          f"episodes={args.episodes} eval_starts={len(eval_starts)} "
          f"(min_dist={args.eval_min_dist})")

    global_step = 0
    t0 = time.time()
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return, ep_len, losses, reached = 0.0, 0, [], False
        info = {}
        while not done:
            action = adapter.select_action(obs, greedy=False)
            next_obs, reward, done, info = env.step(action)
            terminal = bool(info["target_reached"])     # truncation must NOT zero bootstrap
            loss = adapter.on_transition(obs, action, reward, next_obs, terminal)
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            ep_return += reward
            ep_len += 1
            global_step += 1
        adapter.on_episode_end()
        reached = bool(info.get("target_reached", False))

        train_log.write(algo=algo, grid=grid_name, seed=seed, episode=ep + 1,
                        global_step=global_step, episode_return=round(ep_return, 4),
                        episode_length=ep_len, success=int(reached),
                        epsilon=round(adapter.epsilon(), 4),
                        mean_loss=(round(float(np.mean(losses)), 6) if losses else ""))

        is_last = ep == args.episodes - 1
        if (ep + 1) % args.eval_every == 0 or is_last:
            m = evaluate(adapter, eval_env, eval_starts, args.max_steps)
            eval_log.write(algo=algo, grid=grid_name, seed=seed, episode=ep + 1,
                           global_step=global_step, n_eval=m["n_eval"],
                           success_rate=round(m["success_rate"], 4),
                           mean_return=round(m["mean_return"], 4),
                           mean_steps_to_goal=("" if np.isnan(m["mean_steps_to_goal"])
                                               else round(m["mean_steps_to_goal"], 2)),
                           eval_kind="final" if is_last else "periodic")
            if is_last or (ep + 1) % (args.eval_every * 4) == 0:
                print(f"  ep {ep + 1:4d}/{args.episodes} | eval_success "
                      f"{m['success_rate']:.2f} | eps {adapter.epsilon():.3f} | "
                      f"{time.time() - t0:5.1f}s")

    models_dir.mkdir(parents=True, exist_ok=True)
    adapter.save(models_dir / f"{algo}_{grid_name}_seed{seed}.pt")


def parse_args():
    p = ArgumentParser(description="Unified DQN vs Dueling DQN comparison runner.")
    p.add_argument("--grids", type=Path, nargs="+",
                   default=[Path("grid_configs/deceptive_grid.npy")])
    p.add_argument("--algos", nargs="+", default=available_algos(),
                   choices=available_algos())
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=200)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=25)
    p.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=20,
                   help="Size of the fixed deterministic eval start set per grid.")
    p.add_argument("--eval-min-dist", dest="eval_min_dist", type=int, default=None,
                   help="Min Manhattan distance from target for eval starts.")
    # environment / shaping (applied IDENTICALLY to both algos)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--eval-sigma", dest="eval_sigma", type=float, default=0.0)
    p.add_argument("--n-rays", dest="n_rays", type=int, default=8)
    p.add_argument("--no-orientation", dest="no_orientation", action="store_true")
    p.add_argument("--progress-reward", dest="progress_reward", type=float, default=0.0)
    p.add_argument("--obstacle-penalty", dest="obstacle_penalty", type=float, default=0.0)
    p.add_argument("--obstacle-threshold", dest="obstacle_threshold", type=float, default=0.15)
    # normalized hyper-parameters (shared)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", dest="buffer_size", type=int, default=50_000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=128)
    p.add_argument("--learning-starts", dest="learning_starts", type=int, default=1_000)
    p.add_argument("--target-update", dest="target_update", type=int, default=500)
    p.add_argument("--double-dqn", dest="double_dqn", action="store_true",
                   help="Use Double-DQN target for BOTH agents (ablation).")
    p.add_argument("--train-freq", dest="train_freq", type=int, default=1)
    p.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=10.0)
    p.add_argument("--state-scale", dest="state_scale", type=float, default=1.0)
    p.add_argument("--eps-start", dest="eps_start", type=float, default=1.0)
    p.add_argument("--eps-end", dest="eps_end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", dest="eps_decay_frac", type=float, default=0.7,
                   help="Fraction of training over which epsilon anneals to eps_end.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", dest="out_dir", type=Path,
                   default=Path("experiments/results"))
    return p.parse_args()


def main():
    args = parse_args()
    hp = build_hp(args, args.episodes, args.max_steps)

    out_dir = args.out_dir
    train_log = CsvLogger(out_dir / "train_metrics.csv", TRAIN_FIELDS)
    eval_log = CsvLogger(out_dir / "eval_metrics.csv", EVAL_FIELDS)
    models_dir = out_dir / "models"

    print(f"Algos: {args.algos} | grids: {[g.stem for g in args.grids]} | "
          f"seeds: {args.seeds} | episodes: {args.episodes}")
    print(f"Shared hp: lr={hp['lr']} gamma={hp['gamma']} hidden={hp['hidden_dim']} "
          f"double_dqn={hp['double_dqn']} eps {hp['eps_start']}->{hp['eps_end']} "
          f"(decay_steps={hp['eps_decay_steps']}, mult={hp['eps_decay_mult']:.5f})")

    try:
        for grid_fp in args.grids:
            for seed in args.seeds:
                for algo in args.algos:
                    run_one(algo, grid_fp, seed, args, hp, train_log, eval_log, models_dir)
    finally:
        train_log.close()
        eval_log.close()

    print(f"\nDone. Wrote:\n  {out_dir / 'train_metrics.csv'}\n  {out_dir / 'eval_metrics.csv'}")


if __name__ == "__main__":
    main()
