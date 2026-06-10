"""Train the DQN baseline on the continuous-observation delivery task.

Example
-------
    # quick run on the small grid (headless)
    python train_dqn.py grid_configs/small_grid.npy --episodes 400

    # A1 grid, multiple seeds, curriculum on (default)
    python train_dqn.py grid_configs/A1_grid.npy --episodes 1500 --seeds 0 1 2

Outputs (per seed) land in ``results/dqn/<grid>/seed<k>/``:
    * ``train_log.csv``  -- per-episode return, length, success, epsilon
    * ``learning_curve.png``
    * ``model.pt``
    * ``eval_path.png``  -- a greedy rollout rendered on the grid
and an aggregated ``results/dqn/<grid>/summary.csv`` across seeds.
"""
from __future__ import annotations

import csv
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import numpy as np

from world.continuous_env import ContinuousEnvironment
from world.path_visualizer import visualize_path
from agents.dqn_agent import DQNAgent


# ----------------------------------------------------------------- curriculum
class Curriculum:
    """Distance-banded curriculum.

    Starts the agent within ``radius`` Manhattan steps of the target and widens
    the band by ``step`` whenever the rolling success rate over the last
    ``window`` episodes exceeds ``threshold``. Disabled -> always full range.
    """

    def __init__(self, enabled: bool, start: int = 4, step: int = 2,
                 max_radius: int = 9999, window: int = 30, threshold: float = 0.8):
        self.enabled = enabled
        self.radius = start
        self.step = step
        self.max_radius = max_radius
        self.window = window
        self.threshold = threshold
        self._recent: deque[float] = deque(maxlen=window)

    def current_radius(self) -> int | None:
        if not self.enabled:
            return None
        return self.radius

    def record(self, success: bool):
        if not self.enabled:
            return
        self._recent.append(1.0 if success else 0.0)
        if (len(self._recent) == self.window
                and np.mean(self._recent) >= self.threshold
                and self.radius < self.max_radius):
            self.radius = min(self.radius + self.step, self.max_radius)
            self._recent.clear()


# ------------------------------------------------------------------- training
def train_one_seed(grid_fp: Path, seed: int, args, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ContinuousEnvironment(
        grid_fp=grid_fp,
        sigma=args.sigma,
        max_steps=args.max_steps,
        frame_stack=args.frame_stack,
        random_seed=seed,
    )
    agent = DQNAgent(
        obs_size=env.obs_size,
        n_actions=env.n_actions,
        hidden=tuple(args.hidden),
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        target_sync=args.target_sync,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        double_dqn=args.double_dqn,
        device=args.device,
        seed=seed,
    )
    curriculum = Curriculum(enabled=not args.no_curriculum,
                            start=args.curriculum_start,
                            max_radius=sum(env.grid_shape))

    log_rows = []
    recent_success: deque[float] = deque(maxlen=50)
    recent_return: deque[float] = deque(maxlen=50)

    for ep in range(args.episodes):
        obs = env.reset(curriculum_radius=curriculum.current_radius())
        ep_return, ep_len, reached = 0.0, 0, False

        while True:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.observe(obs, action, reward, next_obs, info["target_reached"])
            agent.learn()

            obs = next_obs
            ep_return += reward
            ep_len += 1
            if done:
                reached = info["target_reached"]
                break

        curriculum.record(reached)
        recent_success.append(1.0 if reached else 0.0)
        recent_return.append(ep_return)
        log_rows.append({
            "episode": ep,
            "return": ep_return,
            "length": ep_len,
            "success": int(reached),
            "epsilon": round(agent.epsilon(), 4),
            "curriculum_radius": curriculum.current_radius() or -1,
        })

        if (ep + 1) % args.log_every == 0:
            print(f"[seed {seed}] ep {ep + 1:4d}/{args.episodes} | "
                  f"return {np.mean(recent_return):7.2f} | "
                  f"success {np.mean(recent_success):4.2f} | "
                  f"eps {agent.epsilon():.3f} | "
                  f"radius {curriculum.current_radius()}")

    # ---- persist artifacts -------------------------------------------------
    _write_csv(out_dir / "train_log.csv", log_rows)
    _plot_learning_curve(log_rows, out_dir / "learning_curve.png", grid_fp.stem, seed)
    agent.save(out_dir / "model.pt")

    # ---- greedy evaluation -------------------------------------------------
    eval_stats = evaluate(env, agent, episodes=args.eval_episodes, seed=seed)
    _render_greedy_path(env, agent, out_dir / "eval_path.png")

    summary = {
        "seed": seed,
        "final_train_success": float(np.mean(recent_success)),
        "eval_success_rate": eval_stats["success_rate"],
        "eval_avg_steps": eval_stats["avg_steps_on_success"],
        "eval_avg_return": eval_stats["avg_return"],
    }
    print(f"[seed {seed}] EVAL success {summary['eval_success_rate']:.2f} | "
          f"avg steps {summary['eval_avg_steps']:.1f}")
    return summary


def evaluate(env: ContinuousEnvironment, agent: DQNAgent,
             episodes: int, seed: int) -> dict:
    """Greedy rollouts from random starts (full difficulty, no curriculum)."""
    successes, returns, success_steps = 0, [], []
    for _ in range(episodes):
        obs = env.reset(curriculum_radius=None)
        ep_return, ep_len = 0.0, 0
        while True:
            action = agent.act(obs, greedy=True)
            obs, reward, done, info = env.step(action)
            ep_return += reward
            ep_len += 1
            if done:
                if info["target_reached"]:
                    successes += 1
                    success_steps.append(ep_len)
                break
        returns.append(ep_return)
    return {
        "success_rate": successes / episodes,
        "avg_return": float(np.mean(returns)),
        "avg_steps_on_success": float(np.mean(success_steps)) if success_steps else float("nan"),
    }


# --------------------------------------------------------------------- output
def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _plot_learning_curve(rows: list[dict], path: Path, grid_name: str, seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = np.array([r["return"] for r in rows])
    success = np.array([r["success"] for r in rows])
    w = min(50, max(1, len(returns) // 10))

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(returns, color="tab:blue", alpha=0.25, label="return")
    ax1.plot(np.arange(w - 1, len(returns)), _moving_average(returns, w),
             color="tab:blue", label=f"return (MA{w})")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("episode return", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(np.arange(w - 1, len(success)), _moving_average(success, w),
             color="tab:green", label=f"success (MA{w})")
    ax2.set_ylabel("success rate", color="tab:green")
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(axis="y", labelcolor="tab:green")

    fig.suptitle(f"DQN on {grid_name} (seed {seed})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _render_greedy_path(env: ContinuousEnvironment, agent: DQNAgent, path: Path):
    """Roll out the greedy policy once and save the visited-path heat map."""
    env.reset(curriculum_radius=None)
    initial_grid = np.copy(env.env.grid)
    agent_path = [env.agent_pos]
    obs = env._stacked_observation()
    for _ in range(env.max_steps):
        action = agent.act(obs, greedy=True)
        obs, _, done, info = env.step(action)
        agent_path.append(info["agent_pos"])
        if done:
            break
    img = visualize_path(initial_grid, agent_path)
    img.save(path)


# ----------------------------------------------------------------------- main
def parse_args():
    p = ArgumentParser(description="DQN baseline trainer (continuous obs).")
    p.add_argument("GRID", type=Path, help="Path to the .npy grid file.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="One or more random seeds (runs are independent).")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--max_steps", type=int, default=200,
                   help="Episode step budget (truncation horizon).")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Environment transition stochasticity.")
    p.add_argument("--frame_stack", type=int, default=1,
                   help="Number of observations stacked together.")

    # DQN hyper-parameters
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_starts", type=int, default=1_000)
    p.add_argument("--target_sync", type=int, default=500)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay_steps", type=int, default=20_000)
    p.add_argument("--double_dqn", action="store_true",
                   help="Use the Double-DQN target (for ablations).")

    # curriculum
    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--curriculum_start", type=int, default=4)

    # bookkeeping
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--device", type=str, default=None,
                   help="'cpu', 'cuda', or omit to auto-select.")
    p.add_argument("--out", type=Path, default=Path("results/dqn"))
    return p.parse_args()


def main():
    args = parse_args()
    grid_fp = args.GRID
    grid_out = args.out / grid_fp.stem
    grid_out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in args.seeds:
        seed_dir = grid_out / f"seed{seed}"
        summaries.append(train_one_seed(grid_fp, seed, args, seed_dir))

    _write_csv(grid_out / "summary.csv", summaries)

    if len(summaries) > 1:
        succ = np.array([s["eval_success_rate"] for s in summaries])
        print(f"\n=== {grid_fp.stem}: eval success {succ.mean():.3f} "
              f"± {succ.std():.3f} over {len(summaries)} seeds ===")
    print(f"Artifacts written to {grid_out}")


if __name__ == "__main__":
    main()
