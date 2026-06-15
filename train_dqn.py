"""Train the DQN baseline on the continuous-observation delivery task.

DQN is the primary benchmark against which the secondary Deep RL methods
(Dueling DQN, PPO, etc.) will be compared.

Examples
--------
    # quick run on a single grid
    python train_dqn.py grid_configs/small_grid.npy --episodes 400

    # all main grids, 3 seeds, custom reward
    python train_dqn.py grid_configs/small_grid.npy grid_configs/A1_grid.npy 
        grid_configs/large_grid.npy --episodes 1000 --seeds 0 1 2 
        --step_penalty -1 --collision_penalty -5 --target_reward 10

Outputs (per seed) land in ``results/dqn/<grid>/seed<k>/``:
    * ``train_log.csv``       -- per-episode return, length, success, epsilon
    * ``learning_curve.png``
    * ``model.pt``
    * ``eval_path.png``       -- a greedy rollout rendered on the grid
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


# ------------------------------------------------------------------ reward fn
def make_reward_fn(step_penalty: float,
                   collision_penalty: float,
                   target_reward: float):
    """Return a reward function with tunable magnitudes.

    The signature matches what ``Environment`` expects:
        fn(grid: np.ndarray, agent_pos: tuple[int, int]) -> float
    """
    def reward_fn(grid, agent_pos):
        match grid[agent_pos]:
            case 0:
                return step_penalty
            case 1 | 2:
                return collision_penalty
            case 3:
                return target_reward
            case _:
                raise ValueError(
                    f"Unexpected grid value {grid[agent_pos]} at {agent_pos}")
    return reward_fn


# ----------------------------------------------------------------- curriculum
class Curriculum:
    """Distance-banded curriculum.

    Starts the agent within ``radius`` Manhattan steps of the target and widens
    the band by ``step`` whenever the rolling success rate over the last
    ``window`` episodes exceeds ``threshold``. Disabled -> always full range.
    """

    def __init__(self, enabled: bool, start: int = 4, step: int = 2,
                 max_radius: int = 9999, window: int = 30,
                 threshold: float = 0.8):
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

    reward_fn = make_reward_fn(args.step_penalty,
                               args.collision_penalty,
                               args.target_reward)

    env = ContinuousEnvironment(
        grid_fp=grid_fp,
        sigma=args.sigma,
        max_steps=args.max_steps,
        frame_stack=args.frame_stack,
        use_orientation=not args.no_orientation,
        progress_reward_weight=args.progress_reward,
        obstacle_proximity_weight=args.obstacle_penalty,
        obstacle_proximity_threshold=args.obstacle_threshold,
        progress_normalize=args.progress_normalize,
        random_seed=seed,
        reward_fn=reward_fn,
    )

    # eps_decay_steps: if not set explicitly, decay over 70% of the total
    # expected env steps so epsilon remains meaningful while the curriculum
    # is still expanding.
    eps_decay_steps = args.eps_decay_steps
    if eps_decay_steps is None:
        eps_decay_steps = max(1000, int(args.episodes * args.max_steps * 0.7))

    # Evaluation uses starts at least 60% of the grid's max Manhattan distance
    # from the target, forcing the agent to navigate from the far side.
    eval_min_dist = args.eval_min_dist
    if eval_min_dist is None:
        eval_min_dist = max(1, int(env.max_manhattan_from_target * 0.6))

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
        eps_decay_steps=eps_decay_steps,
        double_dqn=args.double_dqn,
        device=args.device,
        seed=seed,
    )
    curriculum = Curriculum(enabled=not args.no_curriculum,
                            start=args.curriculum_start,
                            max_radius=env.max_manhattan_from_target)

    log_rows = []
    recent_success: deque[float] = deque(maxlen=50)
    recent_return: deque[float] = deque(maxlen=50)

    print(f"\n[{grid_fp.stem} | seed {seed}] "
          f"obs={env.obs_size} frame_stack={args.frame_stack} "
          f"orientation={'on' if not args.no_orientation else 'off'} "
          f"progress={args.progress_reward} obstacle_pen={args.obstacle_penalty} "
          f"reward=({args.step_penalty}/{args.collision_penalty}/+{args.target_reward}) "
          f"eps_decay={eps_decay_steps} "
          f"eval_min_dist={eval_min_dist}/{env.max_manhattan_from_target}")

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
            print(f"  ep {ep + 1:4d}/{args.episodes} | "
                  f"return {np.mean(recent_return):7.2f} | "
                  f"success {np.mean(recent_success):4.2f} | "
                  f"eps {agent.epsilon():.3f} | "
                  f"radius {curriculum.current_radius()}")

    # ---- persist artifacts -------------------------------------------------
    _write_csv(out_dir / "train_log.csv", log_rows)
    _plot_learning_curve(log_rows, out_dir / "learning_curve.png",
                         grid_fp.stem, seed)
    agent.save(out_dir / "model.pt")

    # ---- greedy evaluation (far starts) ------------------------------------
    eval_stats = evaluate(env, agent, episodes=args.eval_episodes,
                          min_start_dist=eval_min_dist)
    _render_greedy_path(env, agent, out_dir / "eval_path.png",
                        min_start_dist=eval_min_dist)

    summary = {
        "seed": seed,
        "grid": grid_fp.stem,
        "frame_stack": args.frame_stack,
        "step_penalty": args.step_penalty,
        "collision_penalty": args.collision_penalty,
        "target_reward": args.target_reward,
        "eval_min_dist": eval_min_dist,
        "final_train_success": float(np.mean(recent_success)),
        "eval_success_rate": eval_stats["success_rate"],
        "eval_avg_steps": eval_stats["avg_steps_on_success"],
        "eval_avg_return": eval_stats["avg_return"],
    }
    print(f"  -> EVAL success {summary['eval_success_rate']:.2f} | "
          f"avg steps {summary['eval_avg_steps']:.1f} "
          f"(min_dist>={eval_min_dist})")
    return summary


def evaluate(env: ContinuousEnvironment, agent: DQNAgent,
             episodes: int, min_start_dist: int | None = None) -> dict:
    """Greedy rollouts from far-from-target starts.

    Args:
        min_start_dist: Minimum Manhattan distance from the target for the
            sampled start position. Defaults to None (any empty cell).
    """
    successes, returns, success_steps = 0, [], []
    for _ in range(episodes):
        obs = env.reset(curriculum_radius=None, min_start_dist=min_start_dist)
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
        "avg_steps_on_success": (float(np.mean(success_steps))
                                 if success_steps else float("nan")),
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


def _plot_learning_curve(rows: list[dict], path: Path,
                         grid_name: str, seed: int):
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


def _render_greedy_path(env: ContinuousEnvironment, agent: DQNAgent,
                        path: Path, min_start_dist: int | None = None):
    """Roll out the greedy policy once from a far start and save the path."""
    env.reset(curriculum_radius=None, min_start_dist=min_start_dist)
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
    p.add_argument("GRID", type=Path, nargs="+",
                   help="One or more .npy grid files to train on sequentially.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="One or more random seeds (runs are independent).")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--max_steps", type=int, default=200,
                   help="Episode step budget (truncation horizon).")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Environment transition stochasticity.")
    p.add_argument("--frame_stack", type=int, default=1,
                   help="Number of consecutive observations stacked.")
    p.add_argument("--no_orientation", action="store_true",
                   help="Disable heading feature; rely on frame_stack instead.")

    # Tunable reward
    p.add_argument("--step_penalty", type=float, default=-1.0,
                   help="Reward for each step taken (negative = penalty).")
    p.add_argument("--collision_penalty", type=float, default=-5.0,
                   help="Reward for hitting a wall or obstacle.")
    p.add_argument("--target_reward", type=float, default=10.0,
                   help="Reward for reaching the target.")

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
    p.add_argument("--progress_reward", type=float, default=0.5,
                   help="Weight for Manhattan-progress shaping bonus (0=off).")
    p.add_argument("--obstacle_penalty", type=float, default=0.0,
                   help="Weight for obstacle-proximity penalty (0=off).")
    p.add_argument("--obstacle_threshold", type=float, default=0.15,
                   help="Normalised ray distance below which proximity penalty fires.")
    p.add_argument("--eps_decay_steps", type=int, default=None,
                   help="Env steps over which epsilon anneals. "
                        "Defaults to 70%% of episodes*max_steps.")
    p.add_argument("--double_dqn", action="store_true",
                   help="Use the Double-DQN target (for ablations).")

    # curriculum
    p.add_argument("--no_curriculum", action="store_true")
    p.add_argument("--curriculum_start", type=int, default=4)

    # evaluation
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--eval_min_dist", type=int, default=None,
                   help="Minimum Manhattan distance from target for eval starts. "
                        "Defaults to 60%% of the grid's maximum distance.")

    # bookkeeping
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", type=str, default=None,
                   help="'cpu', 'cuda', or omit to auto-select.")
    p.add_argument("--out", type=Path, default=Path("results/dqn"))

    # Optional progress normalization for reward shaping
    p.add_argument("--progress_normalize", action="store_true",
                   help="Option 2: normalise progress reward by grid scale.")
    return p.parse_args()


def main():
    args = parse_args()
    all_summaries = []

    for grid_fp in args.GRID:
        if not grid_fp.exists():
            raise FileNotFoundError(
                f"Grid file does not exist: {grid_fp}. "
                "Pass one or more real .npy grid paths instead of '...'."
            )

        grid_out = args.out / grid_fp.stem
        grid_out.mkdir(parents=True, exist_ok=True)

        seed_summaries = []
        for seed in args.seeds:
            seed_dir = grid_out / f"seed{seed}"
            seed_summaries.append(
                train_one_seed(grid_fp, seed, args, seed_dir))

        _write_csv(grid_out / "summary.csv", seed_summaries)
        all_summaries.extend(seed_summaries)

        if len(seed_summaries) > 1:
            succ = np.array([s["eval_success_rate"] for s in seed_summaries])
            print(f"\n  [{grid_fp.stem}] eval success "
                  f"{succ.mean():.3f} ± {succ.std():.3f} "
                  f"over {len(args.seeds)} seeds")

    _write_csv(args.out / "all_grids_summary.csv", all_summaries)
    print(f"\nAll artifacts written to {args.out}")


if __name__ == "__main__":
    main()
