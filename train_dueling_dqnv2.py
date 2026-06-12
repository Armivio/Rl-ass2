from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from world.continuous_env import ContinuousEnvironment
from agents.DuelingDQN_agentv2 import DuelingDQNAgent


def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Dueling DQN v2 Trainer.")

    p.add_argument(
        "GRID",
        type=Path,
        nargs="+",
        help="Paths to one or more grid files.",
    )

    p.add_argument("--no_gui", action="store_true", help="Disables rendering to train faster.")

    p.add_argument("--gammas", type=float, nargs="+", default=[0.9], help="Gamma values to test.")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.1], help="Sigma values to test.")
    p.add_argument("--epsilons", type=float, nargs="+", default=[1.0], help="Initial epsilon values to test.")
    p.add_argument("--max_steps", type=int, nargs="+", default=[500], help="Maximum steps per episode.")

    p.add_argument("--fps", type=int, default=30, help="Rendering FPS. Only used if --no_gui is not set.")
    p.add_argument("--iter", type=int, default=3000, help="Number of training episodes.")
    p.add_argument("--eval_iter", type=int, default=100, help="Number of evaluation episodes.")

    p.add_argument("--alpha", type=float, default=1e-3, help="DQN learning rate.")

    p.add_argument("--epsilon_decay", type=float, default=0.995, help="Multiplicative epsilon decay per episode.")
    p.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum epsilon after decay.")

    p.add_argument("--batch_size", type=int, default=64, help="Replay minibatch size.")
    p.add_argument("--buffer_size", type=int, default=50_000, help="Replay buffer size.")
    p.add_argument("--train_start", type=int, default=500, help="Number of transitions before gradient updates start.")
    p.add_argument("--target_update_freq", type=int, default=200, help="Hard target-network update frequency in gradient steps.")
    p.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer width.")
    p.add_argument("--state_scale", type=float, default=None, help="State normalization scale. If omitted, inferred from grid shape when possible.")
    p.add_argument("--disable_double_dqn", action="store_true", help="Use vanilla DQN target instead of Double DQN target.")

    p.add_argument("--train_freq", type=int, default=1, help="Run one gradient update every N environment steps.")
    p.add_argument("--max_grad_norm", type=float, default=10.0, help="Gradient clipping norm.")

    p.add_argument("--random_seed", type=int, default=0, help="Random seed value.")
    p.add_argument(
        "--start_pos",
        type=str,
        default=None,
        help="Agent start position as col,row, for example 1,12. If omitted, environment default is used.",
    )

    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/dueling_dqn_v2"),
        help="Folder to save results.",
    )

    return p.parse_args()


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_state_scale(grid_path: Path) -> float:
    try:
        grid = np.load(grid_path)
        return float(max(grid.shape[0], grid.shape[1], 1))
    except Exception:
        return 20.0


def make_name(value):
    return str(value).replace(".", "p").replace("/", "_").replace("\\", "_")


def make_run_name(setup_id, grid, gamma, sigma, epsilon, max_steps):
    return (
        f"setup_{setup_id:02d}"
        f"_{grid.stem}"
        f"_gamma_{make_name(gamma)}"
        f"_sigma_{make_name(sigma)}"
        f"_eps_{make_name(epsilon)}"
        f"_steps_{make_name(max_steps)}"
    )


def rolling_mean(values, window=100):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i + 1])))
    return result


def save_training_log(path, logs):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "setup_id",
        "episode",
        "discounted_return",
        "episode_length",
        "success",
        "epsilon",
        "mean_loss",
        "rolling_return",
        "rolling_success_rate",
        "rolling_episode_length",
        "rolling_loss",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)


def save_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "setup_id",
        "grid",
        "gamma",
        "sigma",
        "epsilon_initial",
        "epsilon_final",
        "epsilon_decay",
        "min_epsilon",
        "alpha",
        "batch_size",
        "buffer_size",
        "train_start",
        "target_update_freq",
        "hidden_dim",
        "state_scale",
        "double_dqn",
        "train_freq",
        "max_grad_norm",
        "max_steps",
        "train_episodes",
        "eval_episodes",
        "train_final_success_rate",
        "train_final_avg_discounted_return",
        "train_final_avg_episode_length",
        "eval_success_rate",
        "eval_avg_discounted_return",
        "eval_avg_episode_length",
        "setup_folder",
        "training_log_path",
        "model_path",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_convergence(setup_dir, logs, title):
    episodes = [row["episode"] for row in logs]
    rolling_returns = [row["rolling_return"] for row in logs]
    rolling_success = [row["rolling_success_rate"] for row in logs]
    rolling_lengths = [row["rolling_episode_length"] for row in logs]
    rolling_losses = [row["rolling_loss"] for row in logs]

    setup_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(episodes, rolling_returns)
    plt.xlabel("Episode")
    plt.ylabel("Rolling average discounted return")
    plt.title(title + " return")
    plt.tight_layout()
    plt.savefig(setup_dir / "convergence_return.png")
    plt.close()

    plt.figure()
    plt.plot(episodes, rolling_success)
    plt.xlabel("Episode")
    plt.ylabel("Rolling success rate")
    plt.title(title + " success rate")
    plt.tight_layout()
    plt.savefig(setup_dir / "convergence_success.png")
    plt.close()

    plt.figure()
    plt.plot(episodes, rolling_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Rolling average episode length")
    plt.title(title + " episode length")
    plt.tight_layout()
    plt.savefig(setup_dir / "convergence_length.png")
    plt.close()

    plt.figure()
    plt.plot(episodes, rolling_losses)
    plt.xlabel("Episode")
    plt.ylabel("Rolling average Huber loss")
    plt.title(title + " training loss")
    plt.tight_layout()
    plt.savefig(setup_dir / "convergence_loss.png")
    plt.close()


def plot_summary(output_dir, rows):
    if len(rows) == 0:
        return

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    setup_ids = [row["setup_id"] for row in rows]
    success_rates = [row["eval_success_rate"] for row in rows]
    returns = [row["eval_avg_discounted_return"] for row in rows]
    lengths = [row["eval_avg_episode_length"] for row in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(setup_ids, success_rates)
    plt.xlabel("Setup")
    plt.ylabel("Final success rate")
    plt.title("Dueling DQN v2 final success rate")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(summary_dir / "dueling_dqn_v2_success_rate_all_setups.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(setup_ids, returns)
    plt.xlabel("Setup")
    plt.ylabel("Average discounted return")
    plt.title("Dueling DQN v2 average discounted return")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(summary_dir / "dueling_dqn_v2_discounted_return_all_setups.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(setup_ids, lengths)
    plt.xlabel("Setup")
    plt.ylabel("Average episode length")
    plt.title("Dueling DQN v2 average episode length")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(summary_dir / "dueling_dqn_v2_episode_length_all_setups.png")
    plt.close()


def evaluate_agent(grid, agent, eval_iter, max_steps, sigma, gamma, random_seed, start_pos):
    agent.eval_mode()

    env = ContinuousEnvironment(
        grid_fp=grid,
        sigma=sigma,
        max_steps=max_steps,
        random_seed=random_seed,
    )

    returns = []
    lengths = []
    successes = []

    for _ in range(eval_iter):
        state = env.reset(agent_start_pos=start_pos) if start_pos is not None else env.reset()
        agent.reset_episode()

        discounted_return = 0.0
        episode_length = max_steps
        success = 0

        for step in range(max_steps):
            action = agent.take_action(state)
            state, reward, done, info = env.step(action)
            discounted_return += (gamma ** step) * reward

            if done:
                episode_length = step + 1
                success = int(info["target_reached"])
                break

        returns.append(discounted_return)
        lengths.append(episode_length)
        successes.append(success)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_discounted_return": float(np.mean(returns)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def train_one_setup(
    setup_id,
    grid,
    no_gui,
    iters,
    eval_iter,
    fps,
    gamma,
    sigma,
    epsilon,
    max_steps,
    alpha,
    epsilon_decay,
    min_epsilon,
    batch_size,
    buffer_size,
    train_start,
    target_update_freq,
    hidden_dim,
    state_scale,
    double_dqn,
    train_freq,
    max_grad_norm,
    random_seed,
    start_pos,
    output_dir,
):
    set_global_seeds(random_seed)

    run_name = make_run_name(setup_id, grid, gamma, sigma, epsilon, max_steps)
    setup_dir = output_dir / run_name
    setup_dir.mkdir(parents=True, exist_ok=True)

    env = ContinuousEnvironment(
        grid_fp=grid,
        sigma=sigma,
        max_steps=max_steps,
        random_seed=random_seed,
    )

    state_scale_for_run = 1.0 if state_scale is None else float(state_scale)

    print("\nRunning setup", setup_id)
    print("Grid:", grid)
    print("Gamma:", gamma)
    print("Sigma:", sigma)
    print("Initial epsilon:", epsilon)
    print("Max steps:", max_steps)
    print("Learning rate:", alpha)
    print("Batch size:", batch_size)
    print("Train start:", train_start)
    print("Target update freq:", target_update_freq)
    print("Observation size:", env.obs_size)
    print("Train frequency:", train_freq)
    print("Max grad norm:", max_grad_norm)
    print("Double DQN target:", double_dqn)
    print("Setup folder:", setup_dir)

    agent = DuelingDQNAgent(
        learning_rate=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        n_actions=env.n_actions,
        state_dim=env.obs_size,
        hidden_dim=hidden_dim,
        buffer_size=buffer_size,
        batch_size=batch_size,
        train_start=train_start,
        target_update_freq=target_update_freq,
        state_scale=state_scale_for_run,
        double_dqn=double_dqn,
        train_freq=train_freq,
        max_grad_norm=max_grad_norm,
        seed=random_seed,
    )

    agent.train_mode()

    episode_returns = []
    episode_lengths = []
    episode_successes = []
    episode_mean_losses = []
    logs = []

    for episode in trange(iters):
        state = env.reset(agent_start_pos=start_pos) if start_pos is not None else env.reset()
        agent.reset_episode()

        discounted_return = 0.0
        episode_length = max_steps
        success = 0
        losses = []
        epsilon_used = agent.epsilon

        for step in range(max_steps):
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)

            loss = agent.update(next_state, reward, action, terminated=info["target_reached"])
            if loss is not None:
                losses.append(loss)

            discounted_return += (gamma ** step) * reward
            state = next_state

            if done:
                episode_length = step + 1
                success = int(info["target_reached"])
                break

        agent.decay_epsilon()

        mean_loss = float(np.mean(losses)) if losses else float("nan")

        episode_returns.append(discounted_return)
        episode_lengths.append(episode_length)
        episode_successes.append(success)
        episode_mean_losses.append(mean_loss)

        rolling_returns = rolling_mean(episode_returns, window=100)
        rolling_success = rolling_mean(episode_successes, window=100)
        rolling_lengths = rolling_mean(episode_lengths, window=100)

        valid_recent_losses = [x for x in episode_mean_losses[-100:] if not np.isnan(x)]
        rolling_loss = float(np.mean(valid_recent_losses)) if valid_recent_losses else float("nan")

        logs.append(
            {
                "setup_id": setup_id,
                "episode": episode + 1,
                "discounted_return": float(discounted_return),
                "episode_length": int(episode_length),
                "success": int(success),
                "epsilon": float(epsilon_used),
                "mean_loss": mean_loss,
                "rolling_return": rolling_returns[-1],
                "rolling_success_rate": rolling_success[-1],
                "rolling_episode_length": rolling_lengths[-1],
                "rolling_loss": rolling_loss,
            }
        )

        if (episode + 1) % 100 == 0:
            avg_return = np.mean(episode_returns[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_success = np.mean(episode_successes[-100:])

            print(
                f"Episode {episode + 1:5d} | "
                f"avg return: {avg_return:8.3f} | "
                f"success: {avg_success:6.3f} | "
                f"avg length: {avg_length:8.3f} | "
                f"epsilon: {agent.epsilon:7.4f} | "
                f"loss: {rolling_loss:9.5f}"
            )

    train_final_success_rate = float(np.mean(episode_successes[-100:]))
    train_final_avg_return = float(np.mean(episode_returns[-100:]))
    train_final_avg_length = float(np.mean(episode_lengths[-100:]))

    log_path = setup_dir / "training_log.csv"
    save_training_log(log_path, logs)

    plot_convergence(setup_dir, logs, run_name)

    model_path = setup_dir / "dueling_dqn_v2_model.pt"
    agent.save_model(model_path)

    eval_result = evaluate_agent(
        grid,
        agent,
        eval_iter,
        max_steps,
        sigma,
        gamma,
        random_seed,
        start_pos,
    )

    print("\nTraining finished for setup", setup_id)
    print("Training final success rate:", train_final_success_rate)
    print("Training final average discounted return:", train_final_avg_return)
    print("Training final average episode length:", train_final_avg_length)
    print("Evaluation success rate:", eval_result["success_rate"])
    print("Evaluation average discounted return:", eval_result["avg_discounted_return"])
    print("Evaluation average episode length:", eval_result["avg_episode_length"])
    print("Model saved to:", model_path)

    setup_summary_path = setup_dir / "summary.csv"
    setup_summary = [
        {
            "setup_id": setup_id,
            "grid": str(grid),
            "gamma": float(gamma),
            "sigma": float(sigma),
            "epsilon_initial": float(epsilon),
            "epsilon_final": float(agent.epsilon),
            "epsilon_decay": float(epsilon_decay) if epsilon_decay is not None else "None",
            "min_epsilon": float(min_epsilon),
            "alpha": float(alpha),
            "batch_size": int(batch_size),
            "buffer_size": int(buffer_size),
            "train_start": int(train_start),
            "target_update_freq": int(target_update_freq),
            "hidden_dim": int(hidden_dim),
            "state_scale": float(state_scale_for_run),
            "double_dqn": bool(double_dqn),
            "train_freq": int(train_freq),
            "max_grad_norm": float(max_grad_norm),
            "max_steps": int(max_steps),
            "train_episodes": int(iters),
            "eval_episodes": int(eval_iter),
            "train_final_success_rate": train_final_success_rate,
            "train_final_avg_discounted_return": train_final_avg_return,
            "train_final_avg_episode_length": train_final_avg_length,
            "eval_success_rate": eval_result["success_rate"],
            "eval_avg_discounted_return": eval_result["avg_discounted_return"],
            "eval_avg_episode_length": eval_result["avg_episode_length"],
            "setup_folder": str(setup_dir),
            "training_log_path": str(log_path),
            "model_path": str(model_path),
        }
    ]

    save_summary(setup_summary_path, setup_summary)
    return setup_summary[0]


def main(
    grid_paths: list[Path],
    no_gui: bool,
    iters: int,
    eval_iter: int,
    fps: int,
    gammas: list[float],
    sigmas: list[float],
    epsilons: list[float],
    max_steps_values: list[int],
    alpha: float,
    epsilon_decay: float,
    min_epsilon: float,
    batch_size: int,
    buffer_size: int,
    train_start: int,
    target_update_freq: int,
    hidden_dim: int,
    state_scale: float | None,
    double_dqn: bool,
    train_freq: int,
    max_grad_norm: float,
    random_seed: int,
    start_pos: tuple[int, int] | None,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    setup_id = 1

    for grid in grid_paths:
        for gamma in gammas:
            for sigma in sigmas:
                for epsilon in epsilons:
                    for max_steps in max_steps_values:
                        summary = train_one_setup(
                            setup_id=setup_id,
                            grid=grid,
                            no_gui=no_gui,
                            iters=iters,
                            eval_iter=eval_iter,
                            fps=fps,
                            gamma=gamma,
                            sigma=sigma,
                            epsilon=epsilon,
                            max_steps=max_steps,
                            alpha=alpha,
                            epsilon_decay=epsilon_decay,
                            min_epsilon=min_epsilon,
                            batch_size=batch_size,
                            buffer_size=buffer_size,
                            train_start=train_start,
                            target_update_freq=target_update_freq,
                            hidden_dim=hidden_dim,
                            state_scale=state_scale,
                            double_dqn=double_dqn,
                            train_freq=train_freq,
                            max_grad_norm=max_grad_norm,
                            random_seed=random_seed,
                            start_pos=start_pos,
                            output_dir=output_dir,
                        )

                        all_summaries.append(summary)
                        setup_id += 1

    summary_path = output_dir / "summary" / "dueling_dqn_v2_eval_summary.csv"
    save_summary(summary_path, all_summaries)
    plot_summary(output_dir, all_summaries)

    print("\nAll setups finished.")
    print("Summary saved to:", summary_path)
    print("Plots saved to:", output_dir)


if __name__ == "__main__":
    args = parse_args()

    start_pos = None
    if args.start_pos is not None:
        parts = args.start_pos.split(",")
        start_pos = (int(parts[0]), int(parts[1]))

    main(
        grid_paths=args.GRID,
        no_gui=args.no_gui,
        iters=args.iter,
        eval_iter=args.eval_iter,
        fps=args.fps,
        gammas=args.gammas,
        sigmas=args.sigmas,
        epsilons=args.epsilons,
        max_steps_values=args.max_steps,
        alpha=args.alpha,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_start=args.train_start,
        target_update_freq=args.target_update_freq,
        hidden_dim=args.hidden_dim,
        state_scale=args.state_scale,
        double_dqn=not args.disable_double_dqn,
        train_freq=args.train_freq,
        max_grad_norm=args.max_grad_norm,
        random_seed=args.random_seed,
        start_pos=start_pos,
        output_dir=args.output_dir,
    )