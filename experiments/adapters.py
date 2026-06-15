"""Agent adapters: drive DQN and Dueling DQN through ONE interface.

The two agents expose different APIs:

* ``DQNAgent``        -- ``act(obs, greedy)`` / ``observe(...)`` + ``learn()`` per step /
                         step-based linear epsilon via ``epsilon()``.
* ``DuelingDQNAgent`` -- ``take_action(state)`` / ``update(next_state, r, a, terminated)`` per step /
                         per-episode multiplicative epsilon via ``decay_epsilon()``.

``AgentAdapter`` normalises action selection, transition consumption and epsilon
scheduling so a single training loop (``run_comparison.py``) is agent-agnostic.

It also normalises **hyper-parameters**: both adapters receive the SAME normalized
``hp`` dict (lr, gamma, buffer_size, batch_size, hidden_dim, target_update,
learning_starts, double_dqn, train_freq, max_grad_norm, eps_start, eps_end, and the
per-run epsilon-schedule values) and translate the common keys to each agent's own
parameter names. This is what makes the DQN-vs-Dueling comparison apples-to-apples.

Torch-dependent agents are imported lazily inside the adapters so that the rest of
the ``experiments`` package (grid tools, metrics, plots) imports without torch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AgentAdapter(ABC):
    """Uniform interface over a single RL agent."""

    name: str = "agent"

    @abstractmethod
    def select_action(self, obs, *, greedy: bool) -> int:
        """Return an action. ``greedy=False`` explores (training); ``True`` exploits (eval)."""

    @abstractmethod
    def on_transition(self, obs, action, reward, next_obs, terminal) -> float | None:
        """Feed one env transition; return a scalar loss or ``None`` (no update)."""

    @abstractmethod
    def on_episode_end(self) -> None:
        """Per-episode hook (epsilon scheduling for agents that decay per episode)."""

    @abstractmethod
    def epsilon(self) -> float:
        """Current exploration rate (for logging)."""

    @abstractmethod
    def set_eval_mode(self, eval_: bool) -> None:
        """Switch the underlying agent between train/eval mode where applicable."""

    @abstractmethod
    def save(self, path) -> None:
        """Persist the model."""


class DQNAdapter(AgentAdapter):
    """Wraps ``agents.dqn_agent.DQNAgent`` (the baseline)."""

    name = "DQN"

    def __init__(self, obs_size: int, n_actions: int, seed: int, hp: dict):
        from agents.dqn_agent import DQNAgent

        h = int(hp.get("hidden_dim", 128))
        self.agent = DQNAgent(
            obs_size=obs_size,
            n_actions=n_actions,
            hidden=(h, h),
            lr=hp.get("lr", 1e-3),
            gamma=hp.get("gamma", 0.99),
            buffer_size=hp.get("buffer_size", 50_000),
            batch_size=hp.get("batch_size", 64),
            learning_starts=hp.get("learning_starts", 1_000),
            target_sync=hp.get("target_update", 500),
            train_freq=hp.get("train_freq", 1),
            eps_start=hp.get("eps_start", 1.0),
            eps_end=hp.get("eps_end", 0.05),
            eps_decay_steps=int(hp["eps_decay_steps"]),
            max_grad_norm=hp.get("max_grad_norm", 10.0),
            double_dqn=hp.get("double_dqn", False),
            device=hp.get("device"),
            seed=seed,
        )

    def select_action(self, obs, *, greedy: bool) -> int:
        return self.agent.act(obs, greedy=greedy)

    def on_transition(self, obs, action, reward, next_obs, terminal) -> float | None:
        self.agent.observe(obs, action, reward, next_obs, terminal)
        return self.agent.learn()

    def on_episode_end(self) -> None:
        pass  # DQN epsilon anneals by env-step inside observe()

    def epsilon(self) -> float:
        return float(self.agent.epsilon())

    def set_eval_mode(self, eval_: bool) -> None:
        pass  # greedy handled via select_action(greedy=True)

    def save(self, path) -> None:
        self.agent.save(Path(path))


class DuelingAdapter(AgentAdapter):
    """Wraps ``agents.DuelingDQN_agentv2.DuelingDQNAgent`` (the main method)."""

    name = "DuelingDQN"

    def __init__(self, obs_size: int, n_actions: int, seed: int, hp: dict):
        from agents.DuelingDQN_agentv2 import DuelingDQNAgent

        self.agent = DuelingDQNAgent(
            learning_rate=hp.get("lr", 1e-3),
            gamma=hp.get("gamma", 0.99),
            epsilon=hp.get("eps_start", 1.0),
            epsilon_decay=float(hp["eps_decay_mult"]),
            min_epsilon=hp.get("eps_end", 0.05),
            n_actions=n_actions,
            state_dim=obs_size,                 # CRITICAL: default is 2, must be obs_size
            hidden_dim=int(hp.get("hidden_dim", 128)),
            buffer_size=hp.get("buffer_size", 50_000),
            batch_size=hp.get("batch_size", 64),
            train_start=hp.get("learning_starts", 1_000),
            target_update_freq=hp.get("target_update", 500),
            state_scale=hp.get("state_scale", 1.0),
            double_dqn=hp.get("double_dqn", False),
            train_freq=hp.get("train_freq", 1),
            max_grad_norm=hp.get("max_grad_norm", 10.0),
            seed=seed,
            device=hp.get("device"),
        )
        self.agent.train_mode()

    def select_action(self, obs, *, greedy: bool) -> int:
        # take_action() stores last_state/last_action that update() consumes next.
        if greedy:
            self.agent.eval_mode()
        else:
            self.agent.train_mode()
        return self.agent.take_action(obs)

    def on_transition(self, obs, action, reward, next_obs, terminal) -> float | None:
        # update() treats its first arg as the NEXT state (old_state = last_state).
        return self.agent.update(next_obs, reward, action, terminated=terminal)

    def on_episode_end(self) -> None:
        self.agent.decay_epsilon()

    def epsilon(self) -> float:
        return float(getattr(self.agent, "epsilon", float("nan")))

    def set_eval_mode(self, eval_: bool) -> None:
        self.agent.eval_mode() if eval_ else self.agent.train_mode()

    def save(self, path) -> None:
        self.agent.save_model(Path(path))


_ADAPTERS = {"DQN": DQNAdapter, "DuelingDQN": DuelingAdapter}


def build_adapter(algo: str, obs_size: int, n_actions: int, seed: int, hp: dict) -> AgentAdapter:
    """Factory: ``algo`` in {"DQN", "DuelingDQN"}."""
    if algo not in _ADAPTERS:
        raise ValueError(f"Unknown algo {algo!r}; expected one of {list(_ADAPTERS)}.")
    return _ADAPTERS[algo](obs_size, n_actions, seed, hp)


def available_algos() -> list[str]:
    return list(_ADAPTERS)
