"""Deep Q-Network (DQN) agent -- implemented from scratch.

This is the Assignment-2 **baseline**. Only the neural-network primitives
(layers, autograd, optimiser) come from PyTorch; every RL component -- the
experience replay buffer, the target network and its synchronisation, the
epsilon-greedy behaviour policy, and the Bellman target / loss -- is written
explicitly here, as required by the assignment ("implemented from scratch
without the use of existing tools").

Key DQN ingredients (Mnih et al., 2015):
  * Experience replay: decorrelates consecutive transitions.
  * Target network: a periodically-synced copy of the online network provides
    stable bootstrap targets.
  * Epsilon-greedy exploration with a linear schedule.
  * Huber (smooth-L1) loss + gradient clipping for stability.

An optional Double-DQN target (van Hasselt et al., 2016) is supported via
``double_dqn`` for later ablation studies, but defaults to off so the baseline
is plain DQN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import BaseAgent


class QNetwork(nn.Module):
    """Simple MLP mapping an observation vector to one Q-value per action."""

    def __init__(self, obs_size: int, n_actions: int,
                 hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_size
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions, stored in NumPy arrays."""

    def __init__(self, capacity: int, obs_size: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # 1.0 if next state is terminal (target reached) -> no bootstrap.
        self.terminals = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._full = False
        self._rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminal):
        i = self._pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminals[i] = float(terminal)
        self._pos = (self._pos + 1) % self.capacity
        self._full = self._full or self._pos == 0

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, len(self), size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.terminals[idx])


class DQNAgent(BaseAgent):
    """DQN agent with experience replay and a target network."""

    def __init__(self,
                 obs_size: int,
                 n_actions: int = 4,
                 hidden: tuple[int, ...] = (128, 128),
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 50_000,
                 batch_size: int = 64,
                 learning_starts: int = 1_000,
                 target_sync: int = 500,
                 train_freq: int = 1,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay_steps: int = 20_000,
                 max_grad_norm: float = 10.0,
                 double_dqn: bool = False,
                 device: str | None = None,
                 seed: int = 0):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_sync = target_sync
        self.train_freq = train_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.max_grad_norm = max_grad_norm
        self.double_dqn = double_dqn

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                              else "cpu"))
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.online = QNetwork(obs_size, n_actions, hidden).to(self.device)
        self.target = QNetwork(obs_size, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_size, seed=seed)

        self.train_step = 0   # gradient updates performed
        self.env_step = 0     # environment transitions observed

    # ----------------------------------------------------------- exploration
    def epsilon(self) -> float:
        """Linearly-annealed exploration rate based on env steps seen."""
        frac = min(1.0, self.env_step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def _greedy_action(self, obs: np.ndarray) -> int:
        t = torch.as_tensor(obs, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        return int(self.online(t).argmax(dim=1).item())

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action for *training* (set ``greedy=True`` to exploit)."""
        if (not greedy) and self._rng.random() < self.epsilon():
            return int(self._rng.integers(self.n_actions))
        return self._greedy_action(obs)

    def take_action(self, state) -> int:
        """Greedy action -- the policy used at evaluation time (BaseAgent API)."""
        return self._greedy_action(np.asarray(state, dtype=np.float32))

    def update(self, state, reward: float, action: int):
        """Unused: DQN learns from full transitions via :meth:`observe`/:meth:`learn`.
        Implemented to satisfy the BaseAgent interface."""
        pass

    # --------------------------------------------------------------- learning
    def observe(self, obs, action, reward, next_obs, terminal):
        """Store a transition and advance the env-step / exploration counter."""
        self.buffer.add(obs, action, reward, next_obs, terminal)
        self.env_step += 1

    def learn(self) -> float | None:
        """Run the periodic gradient update(s); returns the loss, or None if not
        enough data has been collected yet or it is not a training step."""
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None
        if self.env_step % self.train_freq != 0:
            return None

        obs, actions, rewards, next_obs, terminals = self.buffer.sample(
            self.batch_size)
        obs = torch.as_tensor(obs, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        terminals = torch.as_tensor(terminals, device=self.device)

        # Q(s, a) for the actions actually taken.
        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Online net selects the action, target net evaluates it.
                next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            target_q = rewards + self.gamma * next_q * (1.0 - terminals)

        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "obs_size": self.obs_size,
            "n_actions": self.n_actions,
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
