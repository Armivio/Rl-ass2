from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import BaseAgent


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN network.

    The network estimates Q(s, a) through two streams:
        V(s): state value
        A(s, a): action advantage

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.feature(states)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


class DuelingDQNAgent(BaseAgent):

    def __init__(
        self,
        learning_rate: float = 1e-3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float | None = 0.995,
        min_epsilon: float = 0.05,
        n_actions: int = 4,
        state_dim: int = 2,
        hidden_dim: int = 128,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        train_start: int = 500,
        target_update_freq: int = 200,
        state_scale: float = 1.0,
        double_dqn: bool = True,
        device: str | None = None,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.state_dim = state_dim

        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_start = train_start
        self.target_update_freq = target_update_freq
        self.state_scale = float(state_scale) if state_scale else 1.0
        self.double_dqn = double_dqn

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.q_net = DuelingQNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=buffer_size)

        self.last_state: tuple[int, ...] | None = None
        self.last_action: int | None = None
        self.training = True
        self.learn_step = 0

        self.observed_states: set[tuple[int, ...]] = set()

    def _ensure_state(self, state) -> tuple[int, ...]:
        state_array = np.asarray(state).reshape(-1)

        if len(state_array) != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got state with shape {np.asarray(state).shape}: {state}"
            )

        state_tuple = tuple(int(x) for x in state_array)
        self.observed_states.add(state_tuple)
        return state_tuple

    def _states_to_tensor(self, states: Iterable[tuple[int, ...]] | tuple[int, ...]) -> torch.Tensor:
        states_array = np.asarray(states, dtype=np.float32)

        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)

        states_array = states_array / self.state_scale
        return torch.tensor(states_array, dtype=torch.float32, device=self.device)

    def train_mode(self):
        self.training = True
        self.q_net.train()

    def eval_mode(self):
        self.training = False
        self.q_net.eval()

    def take_action(self, state) -> int:
        state = self._ensure_state(state)

        if self.training and random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = self._states_to_tensor(state)
                q_values = self.q_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        self.last_state = state
        self.last_action = action
        return action

    def update(self, state, reward: float, action: int, terminated: bool = False):
        """
        Returns:
            float | None: latest minibatch loss, or None if the replay buffer is
            not ready yet.
        """
        if not self.training:
            return None

        if self.last_state is None or self.last_action is None:
            return None

        old_state = self.last_state
        old_action = int(action) if action is not None else int(self.last_action)
        next_state = self._ensure_state(state)
        done = bool(terminated)

        self.replay_buffer.append(
            (
                old_state,
                old_action,
                float(reward),
                next_state,
                done,
            )
        )

        loss = self._learn_from_replay()

        if done:
            self.reset_episode()

        return loss

    def _learn_from_replay(self):
        if len(self.replay_buffer) < self.train_start:
            return None

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = self._states_to_tensor(states)
        next_states_tensor = self._states_to_tensor(next_states)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.q_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Online network chooses the action; target network evaluates it.
                next_actions = self.q_net(next_states_tensor).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_tensor).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states_tensor).max(dim=1)[0]

            target_q = rewards_tensor + self.gamma * next_q * (1.0 - dones_tensor)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        if self.epsilon_decay is not None:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        self.last_state = None
        self.last_action = None

    def get_q_values(self, state) -> np.ndarray:
        state = self._ensure_state(state)
        self.q_net.eval()

        with torch.no_grad():
            q_values = self.q_net(self._states_to_tensor(state)).squeeze(0).cpu().numpy()

        if self.training:
            self.q_net.train()

        return q_values

    def get_policy(self) -> dict[tuple[int, ...], int]:
        policy = {}
        for state in self.observed_states:
            policy[state] = int(np.argmax(self.get_q_values(state)))
        return policy

    def get_value_function(self) -> dict[tuple[int, ...], float]:
        values = {}
        for state in self.observed_states:
            values[state] = float(np.max(self.get_q_values(state)))
        return values

    def save_model(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step": self.learn_step,
            "observed_states": list(self.observed_states),
            "replay_buffer": list(self.replay_buffer),
            "config": {
                "learning_rate": self.alpha,
                "gamma": self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "min_epsilon": self.min_epsilon,
                "n_actions": self.n_actions,
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "train_start": self.train_start,
                "target_update_freq": self.target_update_freq,
                "state_scale": self.state_scale,
                "double_dqn": self.double_dqn,
            },
        }

        torch.save(checkpoint, path)

    def load_model(self, path: str | Path):
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.learn_step = checkpoint.get("learn_step", 0)
        self.observed_states = {tuple(s) for s in checkpoint.get("observed_states", [])}

        self.replay_buffer.clear()
        for old_state, old_action, reward, next_state, done in checkpoint.get("replay_buffer", []):
            self.replay_buffer.append(
                (
                    tuple(old_state),
                    int(old_action),
                    float(reward),
                    tuple(next_state),
                    bool(done),
                )
            )

    # Compatibility names. They are not actually saving a Q-table anymore.
    def save_q_table(self, path: str | Path):
        self.save_model(path)

    def load_q_table(self, path: str | Path):
        self.load_model(path)
