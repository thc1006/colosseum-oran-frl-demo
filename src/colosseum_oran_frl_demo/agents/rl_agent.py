# ─── src/colosseum_oran_frl_demo/agents/rl_agent.py ───
"""
Minimal DQN agent，抽自原 Notebook，並針對 dtype、detach、epsilon 更新做最佳化。

Key points
----------
* replay() 中的 target Q 值加上 .detach()，避免意外梯度回傳
* 使用 torch.float32 以降低 GPU / CPU 記憶體佔用
* epsilon 指數衰減，並保留 epsilon_min
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RLAgent:
    """A lightweight DQN agent tailored for the slice-sim environment."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        memory_capacity: int = 20_000,
        device: str = """cpu""",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # ε-greedy settings
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998

        self.device = torch.device(device)
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=memory_capacity
        )

    # ----------------------------------------------------
    # Interaction
    # ----------------------------------------------------
    @torch.no_grad()
    def act(self, state: np.ndarray, is_eval: bool = False) -> int:
        """Return an action (0 … action_size-1)."""
        if (not is_eval) and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        q_vals = self.model(state_t)
        return int(torch.argmax(q_vals).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    # ----------------------------------------------------
    # Learning
    # ----------------------------------------------------
    def replay(self, batch_size: int = 64) -> float | None:
        """
        Trains the agent by replaying a batch of experiences from memory.

        Args:
            batch_size: The size of the batch to use for training.

        Returns:
            The loss value, or None if there is not enough memory.
        """
        if len(self.memory) < batch_size:
            return None

        minibatch: List[Tuple] = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        ).view(-1, 1)

        # Predicted Q(s,a)
        q_pred = self.model(states_t).gather(1, actions_t).squeeze()

        # Target: r + γ * max_a' Q(s',a') * (1-done)
        with torch.no_grad():
            max_next_q = torch.max(self.model(next_t), dim=1)[0]
            q_target = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.loss_fn(q_pred, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε-decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()
