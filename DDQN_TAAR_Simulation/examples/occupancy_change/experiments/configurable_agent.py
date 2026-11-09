"""Configurable Double-DQN agent used only by the paper ablation suite.

The original project hard-wires prioritized experience replay (PER) inside
``DQNAgent``.  For a valid PER ablation we need the *same network and update
rule* with only the replay mechanism switched between uniform replay and PER.
This module provides that switch without changing the original project API.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from path_planner.agent import DQN, PrioritizedReplayBuffer


class UniformReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done, td_error=None):
        del td_error
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, int(batch_size))


@dataclass
class AgentConfig:
    lr: float = 1.5e-3
    gamma: float = 0.99
    tau: float = 0.01
    batch_size: int = 128
    update_every: int = 4
    grad_steps_per_update: int = 2
    replay_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_anneal_steps: int = 200_000


class ConfigurableDQNAgent:
    """Double DQN with a clean ``use_per`` switch.

    Network architecture, target construction, optimizer, soft update and Huber
    loss follow the project's existing DQNAgent.  The only ablated component is
    the replay sampler / importance weighting.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        use_per: bool = True,
        config: AgentConfig | None = None,
    ):
        cfg = config or AgentConfig()
        self.cfg = cfg
        self.device = torch.device(device)
        self.use_per = bool(use_per)

        self.policy = DQN(state_dim, action_dim).to(self.device)
        self.target = DQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr)

        if self.use_per:
            self.buf = PrioritizedReplayBuffer(
                capacity=cfg.replay_capacity,
                alpha=cfg.per_alpha,
                beta_start=cfg.per_beta_start,
            )
        else:
            self.buf = UniformReplayBuffer(capacity=cfg.replay_capacity)

        self.batch_size = int(cfg.batch_size)
        self.gamma = float(cfg.gamma)
        self.tau = float(cfg.tau)
        self.action_dim = int(action_dim)
        self.update_every = int(cfg.update_every)
        self.grad_steps_per_update = int(cfg.grad_steps_per_update)
        self._step_count = 0
        self.update_steps = 0
        self.beta_start = float(cfg.per_beta_start)
        self.beta_anneal_steps = int(cfg.per_beta_anneal_steps)

    def select_action(self, state, eps: float):
        if random.random() < float(eps):
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy(x).argmax(dim=1).item())

    @torch.no_grad()
    def soft_update(self):
        for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * policy_param.data)

    def update(self):
        if len(self.buf) < self.batch_size:
            return None

        if self.use_per:
            batch, indices, weights = self.buf.sample(self.batch_size)
        else:
            batch = self.buf.sample(self.batch_size)
            indices = None
            weights = np.ones(self.batch_size, dtype=np.float32)

        s, a, r, ns, done = zip(*batch)
        s = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(a), dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(np.asarray(r), dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(np.asarray(ns), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(done), dtype=torch.float32, device=self.device).unsqueeze(1)
        w = torch.as_tensor(np.asarray(weights), dtype=torch.float32, device=self.device).unsqueeze(1)

        if not (torch.isfinite(s).all() and torch.isfinite(r).all() and torch.isfinite(ns).all()):
            return None

        with torch.no_grad():
            next_actions = self.policy(ns).argmax(dim=1, keepdim=True)
            next_q_values = self.target(ns).gather(1, next_actions)
            target_q = r + self.gamma * next_q_values * (1.0 - done)
            target_q = torch.clamp(target_q, -200.0, 200.0)

        current_q = self.policy(s).gather(1, a)
        td_errors = target_q - current_q
        per_sample = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (w * per_sample).mean()

        if not torch.isfinite(loss):
            return None

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.opt.step()

        if self.use_per:
            self.buf.update_priorities(indices, td_errors.view(-1).detach().cpu().numpy())

        self.update_steps += 1
        self.soft_update()

        if self.use_per:
            if self.beta_anneal_steps > 0:
                frac = min(1.0, self.update_steps / self.beta_anneal_steps)
                self.buf.beta = self.beta_start + (1.0 - self.beta_start) * frac
            else:
                self.buf.beta = 1.0

        return float(loss.item())

    def step_and_maybe_update(self):
        self._step_count += 1
        if self._step_count % self.update_every != 0:
            return None
        losses = []
        for _ in range(self.grad_steps_per_update):
            value = self.update()
            if value is not None:
                losses.append(value)
        return float(np.mean(losses)) if losses else None

    def checkpoint(self, **extra) -> Dict[str, Any]:
        payload = {
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "use_per": self.use_per,
            "agent_config": self.cfg.__dict__,
        }
        payload.update(extra)
        return payload

    def load_checkpoint(self, payload: Dict[str, Any], load_optimizer: bool = False):
        self.policy.load_state_dict(payload["policy"])
        self.target.load_state_dict(payload.get("target", payload["policy"]))
        if load_optimizer and "opt" in payload:
            self.opt.load_state_dict(payload["opt"])
        self.policy.eval()
        self.target.eval()
