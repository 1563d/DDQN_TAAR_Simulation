from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    """46-128-128-26 network, matching the manuscript implementation."""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6, beta_start=0.4, action_dim=26):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.action_dim = int(action_dim)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, s, a, r, ns, d, next_valid_mask=None):
        """Store one transition plus the next-state feasibility mask.

        ``next_valid_mask`` is required by masked Double-DQN target selection.
        The optional default keeps the container backward-compatible with code
        that does not supply a mask, in which case all actions are considered.
        """
        p = self.priorities.max() if self.buffer else 1.0
        if next_valid_mask is None:
            next_valid_mask = np.ones(self.action_dim, dtype=np.bool_)
        item = (
            np.asarray(s, dtype=np.float32),
            int(a),
            float(r),
            np.asarray(ns, dtype=np.float32),
            bool(d),
            np.asarray(next_valid_mask, dtype=np.bool_),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.priorities[self.pos] = max(float(p), 1e-6)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError('not enough samples')
        p = self.priorities[:n].astype(np.float64)
        if not np.isfinite(p).all() or p.sum() <= 0:
            p = np.ones(n, dtype=np.float64)
        probs = p / p.sum()
        idx = np.random.choice(n, int(batch_size), replace=True, p=probs)
        w = (n * probs[idx]) ** (-self.beta)
        w = w / (w.max() + 1e-8)
        return [self.buffer[i] for i in idx], idx, w.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.priorities[int(i)] = (abs(float(e)) + 1e-6) ** self.alpha


class DDQNPERAgent:
    def __init__(self, state_dim, action_dim, device='cpu', cfg=None):
        cfg = cfg or {}
        self.device = torch.device(device)
        self.policy = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.lr = float(cfg.get('lr', 1.5e-3))
        self.gamma = float(cfg.get('gamma', 0.99))
        self.tau = float(cfg.get('tau', 0.01))
        self.batch_size = int(cfg.get('batch_size', 128))
        self.update_every = int(cfg.get('update_every', 4))
        self.grad_steps = int(cfg.get('grad_steps_per_update', 2))
        self.beta_start = float(cfg.get('per_beta_start', 0.4))
        self.beta_steps = int(cfg.get('per_beta_anneal_steps', 200000))
        self.action_dim = int(action_dim)
        self.use_action_masking = bool(cfg.get('feasibility_action_masking', True))
        self.step_count = 0
        self.update_steps = 0
        self.opt = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.buf = PrioritizedReplayBuffer(
            int(cfg.get('replay_capacity', 100000)),
            float(cfg.get('per_alpha', 0.6)),
            self.beta_start,
            action_dim=self.action_dim,
        )

    def _normalize_mask(self, valid_mask):
        if (not self.use_action_masking) or valid_mask is None:
            return np.ones(self.action_dim, dtype=np.bool_)
        mask = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
        if mask.size != self.action_dim:
            raise ValueError(f'valid_mask has {mask.size} entries; expected {self.action_dim}')
        # A fully masked state should not occur in the validated scenarios.
        # Fall back to the unmasked action set instead of producing NaNs/crashing;
        # callers separately record zero-valid-action events for auditability.
        if not mask.any():
            return np.ones(self.action_dim, dtype=np.bool_)
        return mask

    def select_action(self, state, epsilon=0.0, valid_mask=None):
        mask = self._normalize_mask(valid_mask)
        valid_idx = np.flatnonzero(mask)
        if random.random() < float(epsilon):
            return int(random.choice(valid_idx.tolist()))
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(x).squeeze(0)
            if self.use_action_masking:
                mt = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                q = q.masked_fill(~mt, -torch.inf)
            return int(q.argmax().item())

    @torch.no_grad()
    def soft_update(self):
        for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * pp.data)

    def update(self):
        if len(self.buf) < self.batch_size:
            return None
        batch, idx, weights = self.buf.sample(self.batch_size)
        s, a, r, ns, d, next_mask = zip(*batch)
        s = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(a), dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(np.asarray(r), dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(np.asarray(ns), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(np.asarray(d), dtype=torch.float32, device=self.device).unsqueeze(1)
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask = torch.as_tensor(np.asarray(next_mask), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q_online_next = self.policy(ns)
            if self.use_action_masking:
                # Defensive fallback for pathological all-false rows.
                all_false = ~next_mask.any(dim=1)
                if all_false.any():
                    next_mask = next_mask.clone()
                    next_mask[all_false] = True
                q_online_next = q_online_next.masked_fill(~next_mask, -torch.inf)
            next_a = q_online_next.argmax(1, keepdim=True)
            next_q = self.target(ns).gather(1, next_a)
            y = torch.clamp(r + self.gamma * next_q * (1 - d), -200.0, 200.0)

        q = self.policy(s).gather(1, a)
        td = y - q
        loss = (w * F.smooth_l1_loss(q, y, reduction='none')).mean()
        if not torch.isfinite(loss):
            return None
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        self.buf.update_priorities(idx, td.detach().view(-1).cpu().numpy())
        self.update_steps += 1
        self.soft_update()
        frac = min(1.0, self.update_steps / max(1, self.beta_steps))
        self.buf.beta = self.beta_start + (1.0 - self.beta_start) * frac
        return float(loss.item())

    def step_and_maybe_update(self):
        self.step_count += 1
        if self.step_count % self.update_every:
            return None
        vals = []
        for _ in range(self.grad_steps):
            v = self.update()
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    def checkpoint(self, **extra):
        out = {
            'policy': self.policy.state_dict(),
            'target': self.target.state_dict(),
            'opt': self.opt.state_dict(),
            'agent_config': {
                'lr': self.lr,
                'gamma': self.gamma,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'update_every': self.update_every,
                'grad_steps_per_update': self.grad_steps,
                'per_beta_start': self.beta_start,
                'per_beta_anneal_steps': self.beta_steps,
                'feasibility_action_masking': self.use_action_masking,
            },
        }
        out.update(extra)
        return out

    def load_checkpoint(self, payload, load_optimizer=False):
        self.policy.load_state_dict(payload['policy'])
        self.target.load_state_dict(payload.get('target', payload['policy']))
        if load_optimizer and 'opt' in payload:
            self.opt.load_state_dict(payload['opt'])
        self.policy.eval()
        self.target.eval()
