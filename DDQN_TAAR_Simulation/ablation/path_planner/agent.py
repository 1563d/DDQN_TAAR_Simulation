# agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# ----------------------------
# Q 网络（DQN）
# ----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------
# 标准经验回放（保留，不删除，避免外部依赖损坏）
# ----------------------------
class ReplayBuffer:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)
    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))
    def sample(self, bs):
        b = random.sample(self.buf, bs)
        s, a, r, ns, d = zip(*b)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)
    def __len__(self):
        return len(self.buf)

# ----------------------------
# 优先经验回放（基于你提供的伪代码）
# ----------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6, beta_start=0.4):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta_start)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done, td_error=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        priority = max_priority if td_error is None else (abs(float(td_error)) + 1e-6)
        self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0:
            raise ValueError("PrioritizedReplayBuffer is empty.")
        if n == self.capacity:
            priorities = self.priorities
        else:
            # 注意：当缓冲区未满时，优先级数组有效部分是 [0, self.pos)
            priorities = self.priorities[:max(self.pos, 1)]
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(probs), batch_size, p=probs)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max() + 1e-8
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + 1e-6) ** self.alpha
            self.priorities[int(idx)] = priority

# ----------------------------
# DQN 智能体（接入 PER + beta 线性退火）
# ----------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu', lr=1.5e-3, gamma=0.99, tau=0.01,
                 bs=128, update_every=4, grad_steps_per_update=2,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_anneal_steps=200_000):
        self.device = torch.device(device)
        self.policy = DQN(state_dim, action_dim).to(self.device)
        self.target = DQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

        # 原 ReplayBuffer 保留不删，这里改用优先经验回放
        # self.buf = ReplayBuffer()
        self.buf = PrioritizedReplayBuffer(capacity=100_000, alpha=per_alpha, beta_start=per_beta_start)

        self.batch_size = bs
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.update_every = update_every
        self.grad_steps_per_update = grad_steps_per_update
        self._step_count = 0
        self.update_steps = 0

        # PER beta 线性退火参数
        self.beta_start = float(per_beta_start)
        self.beta_anneal_steps = int(per_beta_anneal_steps)

    def select_action(self, s, eps):
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(s).argmax(dim=1).item()

    @torch.no_grad()
    def soft_update(self):
        for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * pp.data)

    def update(self):
        if len(self.buf) < self.batch_size:
            return None

        # 从 PER 采样
        batch, indices, weights = self.buf.sample(self.batch_size)
        s, a, r, ns, done = zip(*batch)

        s    = torch.as_tensor(np.array(s),    dtype=torch.float32, device=self.device)
        a    = torch.as_tensor(np.array(a),    dtype=torch.int64,   device=self.device).unsqueeze(1)  # [B,1]
        r    = torch.as_tensor(np.array(r),    dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        ns   = torch.as_tensor(np.array(ns),   dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.array(done), dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        w    = torch.as_tensor(np.array(weights), dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]

        if (not torch.isfinite(s).all()
            or not torch.isfinite(r).all()
            or not torch.isfinite(ns).all()):
            print("⚠️ Non-finite values in batch, skip update.")
            return None

        # -------- Double DQN target 计算（唯一改动点）--------
        with torch.no_grad():
            # 用 policy 网络在 ns 上选择动作
            next_actions = self.policy(ns).argmax(dim=1, keepdim=True)           # [B,1]
            # 用 target 网络评估这些动作的 Q 值
            next_q_values = self.target(ns).gather(1, next_actions)              # [B,1]
            target_q = r + self.gamma * next_q_values * (1.0 - done)
            target_q = torch.clamp(target_q, -200.0, 200.0)
        # ---------------------------------------------------

        current_q = self.policy(s).gather(1, a)

        # TD 误差与加权 Huber 损失（重要性采样权重）
        td_errors = target_q - current_q  # [B,1]
        loss_per_sample = F.smooth_l1_loss(current_q, target_q, reduction='none')  # [B,1]
        loss = (w * loss_per_sample).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Invalid loss encountered, skip update.")
            return None

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.opt.step()

        # 更新优先级
        with torch.no_grad():
            self.buf.update_priorities(indices, td_errors.view(-1).detach().cpu().numpy())

        # 计数 + 目标网络软更新
        self.update_steps += 1
        self.soft_update()

        # 线性退火 beta: beta_start -> 1.0
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
            l = self.update()
            if l is not None:
                losses.append(l)
        return float(np.mean(losses)) if losses else None