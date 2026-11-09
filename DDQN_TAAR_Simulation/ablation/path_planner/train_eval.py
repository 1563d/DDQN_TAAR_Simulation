import numpy as np
import torch
import random
from scipy.io import savemat
from tqdm import tqdm

from .agent import DQNAgent
from .smoothing import smooth_path_safe
from .env import GridWorld3DEnv
from .viz import save_training_curves, animate_path, save_static_path, show_interactive_mpl


def _euclid(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def greedy_success_once(size_x, size_y, size_z, agent, max_steps=3000):
    env = GridWorld3DEnv(size_x=size_x, size_y=size_y, size_z=size_z,
                         max_steps=5 * (size_x + size_y + size_z),
                         cube1=None, cube2=None)
    s, _ = env.reset()
    for _ in range(max_steps):
        a = agent.select_action(s, 0.0)
        s, r, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break
    reached_wp = bool(getattr(env, "wp_reached", False))
    d_goal = env._distance_to_target(env.pos, env.goal)
    reached_goal = reached_wp and (d_goal <= 0.5)
    return reached_goal


def detect_and_print_mode_change(prev_pos, next_pos, step_idx=None):
    if prev_pos is None or next_pos is None:
        return
    try:
        pz = int(prev_pos[2])
        nz = int(next_pos[2])
    except Exception:
        return
    where_str = f" | pos=({int(next_pos[0])},{int(next_pos[1])},{int(next_pos[2])})"
    step_str = f" | step={int(step_idx)}" if step_idx is not None else ""
    if pz == 0 and nz > 0:
        print(f"[Mode] Switch: Ground -> Air (z: 0 -> >0){where_str}{step_str}")
    elif pz > 0 and nz == 0:
        print(f"[Mode] Switch: Air -> Ground (z: >0 -> 0){where_str}{step_str}")


class ImprovedCurriculumConfig:
    """
    改进的课程学习配置，提供更细粒度的难度控制
    """
    def __init__(
        self,
        total_episodes=400,
        phase_episodes=(80, 80, 80, 80, 80),
        phase_min_episodes=(20, 20, 20, 20, 0),
        phase_max_episodes=(50, 50, 50, 50, None),
        mastery_consecutive_checks=2,
    ):
        self.total_episodes = int(total_episodes)
        self.mastery_consecutive_checks = max(1, int(mastery_consecutive_checks))

        # 定义5个递进的难度级别的障碍开关
        self.phase_masks = [
            # Phase 0: 仅塔筒和叶片（学习基础导航）
            dict(A=False, B=False, C=False, D=False, E=False, CAB=False,
                 W1=False, W2=False, W3=False, W4=False),

            # Phase 1: 添加围墙（学习简单避障）
            dict(A=False, B=False, C=False, D=False, E=False, CAB=False,
                 W1=True, W2=True, W3=True, W4=True),

            # Phase 2: 添加柜子和少量障碍（学习路径规划）
            dict(A=True, B=False, C=False, D=False, E=True, CAB=True,
                 W1=True, W2=True, W3=True, W4=True),

            # Phase 3: 添加更多障碍（强制陆空切换）
            dict(A=True, B=False, C=True, D=True, E=True, CAB=True,
                 W1=True, W2=True, W3=True, W4=True),

            # Phase 4: 全部障碍（复杂场景）
            dict(A=True, B=True, C=True, D=True, E=True, CAB=True,
                 W1=True, W2=True, W3=True, W4=True),
        ]

        # 每个阶段的配置参数（长度与 phase_masks 对齐）
        self.phase_configs = [
            {"name": "Basic Navigation", "success_threshold": 0.7, "epsilon_decay": 0.95},
            {"name": "Simple Obstacles", "success_threshold": 0.65, "epsilon_decay": 0.93},
            {"name": "Path Planning", "success_threshold": 0.6, "epsilon_decay": 0.91},
            {"name": "Modal Switching", "success_threshold": 0.55, "epsilon_decay": 0.90},
            {"name": "Complex Scenario", "success_threshold": 0.5, "epsilon_decay": 0.88},
        ]

        # 如果没有指定每阶段回合数，平均分配（默认按5阶段）
        if phase_episodes is None:
            phase_episodes = [80, 80, 80, 80, 80]

        # 复制列表
        self.phase_episodes = list(map(int, phase_episodes))

        # 规范长度：与阶段数对齐（扩展或截断）
        n_phases = len(self.phase_masks)
        if len(self.phase_episodes) < n_phases:
            # 补齐缺失阶段预算，均分剩余总回合数；若无剩余则补零
            used = sum(self.phase_episodes)
            remaining = max(0, self.total_episodes - used)
            missing = n_phases - len(self.phase_episodes)
            base = remaining // missing if missing > 0 else 0
            extra = remaining % missing if missing > 0 else 0
            self.phase_episodes += [base] * missing
            for i in range(extra):
                self.phase_episodes[len(phase_episodes) + i] += 1
        elif len(self.phase_episodes) > n_phases:
            self.phase_episodes = self.phase_episodes[:n_phases]

        # 最后根据 total_episodes 调整最后一个阶段，使总和严格等于 total_episodes
        if sum(self.phase_episodes) != self.total_episodes:
            diff = self.total_episodes - sum(self.phase_episodes[:-1])
            self.phase_episodes[-1] = max(0, diff)

        # Adaptive-curriculum bounds used by the paper ablation protocol.
        # Non-final phases are short warm-up stages: they must receive a minimum
        # amount of training, can advance immediately after stable mastery, and
        # are capped so that a substantial part of the fixed 400-episode budget
        # is always reserved for the final full-complexity task.
        self.phase_min_episodes = list(phase_min_episodes)
        self.phase_max_episodes = list(phase_max_episodes)
        if len(self.phase_min_episodes) < n_phases:
            self.phase_min_episodes += [0] * (n_phases - len(self.phase_min_episodes))
        self.phase_min_episodes = self.phase_min_episodes[:n_phases]
        if len(self.phase_max_episodes) < n_phases:
            self.phase_max_episodes += [None] * (n_phases - len(self.phase_max_episodes))
        self.phase_max_episodes = self.phase_max_episodes[:n_phases]
        for i in range(n_phases):
            self.phase_min_episodes[i] = max(0, int(self.phase_min_episodes[i]))
            if self.phase_max_episodes[i] is not None:
                self.phase_max_episodes[i] = max(
                    self.phase_min_episodes[i], int(self.phase_max_episodes[i])
                )
        # The final phase consumes all remaining global episodes.
        self.phase_max_episodes[-1] = None

    def get_phase_config(self, phase_idx):
        """获取指定阶段的配置"""
        if 0 <= phase_idx < len(self.phase_configs):
            return self.phase_configs[phase_idx]
        return self.phase_configs[-1]

    def get_difficulty_metrics(self, phase_idx):
        """返回当前难度的量化指标（用于日志记录）"""
        if phase_idx < 0 or phase_idx >= len(self.phase_masks):
            phase_idx = len(self.phase_masks) - 1

        mask = self.phase_masks[phase_idx]
        # 计算启用的障碍物数量
        obstacles_enabled = sum([
            int(mask.get("A", False)),
            int(mask.get("B", False)),
            int(mask.get("C", False)),
            int(mask.get("D", False)),
            int(mask.get("E", False)),
            int(mask.get("CAB", False)),
        ])

        walls_enabled = sum([
            int(mask.get("W1", False)),
            int(mask.get("W2", False)),
            int(mask.get("W3", False)),
            int(mask.get("W4", False)),
        ])

        return {
            "phase": phase_idx,
            "name": self.phase_configs[phase_idx]["name"],
            "obstacle_count": obstacles_enabled,
            "wall_count": walls_enabled,
            "total_obstacles": obstacles_enabled + walls_enabled,
            "complexity": (obstacles_enabled + walls_enabled * 0.5) / 10.0,  # 归一化复杂度
        }


def _apply_curriculum_obstacles(env, mask):
    Z, Y, X = env.size_z, env.size_y, env.size_x

    def _slice_safe(z0, z1, y0, y1, x0, x1):
        z0 = max(0, min(Z, z0)); z1 = max(0, min(Z, z1))
        y0 = max(0, min(Y, y0)); y1 = max(0, min(Y, y1))
        x0 = max(0, min(X, x0)); x1 = max(0, min(X, x1))
        return z0, z1, y0, y1, x0, x1

    def _apply_block(on, z0, z1, y0, y1, x0, x1, val_if_on=1):
        z0, z1, y0, y1, x0, x1 = _slice_safe(z0, z1, y0, y1, x0, x1)
        if (z1 > z0) and (y1 > y0) and (x1 > x0):
            env.grid[z0:z1, y0:y1, x0:x1] = (val_if_on if on else 0)

    _apply_block(bool(mask.get("A", True)), 0, 6, 64, 67, 36, 50)
    _apply_block(bool(mask.get("B", True)), 0, 6, 32, 35, 0, 12)
    _apply_block(bool(mask.get("C", True)), 0, 6, 48, 51, 12, 24)
    _apply_block(bool(mask.get("D", True)), 0, 6, 70, 76, 24, 27)
    _apply_block(bool(mask.get("E", True)), 0, 6, 34, 40, 36, 42)

    cab_x0, cab_x1 = 3, 6
    cab_y0, cab_y1 = 56, 59
    cab_h = min(6, Z)
    _apply_block(bool(mask.get("CAB", True)), 0, cab_h, cab_y0, cab_y1, cab_x0, cab_x1)

    ix0, ix1, iy0, iy1 = env.wall_inner
    ox0 = max(0, ix0 - 1)
    ox1 = min(X - 1, ix1 + 1)
    oy0 = max(0, iy0 - 1)
    oy1 = min(Y - 1, iy1 + 1)
    wall_h = min(6, Z)

    W1_on = bool(mask.get("W1", True))
    W2_on = bool(mask.get("W2", True))
    W3_on = bool(mask.get("W3", True))
    W4_on = bool(mask.get("W4", True))

    _apply_block(W1_on, 0, wall_h, max(0, oy0 - 30), max(0, oy0 + 1 - 30), ox0, ox1 + 1)
    _apply_block(W2_on, 0, wall_h, max(0, oy1 - 30), max(0, oy1 + 1 - 30), ox0, ox1 + 1)
    _apply_block(W3_on, 0, wall_h, max(0, oy0 - 30), max(0, oy1 + 1 - 30), ox0, ox0 + 1)
    _apply_block(W4_on, 0, wall_h, max(0, oy0 - 30), max(0, oy1 + 1 - 30), ox1, ox1 + 1)

    env._refresh_padded_grid()


class ImprovedAdaptiveCurriculumRunner:
    """
    改进的自适应课程训练器
    """
    def __init__(
        self,
        env,
        agent,
        curriculum: ImprovedCurriculumConfig,
        batch_train_episodes=30,  # 增加批次大小
        eval_trials_per_check=20,  # 减少评估次数
        max_total_episodes=None,
        seed=42,
        verbose=True,
        episode_end_callback=None
    ):
        self.env = env
        self.agent = agent
        self.cur = curriculum
        self.batch_train_episodes = int(batch_train_episodes)
        self.eval_trials_per_check = int(eval_trials_per_check)
        self.max_total_episodes = int(max_total_episodes) if max_total_episodes else int(curriculum.total_episodes)
        self.seed = seed
        self.verbose = verbose
        self.episode_end_callback = episode_end_callback

        # 训练统计
        self.rewards = []
        self.best_reward = -float('inf')
        self.global_episode_idx = 0
        self.phase_rewards = {i: [] for i in range(len(curriculum.phase_masks))}
        self.phase_success_rates = {i: [] for i in range(len(curriculum.phase_masks))}
        self.phase_history = []

    def _apply_phase(self, phase_idx):
        """应用指定阶段的环境配置"""
        mask = self.cur.phase_masks[phase_idx]

        # 重置环境
        if self.global_episode_idx == 0:
            s, _ = self.env.reset(seed=self.seed)
        else:
            s, _ = self.env.reset()

        # 应用障碍物遮罩
        _apply_curriculum_obstacles(self.env, mask)

        if self.verbose and self.global_episode_idx % 50 == 0:
            metrics = self.cur.get_difficulty_metrics(phase_idx)
            print(f"\n[Curriculum] Phase {phase_idx}: {metrics['name']}")
            print(f"  Obstacles: {metrics['obstacle_count']}, Walls: {metrics['wall_count']}")

        return self.env._norm_state(self.env.pos)

    def _train_one_episode(self, eps, pbar=None, phase_idx=0):
        """训练单个回合"""
        env = self.env
        agent = self.agent

        # 应用当前阶段配置
        s = self._apply_phase(phase_idx)

        total = 0.0
        last_loss = None
        step_count = 0
        max_steps = getattr(env, "max_steps", 1000)

        # 获取阶段特定的配置
        phase_config = self.cur.get_phase_config(phase_idx)

        for _ in range(max_steps):
            prev_pos = getattr(env, "pos", None)
            a = agent.select_action(s, eps)
            ns, r, terminated, truncated, _ = env.step(a)
            cur_pos = getattr(env, "pos", None)
            done = bool(terminated) or bool(truncated)

            # Paper ablation protocol: the reward function is held identical
            # across all four PER/CL variants. Curriculum learning changes only
            # the environment difficulty schedule, not the reward coefficients.

            agent.buf.push(s, a, r, ns, done)
            s = ns
            total += r
            step_count += 1

            # 更新网络
            loss = agent.step_and_maybe_update()
            if loss is not None:
                last_loss = loss

            if done:
                break

        # 额外的更新步骤
        for _ in range(5):
            loss = agent.update()
            if loss is not None:
                last_loss = loss

        # 记录统计
        self.rewards.append(total)
        self.phase_rewards[phase_idx].append(total)
        self.best_reward = max(self.best_reward, total)
        self.global_episode_idx += 1

        # Optional publication-evaluation callback. It is inference-only and is
        # not inserted into replay memory, so the training-sample budget remains
        # identical across CL and non-CL variants.
        if self.episode_end_callback is not None:
            self.episode_end_callback(self.global_episode_idx, self.agent)

        # 更新进度条
        if pbar is not None:
            wp_reached = bool(getattr(env, "wp_reached", False))
            goal_reached = env._distance_to_target(env.pos, env.goal) <= 0.5

            postfix = dict(
                phase=f"{phase_idx + 1}/{len(self.cur.phase_masks)}",
                eps=f"{eps:.2f}",
                loss=("-" if last_loss is None else f"{last_loss:.3f}"),
                R=f"{total:.1f}",
                bestR=f"{self.best_reward:.1f}",
                WP="✓" if wp_reached else "✗",
                Goal="✓" if goal_reached else "✗",
                steps=step_count
            )
            pbar.set_postfix(**postfix)

        return total

    def _estimate_success_rate(self, phase_idx):
        """评估当前阶段的成功率"""
        succ = 0
        wp_reached_count = 0

        for _ in range(self.eval_trials_per_check):
            # 创建一个临时环境进行评估
            eval_env = GridWorld3DEnv(
                size_x=self.env.size_x,
                size_y=self.env.size_y,
                size_z=self.env.size_z,
                max_steps=self.env.max_steps,
                cube1=None,
                cube2=None
            )

            s, _ = eval_env.reset()
            _apply_curriculum_obstacles(eval_env, self.cur.phase_masks[phase_idx])
            s = eval_env._norm_state(eval_env.pos)

            for _ in range(eval_env.max_steps):
                a = self.agent.select_action(s, 0.0)  # 贪心策略
                s, r, terminated, truncated, _ = eval_env.step(a)
                if terminated or truncated:
                    break

            wp_reached = bool(getattr(eval_env, "wp_reached", False))
            d_goal = eval_env._distance_to_target(eval_env.pos, eval_env.goal)

            if wp_reached:
                wp_reached_count += 1
            if wp_reached and d_goal <= 0.5:
                succ += 1

        success_rate = succ / self.eval_trials_per_check
        wp_rate = wp_reached_count / self.eval_trials_per_check

        self.phase_success_rates[phase_idx].append(success_rate)

        if self.verbose:
            print(f"\n[Eval] Phase {phase_idx + 1}: WP rate={wp_rate * 100:.1f}%, Success={success_rate * 100:.1f}%")

        return success_rate

    def run(self):
        """Run the adaptive warm-up curriculum under a fixed global budget.

        Paper protocol v3:
        - non-final phases: minimum exposure -> periodic mastery checks -> early
          advancement when mastery is stable -> forced advancement at the cap;
        - final phase: receives every unused episode and always runs until the
          global training budget is exhausted.
        """
        env = self.env
        agent = self.agent

        eps_start, eps_end = 1.0, 0.05
        eps_decay_steps = max(int(0.7 * self.max_total_episodes), 1)
        pbar = tqdm(
            total=self.max_total_episodes,
            desc="Improved Curriculum Training",
            unit="ep",
            ncols=140,
        )

        n_phases = len(self.cur.phase_masks)
        min_ep = list(self.cur.phase_min_episodes)
        max_ep = list(self.cur.phase_max_episodes)
        mastery_hold = int(self.cur.mastery_consecutive_checks)

        if self.verbose:
            print(
                f"[Curriculum-v3] min episodes={min_ep}, max episodes={max_ep}, "
                f"mastery hold={mastery_hold}, total={self.max_total_episodes}"
            )

        phase_idx = 0
        used_in_phase = 0
        consecutive_mastery = 0
        phase_start_episode = 1

        def _advance(reason, last_success_rate=None):
            nonlocal phase_idx, used_in_phase, consecutive_mastery, phase_start_episode
            self.phase_history.append({
                "phase": int(phase_idx),
                "name": self.cur.get_phase_config(phase_idx)["name"],
                "start_episode": int(phase_start_episode),
                "end_episode": int(self.global_episode_idx),
                "episodes_used": int(used_in_phase),
                "reason": str(reason),
                "last_success_rate": (
                    float(last_success_rate) if last_success_rate is not None else np.nan
                ),
            })
            if self.verbose:
                print(
                    f"\n[Curriculum-v3] Phase {phase_idx + 1} -> {phase_idx + 2}: "
                    f"{reason}; used={used_in_phase}"
                )
            phase_idx += 1
            used_in_phase = 0
            consecutive_mastery = 0
            phase_start_episode = self.global_episode_idx + 1

        while self.global_episode_idx < self.max_total_episodes and phase_idx < n_phases:
            phase_config = self.cur.get_phase_config(phase_idx)
            success_threshold = float(phase_config["success_threshold"])
            is_last_phase = phase_idx == n_phases - 1

            # The final full-complexity phase receives the entire remaining
            # budget. Earlier phases are capped by max_ep.
            if is_last_phase:
                batch_size = min(
                    self.batch_train_episodes,
                    self.max_total_episodes - self.global_episode_idx,
                )
            else:
                phase_cap = int(max_ep[phase_idx])
                remaining_cap = phase_cap - used_in_phase
                if remaining_cap <= 0:
                    _advance("maximum phase budget reached")
                    continue
                batch_size = min(
                    self.batch_train_episodes,
                    remaining_cap,
                    self.max_total_episodes - self.global_episode_idx,
                )

            if batch_size <= 0:
                break

            for _ in range(batch_size):
                if self.global_episode_idx >= self.max_total_episodes:
                    break
                frac = min(1.0, self.global_episode_idx / eps_decay_steps)
                eps = eps_start + (eps_end - eps_start) * frac
                eps *= phase_config["epsilon_decay"] ** (phase_idx / n_phases)
                self._train_one_episode(eps=eps, pbar=pbar, phase_idx=phase_idx)
                pbar.update(1)
                used_in_phase += 1

            if self.global_episode_idx >= self.max_total_episodes:
                break

            if is_last_phase:
                # No extra curriculum-gating rollout is needed in the final
                # phase. The common fixed-full-map evaluator already runs every
                # 20 episodes, and the final phase must consume the full remaining
                # training budget.
                continue

            # Do not allow advancement before the minimum exposure.
            if used_in_phase < int(min_ep[phase_idx]):
                continue

            success_rate = self._estimate_success_rate(phase_idx)
            if success_rate >= success_threshold:
                consecutive_mastery += 1
            else:
                consecutive_mastery = 0

            if consecutive_mastery >= mastery_hold:
                _advance("stable mastery", success_rate)
                continue

            if used_in_phase >= int(max_ep[phase_idx]):
                _advance("maximum phase budget reached", success_rate)

        # Log the final phase occupancy of the fixed budget.
        if phase_idx == n_phases - 1 and used_in_phase > 0:
            self.phase_history.append({
                "phase": int(phase_idx),
                "name": self.cur.get_phase_config(phase_idx)["name"],
                "start_episode": int(phase_start_episode),
                "end_episode": int(self.global_episode_idx),
                "episodes_used": int(used_in_phase),
                "reason": "global training budget exhausted",
                "last_success_rate": (
                    float(self.phase_success_rates[phase_idx][-1])
                    if self.phase_success_rates[phase_idx] else np.nan
                ),
            })

        pbar.close()

        # 保存最佳模型
        try:
            torch.save({
                "policy": agent.policy.state_dict(),
                "target": agent.target.state_dict(),
                "opt": agent.opt.state_dict(),
                "episode": self.global_episode_idx - 1,
                "best_reward": self.best_reward,
                "phase_rewards": self.phase_rewards,
                "phase_success_rates": self.phase_success_rates,
            }, "best_curriculum_dqn.pth")

            # 加载最佳权重
            ckpt = torch.load("best_curriculum_dqn.pth", map_location="cpu")
            agent.policy.load_state_dict(ckpt["policy"])
            agent.target.load_state_dict(ckpt["target"])
            agent.policy.eval()

            print(f"\n[Training Complete] Best reward: {self.best_reward:.2f}")

        except Exception as e:
            print(f"⚠️ Save/Load checkpoint failed: {e}")

        baseline_snapshot = env.get_static_occupancy_snapshot()
        return agent, self.rewards, baseline_snapshot


def train(env, episodes=1000, seed=42, save_plots=True, save_best_ckpt="best_dqn.pth",
          curriculum: ImprovedCurriculumConfig = None):
    """
    入口函数保持签名不变：
    - 若 curriculum 为 None：执行原单阶段训练（退回简单实现）。
    - 若 curriculum 不为 None：使用自适应课程驱动器。
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                     device='cuda' if torch.cuda.is_available() else 'cpu')

    if curriculum is None:
        # 简单单阶段训练（保留你原逻辑的最小实现）
        rewards = []
        eps_start, eps_end = 1.0, 0.05
        eps_decay_steps = max(int(0.6 * episodes), 1)
        pbar = tqdm(range(episodes), desc="Training", unit="ep", ncols=140)
        best_reward = -float('inf')
        for ep in pbar:
            if ep == 0:
                s, _ = env.reset(seed=seed)
            else:
                s, _ = env.reset()
            total = 0.0; last_loss = None; step_count = 0
            max_steps = getattr(env, "max_steps", 1000)
            frac = min(1.0, ep / eps_decay_steps)
            eps = eps_start + (eps_end - eps_start) * frac
            for _ in range(max_steps):
                prev_pos = getattr(env, "pos", None)
                a = agent.select_action(s, eps)
                ns, r, terminated, truncated, _ = env.step(a)
                cur_pos = getattr(env, "pos", None)
                done = bool(terminated) or bool(truncated)
                agent.buf.push(s, a, r, ns, done)
                s = ns; total += r; step_count += 1
                detect_and_print_mode_change(prev_pos, cur_pos, step_idx=step_count)
                loss = agent.step_and_maybe_update()
                if loss is not None:
                    last_loss = loss
                if done:
                    break
            for _ in range(5):
                loss = agent.update()
                if loss is not None:
                    last_loss = loss
            rewards.append(total)
            best_reward = max(best_reward, total)
            postfix = dict(eps=f"{eps:.2f}", loss=("-" if last_loss is None else f"{last_loss:.3f}"),
                           R=f"{total:.1f}", bestR=f"{best_reward:.1f}", steps=step_count)
            pbar.set_postfix(**postfix)
        pbar.close()
        try:
            torch.save({
                "policy": agent.policy.state_dict(),
                "target": agent.target.state_dict(),
                "opt": agent.opt.state_dict(),
                "episode": episodes - 1,
                "best_reward": best_reward,
            }, save_best_ckpt)
            ckpt = torch.load(save_best_ckpt, map_location="cpu")
            agent.policy.load_state_dict(ckpt["policy"])
            agent.target.load_state_dict(ckpt["target"])
            agent.policy.eval()
        except Exception as e:
            print(f"⚠️ Save/Load best checkpoint failed: {e}")
        if save_plots:
            try:
                save_training_curves(rewards, "train_rewards.png", "train_rewards.gif")
            except Exception as e:
                print("⚠️ save_training_curves failed:", e)
        baseline_snapshot = env.get_static_occupancy_snapshot()
        return agent, rewards, baseline_snapshot
    else:
        runner = ImprovedAdaptiveCurriculumRunner(
            env=env,
            agent=agent,
            curriculum=curriculum,
            batch_train_episodes=30,
            eval_trials_per_check=20,
            max_total_episodes=episodes,
            seed=seed,
            verbose=True
        )
        agent, rewards, baseline_snapshot = runner.run()
        if save_plots:
            try:
                save_training_curves(rewards, "train_rewards.png", "train_rewards.gif")
            except Exception as e:
                print("⚠️ save_training_curves failed:", e)
        return agent, rewards, baseline_snapshot


def test(env, agent, baseline_snapshot=None, steps=3000, anim_stride=2, use_ffmpeg=False, show_interactive=True, rewards=None):
    if hasattr(agent, "policy"):
        agent.policy.eval()

    bm_cfg = BehaviorManagerConfig(check_interval=1, max_replan_trials=2, verbose=True)
    bm = BehaviorManager(env, agent, bm_cfg)

    if baseline_snapshot is not None:
        bm.set_baseline_from_snapshot(baseline_snapshot)
        if bm.cfg.verbose:
            print("[Test] Using training baseline snapshot for change detection.")
    else:
        s, _ = env.reset()
        bm.set_baseline_from_env()

    s, _ = env.reset()

    path = [getattr(env, "pos", None)]
    total = 0.0
    step_idx = 0
    replanned_once_path = None

    while step_idx < steps:
        if bm.should_check(step_idx):
            if bm.need_replan(step_idx):
                new_path = bm.replan_from_here()
                bm.mark_replanned(step_idx)
                replanned_once_path = new_path
            bm._last_check_step = step_idx

        prev_pos = getattr(env, "pos", None)
        a = agent.select_action(s, 0.0)
        ns, r, terminated, truncated, _ = env.step(a)
        cur_pos = getattr(env, "pos", None)

        s = ns
        total += r
        path.append(cur_pos)

        step_idx += 1
        detect_and_print_mode_change(prev_pos, cur_pos, step_idx=step_idx)

        if bool(terminated) or bool(truncated):
            break

    path = [p for p in path if p is not None]

    print(f"Test: steps={len(path)}, reward={total:.2f}")
    print(f"  WP reached: {getattr(env, 'wp_reached', False)}")
    print(f"  Final pos: {getattr(env, 'pos', None)}, Goal: {getattr(env, 'goal', None)}")

    try:
        path_smooth = smooth_path_safe(env, path, clearance=2.0, shortcut_iters=600, fillet=True, fillet_scale=0.8,
                                       cr_alpha=0.25, cr_samples=12, cr_tangent_cap=0.7)
    except Exception as e:
        print(f"⚠️ Path smoothing failed: {e}")
        path_smooth = np.asarray(path, dtype=float)

    np.savetxt("path_smooth.csv", path_smooth, fmt="%.3f", delimiter=",", header="x,y,z", comments="")
    path_arr = np.asarray(path, dtype=np.int32)
    np.savetxt("path_centers.csv", path_arr, fmt="%.3f", delimiter=",", header="x,y,z", comments="")

    grid_logical = (env.grid.astype(np.uint8) > 0)
    obs_idx = np.argwhere(grid_logical)
    obs_centers = np.stack([obs_idx[:, 2], obs_idx[:, 1], obs_idx[:, 0]], axis=1).astype(np.float32)
    savemat("grid_world.mat", {
        "grid": grid_logical,
        "obs_centers": obs_centers,
        "size_x": env.size_x, "size_y": env.size_y, "size_z": env.size_z,
        "start": np.array(env.start, dtype=np.int32),
        "goal": np.array(env.goal, dtype=np.int32),
        "path_raw": np.array(path, dtype=np.int32),
        "path_smooth": path_smooth.astype(np.float32),
        "replanned_path": (replanned_once_path if replanned_once_path is not None else np.array([], dtype=float))
    })

    try:
        animate_path(env, path_raw=path, path_smooth=path_smooth,
                     out_gif="test.gif",
                     out_mp4=("test.mp4" if use_ffmpeg else None),
                     anim_stride=2, interval_ms=300)
        save_static_path(env, path_raw=path, path_smooth=path_smooth, png_path="ruins_path.png", dpi=140)
    except Exception as e:
        print(f"⚠️ Animation/plot failed: {e}")

    print("show_interactive =", show_interactive)
    if show_interactive:
        show_interactive_mpl(env, np.asarray(path, dtype=float), path_smooth)

    print(f"Test reward: {total:.2f} | Raw length: {len(path)} | Smooth length: {len(path_smooth)}")
    try:
        import csv, os
        if rewards is not None:
            csv_path = os.path.join(".", "reward.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "reward"])
                for i, R in enumerate(rewards):
                    w.writerow([i, float(R)])
        else:
            if bm.cfg.verbose:
                print("[Test] No rewards provided — skipping reward.csv save.")
    except Exception as e:
        print(f"⚠️ Save reward.csv failed: {e}")
    return path_smooth, float(total)


# ================= BehaviorManager & RobotSimulator =================

class BehaviorManagerConfig:
    def __init__(
        self,
        check_interval=1,
        max_replan_trials=2,
        verbose=True
    ):
        self.check_interval = int(check_interval)
        self.max_replan_trials = int(max_replan_trials)
        self.verbose = bool(verbose)


class BehaviorManager:
    def __init__(self, env, agent, config: BehaviorManagerConfig = None):
        self.env = env
        self.agent = agent
        self.cfg = config or BehaviorManagerConfig()
        self._baseline_occ = None
        self._last_check_step = -1
        self._replan_count = 0
        self._waypoint_reached_last = False

    def set_baseline_from_env(self):
        g = self.env.grid.astype(np.uint8) > 0
        self._baseline_occ = g.copy()
        self._waypoint_reached_last = bool(getattr(self.env, "wp_reached", False))
        if self.cfg.verbose:
            print("[Behavior] Baseline occupancy snapshot taken from current env.")

    def replan_from_start(self, max_steps=3000, reset_waypoint=True):
        env = self.env
        saved = {
            "pos": tuple(env.pos),
            "wp_reached": bool(getattr(env, "wp_reached", False)),
            "steps": int(getattr(env, "steps", 0)),
            "last_move_dir": getattr(env, "last_move_dir", None),
            "prev_action": getattr(env, "prev_action", None),
            "best_dist_to_wp": getattr(env, "best_dist_to_wp", None),
            "best_dist_to_goal": getattr(env, "best_dist_to_goal", None),
            "steps_since_wp": getattr(env, "steps_since_wp", None),
        }

        try:
            env.pos = tuple(env.start)
            env.steps = 0
            if reset_waypoint:
                env.wp_reached = False
            env.last_move_dir = None
            env.prev_action = None
            try:
                env.best_dist_to_wp = env._distance_to_target(env.start, env.waypoint)
                env.best_dist_to_goal = float('inf')
                env.steps_since_wp = 0
            except Exception:
                pass

            local_path = [tuple(env.pos)]
            s = env._norm_state(env.pos)
            local_steps = 0
            for _ in range(max_steps):
                prev_pos = tuple(env.pos)
                a = self.agent.select_action(s, eps=0.0)
                ns, r, terminated, truncated, _ = env.step(a)
                s = ns
                local_steps += 1
                detect_and_print_mode_change(prev_pos, env.pos, step_idx=local_steps)

                local_path.append(tuple(env.pos))
                if bool(terminated) or bool(truncated):
                    break

            return np.asarray(local_path, dtype=float)

        finally:
            env.pos = saved["pos"]
            env.wp_reached = saved["wp_reached"]
            env.steps = saved["steps"]
            env.last_move_dir = saved["last_move_dir"]
            env.prev_action = saved["prev_action"]
            if saved["best_dist_to_wp"] is not None:
                env.best_dist_to_wp = saved["best_dist_to_wp"]
            if saved["best_dist_to_goal"] is not None:
                env.best_dist_to_goal = saved["best_dist_to_goal"]
            if saved["steps_since_wp"] is not None:
                env.steps_since_wp = saved["steps_since_wp"]

    def set_baseline_from_snapshot(self, snapshot):
        snap = np.asarray(snapshot, dtype=bool)
        if snap.ndim != 3:
            raise ValueError("baseline snapshot must have shape (Z, Y, X)")
        self._baseline_occ = snap.copy()
        self._waypoint_reached_last = bool(getattr(self.env, "wp_reached", False))
        if self.cfg.verbose:
            print("[Behavior] Baseline set from provided snapshot.")

    def _extract_current_occ(self):
        return (self.env.grid.astype(np.uint8) > 0)

    def _has_any_change(self):
        if self._baseline_occ is None:
            return False
        cur = self._extract_current_occ()
        if cur.shape != self._baseline_occ.shape:
            return True
        changed = np.any(np.logical_xor(cur, self._baseline_occ))
        return bool(changed)

    def should_check(self, step_idx):
        return (self._last_check_step < 0) or ((step_idx - self._last_check_step) >= self.cfg.check_interval)

    def need_replan(self, step_idx):
        if self._replan_count >= self.cfg.max_replan_trials:
            return False
        self._last_check_step = step_idx
        any_change = self._has_any_change()
        if self.cfg.verbose and any_change:
            print(f"[Behavior] any_change={any_change}")
        return any_change

    def mark_replanned(self, step_idx):
        self._replan_count += 1
        self._baseline_occ = (self.env.grid.astype(np.uint8) > 0).copy()
        if self.cfg.verbose:
            print(f"[Behavior] Replan marked. Total replans={self._replan_count}")

    def replan_from_here(self, max_steps=3000):
        env = self.env
        cur_pos = tuple(getattr(env, "pos"))
        cur_wp_reached = bool(getattr(env, "wp_reached", False))
        cur_steps = int(getattr(env, "steps", 0))

        # 保存 waypoint，用于重规划期间“直达终点”并在结束后恢复
        saved = {
            "pos": env.pos,
            "wp_reached": env.wp_reached,
            "steps": env.steps,
            "waypoint": getattr(env, "waypoint", None),
        }

        try:
            env.pos = cur_pos
            env.wp_reached = cur_wp_reached
            env.steps = cur_steps

            # 重规划期间只看终点
            env.wp_reached = True
            if hasattr(env, "waypoint"):
                env.waypoint = env.goal

            local_path = [cur_pos]
            s = env._norm_state(cur_pos)
            total_steps = 0
            while total_steps < max_steps:
                prev_pos = tuple(env.pos)
                a = self.agent.select_action(s, eps=0.0)
                ns, r, terminated, truncated, _ = env.step(a)
                s = ns
                total_steps += 1
                detect_and_print_mode_change(prev_pos, env.pos, step_idx=total_steps)

                local_path.append(tuple(env.pos))
                if bool(terminated) or bool(truncated):
                    break

            return np.array(local_path, dtype=float)

        finally:
            env.pos = saved["pos"]
            env.wp_reached = saved["wp_reached"]
            env.steps = saved["steps"]
            # 恢复 waypoint（如果存在）
            if "waypoint" in saved and saved["waypoint"] is not None and hasattr(env, "waypoint"):
                env.waypoint = saved["waypoint"]


class RobotSimulator:
    def __init__(
        self,
        env,
        agent,
        baseline_snapshot,
        detect_radius=5,
        wait_ticks=5,
        verbose=True,
        trigger_dist=0.5,
        enable_deviation_check=True,
        enable_replan=True
    ):
        """
        RobotSimulator（更新版）：
        - 在飞行阶段引入一次“确定性”的风扰（只发生一次，位置和步数预先设定）；
        - 当偏移超过阈值（trigger_dist）时，可由决策层触发一次从当前位置到目标点的重规划。
        通过 enable_deviation_check / enable_replan 控制是否检测偏离和是否重规划：
        - enable_deviation_check=False, enable_replan=False -> 只按原路径开环 + 风扰，一般无法回到目标；
        - enable_deviation_check=True, enable_replan=True  -> 偏离后触发一次 DDQN 重规划。
        """
        self.env = env
        self.agent = agent
        self.verbose = bool(verbose)
        self.wait_ticks = int(wait_ticks)

        self.enable_deviation_check = bool(enable_deviation_check)
        self.enable_replan = bool(enable_replan)

        # 偏离触发重规划的阈值：当前位置到参考路径的最近距离 > deviation_threshold
        self.deviation_threshold = float(trigger_dist)

        # ---------------- 风扰相关（确定性 + 只触发一次）----------------
        self.wind_enabled = True
        self.wind_step = 90            # 在第 wind_step 个仿真步触发风扰
        self.wind_delta = np.asarray((2, 2, -2), dtype=int)  # 固定位移 (dx, dy, dz)
        self.wind_applied = False      # 是否已经施加过风扰

        # BehaviorManager 负责执行“从当前位置重规划”逻辑
        max_trials = 1 if self.enable_replan else 0
        self.bm = BehaviorManager(
            env,
            agent,
            BehaviorManagerConfig(check_interval=1, max_replan_trials=max_trials, verbose=verbose)
        )
        # 为保持接口兼容，仍允许从 snapshot 设置 baseline（但不再用于仿真中的障碍物检测）
        try:
            self.bm.set_baseline_from_snapshot(baseline_snapshot)
        except Exception:
            pass

    # ---------- 内部工具：根据路径构造每段方向增量 ----------
    def _build_seg_dirs(self, path_xyz):
        """
        根据给定的离线路径 path_xyz (N,3)，构造每一段的方向增量数组 seg_dirs (N,3)：
        第 k 段的方向 = clip(path[k+1] - path[k], -1, 1)
        最后一段设为 (0,0,0) 表示不再前进。
        """
        path_int = np.asarray(path_xyz, dtype=int)
        if path_int.ndim != 2 or path_int.shape[0] == 0:
            raise ValueError("path for seg_dirs must be non-empty (N,3)")
        seg_dirs = np.zeros_like(path_int, dtype=int)
        for k in range(len(path_int) - 1):
            delta = path_int[k + 1] - path_int[k]
            seg_dirs[k] = np.clip(delta, -1, 1)
        seg_dirs[-1] = np.array((0, 0, 0), dtype=int)
        return seg_dirs

    # ---------- 内部工具：位置到路径的距离 ----------
    def _distance_to_path(self, pos_xyz, path_xyz):
        """返回 pos_xyz 到给定离散路径 path_xyz 的最近欧氏距离。"""
        pts = np.asarray(path_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return 0.0
        p = np.asarray(pos_xyz, dtype=float)[None, :]
        d2 = np.sum((pts - p) ** 2, axis=1)
        return float(np.sqrt(d2.min()))

    # ---------- 内部工具：风扰（确定性，只触发一次） ----------
    def _apply_wind(self, pos_xyz, step_idx):
        """
        在给定的仿真步数 step_idx 触发一次确定性风扰：
        - 仅当：wind_enabled=True 且 wind_applied=False 且 step_idx == wind_step 时生效；
        - 采用固定位移 self.wind_delta；
        - 若扰动后越界则裁剪；若落在障碍物上，则保持原位置不变。
        """
        x, y, z = map(int, pos_xyz)

        # 未启用 / 已经施加过风扰 -> 不再改变位置
        if (not self.wind_enabled) or self.wind_applied:
            return (x, y, z)

        # 只在指定的步数触发
        if int(step_idx) != int(self.wind_step):
            return (x, y, z)

        # 只在空中触发（如不需要这个限制可删除下面两行）
        if z <= 0:
            return (x, y, z)

        before = (x, y, z)

        dx, dy, dz = map(int, self.wind_delta)
        nx = x + dx
        ny = y + dy
        nz = z + dz

        # 边界裁剪
        nx = int(np.clip(nx, 0, self.env.size_x - 1))
        ny = int(np.clip(ny, 0, self.env.size_y - 1))
        nz = int(np.clip(nz, 0, self.env.size_z - 1))

        # 若扰动后落在障碍物上，则放弃扰动（但打印说明）
        if self.env.grid[nz, ny, nx] > 0:
            after = (x, y, z)
            print(f"[Wind] 尝试在 step={step_idx} 施加风扰，但目标位置是障碍物，保持原位 {after}")
            self.wind_applied = True
            return after

        after = (nx, ny, nz)
        # 更新环境位置
        self.env.pos = after
        self.wind_applied = True

        print(f"[Wind] disturbance applied at step={step_idx}, pos_before={before}, pos_after={after}")
        print(f"[Wind] 步数={step_idx}, 扰动后当前位置={after}")

        return after

    # ---------- 主仿真逻辑：开环执行路径增量 + 风扰 + （可选）偏离检测/重规划 ----------
    def simulate_follow_with_replan(self, path_initial, max_ticks=100000):
        """
        给定一条预先规划好的路径 path_initial（N,3），机器人尝试沿该路径运动：
        - 使用“开环路径增量”：每一步只执行离线路径当前段的增量方向；
        - 在飞行阶段固定某一步引入一次风扰（self.wind_step + self.wind_delta）；
        - 若 enable_deviation_check=True 且 enable_replan=True：
            当偏离超过阈值时触发一次基于 DDQN 的从当前位置到目标点的重规划，
            并用新路径替换当前执行路径。
        - 若 enable_deviation_check=False 或 enable_replan=False：
            不进行偏离检测 / 不触发重规划，只是沿原路径 + 风扰前进，
            通常会因为风扰而无法到达目标点。
        """
        path_initial = np.asarray(path_initial, dtype=float)
        if path_initial.ndim != 2 or path_initial.shape[1] != 3 or len(path_initial) == 0:
            raise ValueError("path_initial must be (N,3) non-empty")

        exec_path = path_initial.copy()          # 当前参考路径（可在重规划后替换）
        seg_dirs = self._build_seg_dirs(exec_path)  # 与 exec_path 一一对应的方向增量
        i = 0                                    # 当前参考路径段的索引
        ticks = 0
        traj = [tuple(self.env.pos)]

        # 起点对齐：确保环境中的当前位置与参考路径起点一致
        if tuple(self.env.pos) != tuple(exec_path[0].astype(int)):
            self.env.pos = tuple(exec_path[0].astype(int))

        while ticks < max_ticks:
            # 到达目标点判定（与环境中的目标半径一致）
            d_goal = self.env._distance_to_target(self.env.pos, self.env.goal)
            if d_goal <= 0.5:
                if self.verbose:
                    print(f"[Sim] Goal reached at tick={ticks}, pos={self.env.pos}")
                break

            # 当前段的参考运动方向（来自离线路径，只看 path，不看当前位置）
            dir_vec = seg_dirs[i]
            cx, cy, cz = map(int, self.env.pos)
            dx, dy, dz = map(int, dir_vec)
            nx, ny, nz = cx + dx, cy + dy, cz + dz

            prev_pos = tuple(self.env.pos)

            # 边界和障碍检查：如果该“方向增量”在当前实际位置下会撞障碍，就停在原地
            if (0 <= nx < self.env.size_x and
                0 <= ny < self.env.size_y and
                0 <= nz < self.env.size_z and
                self.env.grid[nz, ny, nx] == 0):
                self.env.pos = (nx, ny, nz)
                moved = (dx != 0 or dy != 0 or dz != 0)
                if moved and i < len(exec_path) - 2:
                    # 只要沿该段方向走了一步，就推进到下一段，
                    # 不要求“回到某个路径点”，因此风扰后不会被吸回原始路径。
                    i += 1
            else:
                # 当前段方向在此实际位置无法前进，则原地等待（不推进路径索引）
                moved = False

            # 计步 + 在当前步尝试触发“确定性风扰”
            ticks += 1
            self.env.pos = self._apply_wind(self.env.pos, ticks)

            detect_and_print_mode_change(prev_pos, self.env.pos, step_idx=ticks)
            traj.append(self.env.pos)

            # ---------- 偏离检测 & 重规划（由开关控制） ----------
            if self.enable_deviation_check and self.enable_replan:
                deviation = self._distance_to_path(self.env.pos, exec_path)

                if (deviation > self.deviation_threshold
                    and self.bm._replan_count < self.bm.cfg.max_replan_trials):

                    if self.verbose:
                        print(f"[Sim] Path deviation detected: dist={deviation:.2f} > threshold={self.deviation_threshold:.2f}")
                        print(f"[Sim] Trigger replan at tick={ticks}, wait={self.wait_ticks}")

                    # 等待一小段时间
                    for _ in range(self.wait_ticks):
                        traj.append(tuple(self.env.pos))
                        ticks += 1
                        if ticks >= max_ticks:
                            break
                    if ticks >= max_ticks:
                        break

                    # 从当前位置重规划
                    new_path = self.bm.replan_from_here(max_steps=self.env.max_steps)
                    self.bm.mark_replanned(step_idx=ticks)

                    if new_path is not None and len(new_path) > 0:
                        new_path = np.asarray(new_path, dtype=float)
                        # 确保新路径首点就是当前实际位姿
                        if not np.allclose(new_path[0], np.asarray(self.env.pos), atol=1e-6):
                            new_path[0] = np.asarray(self.env.pos, dtype=float)
                        exec_path = new_path
                        seg_dirs = self._build_seg_dirs(exec_path)  # 重建方向增量
                        i = 0                                       # 从新路径的起点重新开始
                        if self.verbose:
                            print(f"[Sim] Replanned path length = {len(exec_path)}")

            # 实时平滑并保存轨迹（失败不终止仿真）
            try:
                traj_smooth = smooth_path_safe(
                    self.env,
                    traj,
                    clearance=2.0,
                    shortcut_iters=600,
                    fillet=True,
                    fillet_scale=0.8,
                    cr_alpha=0.25,
                    cr_samples=12,
                    cr_tangent_cap=0.7
                )
                np.savetxt("traj_smooth.csv", traj_smooth, fmt="%.3f", delimiter=",", header="x,y,z", comments="")
            except Exception as e:
                if self.verbose:
                    print(f"[Sim] Path smoothing failed: {e}")

        return np.asarray(traj, dtype=int)

    # 保持与原版本完全一致，不做改动
    def _step_towards(self, current, target):
        cx, cy, cz = map(int, current)
        tx, ty, tz = map(int, target)

        dx = np.clip(tx - cx, -1, 1)
        dy = np.clip(ty - cy, -1, 1)
        dz = np.clip(tz - cz, -1, 1)

        nx, ny, nz = cx + dx, cy + dy, cz + dz

        if not (0 <= nx < self.env.size_x and 0 <= ny < self.env.size_y and 0 <= nz < self.env.size_z):
            return (cx, cy, cz), False
        if self.env.grid[nz, ny, nx] > 0:
            return (cx, cy, cz), False

        return (nx, ny, nz), (dx != 0 or dy != 0 or dz != 0)
