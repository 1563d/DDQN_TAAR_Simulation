"""Publication-oriented ablation and multi-seed evaluation suite.

Experiments
-----------
Training ablation:
    1. DDQN                 (uniform replay, no curriculum)
    2. DDQN+PER             (PER, no curriculum)
    3. DDQN+CL              (uniform replay, curriculum)
    4. DDQN+PER+CL          (PER, curriculum; proposed training strategy)

Replanning ablation (uses the DDQN+PER+CL checkpoint for each seed):
    1. No replanning
    2. Periodic replanning
    3. Event-triggered replanning

Outputs include seed-level raw data, mean/SD/95% CI summaries, paired
significance tests, publication-ready figures, and LaTeX tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_planner.env import GridWorld3DEnv
from path_planner.train_eval import ImprovedCurriculumConfig, ImprovedAdaptiveCurriculumRunner, _apply_curriculum_obstacles
from experiments.configurable_agent import AgentConfig, ConfigurableDQNAgent


VARIANTS = {
    "DDQN": dict(use_per=False, use_cl=False),
    "DDQN_PER": dict(use_per=True, use_cl=False),
    "DDQN_CL": dict(use_per=False, use_cl=True),
    "DDQN_PER_CL": dict(use_per=True, use_cl=True),
}
DISPLAY_NAMES = {
    "DDQN": "DDQN",
    "DDQN_PER": "DDQN+PER",
    "DDQN_CL": "DDQN+CL",
    "DDQN_PER_CL": "DDQN+PER+CL",
    "no_replan": "No replanning",
    "periodic": "Periodic replanning",
    "event": "Event-triggered replanning",
}


@dataclass
class ExperimentConfig:
    size_x: int = 60
    size_y: int = 80
    size_z: int = 27
    episodes: int = 400
    seeds: Tuple[int, ...] = tuple(range(20))
    final_return_window: int = 100
    moving_average_window: int = 20
    # Legacy reward-curve convergence fields are retained only for backward
    # config compatibility; publication convergence is computed from the fixed
    # full-map greedy evaluation below.
    convergence_hold: int = 10
    convergence_band: float = 0.15
    fixed_eval_interval: int = 20
    fixed_eval_hold: int = 3
    fixed_eval_seed: int = 20260810
    # Curriculum v3: short adaptive warm-up phases. Each non-final phase must
    # be seen for at least 20 episodes, is checked every 10 episodes, and is
    # force-advanced by 50 episodes. The final full-complexity phase consumes
    # all remaining episodes.
    curriculum_min_episodes: Tuple[int, ...] = (20, 20, 20, 20, 0)
    curriculum_max_episodes: Tuple[int | None, ...] = (50, 50, 50, 50, None)
    curriculum_eval_interval: int = 10
    curriculum_mastery_hold: int = 2
    protocol_version: str = "paper_ablation_v3_adaptive_curriculum"
    device: str = "auto"
    periodic_interval: int = 20
    event_check_interval: int = 1
    max_replans_periodic: int = 20
    max_replans_event: int = 3
    dynamic_inject_fraction: float = 0.45
    dynamic_obstacle_lead: int = 10
    dynamic_cube_radius_xy: int = 1
    dynamic_cube_radius_z: int = 1
    execution_tick_factor: float = 1.5
    replanning_scenario: str = "deviation"
    wind_air_fraction: float = 0.85
    wind_dx: int = 2
    wind_dy: int = 2
    wind_dz: int = -2
    deviation_threshold: float = 0.5

    @property
    def max_steps(self) -> int:
        return 5 * (self.size_x + self.size_y + self.size_z)

    @property
    def max_execution_ticks(self) -> int:
        return int(math.ceil(self.max_steps * self.execution_tick_factor))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def make_env(cfg: ExperimentConfig, seed: int | None = None) -> GridWorld3DEnv:
    return GridWorld3DEnv(
        size_x=cfg.size_x,
        size_y=cfg.size_y,
        size_z=cfg.size_z,
        max_steps=cfg.max_steps,
        cube1=None,
        cube2=None,
        seed=seed,
    )


def make_agent(env: GridWorld3DEnv, cfg: ExperimentConfig, use_per: bool):
    return ConfigurableDQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        device=resolve_device(cfg.device),
        use_per=use_per,
        config=AgentConfig(),
    )


def curriculum_phase_for_episode(ep: int, total_episodes: int) -> int:
    # Fixed 5-stage schedule (same total training budget across all variants).
    # This follows the five difficulty maps in the current project and avoids
    # adding extra evaluation episodes to CL variants, which keeps the ablation
    # computational budget directly comparable.
    phase_len = max(1, int(math.ceil(total_episodes / 5.0)))
    return min(4, ep // phase_len)


def train_variant(
    cfg: ExperimentConfig,
    variant: str,
    seed: int,
    out_dir: Path,
    quiet: bool = False,
) -> Tuple[ConfigurableDQNAgent, List[float], Dict]:
    """Train one PER/CL ablation variant under protocol v3.

    All variants use exactly the same reward function.  CL changes only the
    environment-difficulty schedule.  In addition, every variant is greedily
    evaluated at the same episode checkpoints on the same full-complexity map;
    those inference-only evaluations are not written to replay memory.
    """
    # Resolve once before the curriculum runner temporarily changes CWD.
    # Without this, a relative results path (e.g. paper_ablation_results_final20/...)
    # is interpreted a second time from inside the seed directory during the
    # fixed-evaluation callback, producing a non-existent nested parent path.
    out_dir = Path(out_dir).expanduser().resolve()

    spec = VARIANTS[variant]
    seed_everything(seed)
    env = make_env(cfg, seed=seed)
    agent = make_agent(env, cfg, use_per=spec["use_per"])

    rewards: List[float] = []
    losses: List[float] = []
    episode_steps: List[float] = []
    fixed_eval_records: List[Dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_eval_path = out_dir / "fixed_eval_curve.csv"

    # Episode-0 baseline, identical final-complexity validation environment.
    append_fixed_eval_record(cfg, agent, 0, fixed_eval_records, fixed_eval_path)

    def _fixed_eval_callback(episode: int, current_agent):
        append_fixed_eval_record(
            cfg, current_agent, episode, fixed_eval_records, fixed_eval_path
        )

    t0 = time.perf_counter()

    if spec["use_cl"]:
        # Adaptive capped warm-up curriculum.  The legacy equal 80/80/80/80/80
        # allocation could spend most of the training budget on easy maps.
        # Protocol v3 guarantees at least 200/400 episodes on the final full map
        # even if no early phase is mastered, while stable mastery can reserve
        # substantially more final-map training.
        curriculum = ImprovedCurriculumConfig(
            total_episodes=cfg.episodes,
            phase_episodes=(50, 50, 50, 50, max(0, cfg.episodes - 200)),
            phase_min_episodes=cfg.curriculum_min_episodes,
            phase_max_episodes=cfg.curriculum_max_episodes,
            mastery_consecutive_checks=cfg.curriculum_mastery_hold,
        )
        old_cwd = os.getcwd()
        try:
            os.chdir(out_dir)
            runner = ImprovedAdaptiveCurriculumRunner(
                env=env,
                agent=agent,
                curriculum=curriculum,
                batch_train_episodes=cfg.curriculum_eval_interval,
                # The phase gate is deterministic (fixed map + greedy policy);
                # one rollout is sufficient. Stability is enforced by two
                # consecutive checks, while publication evaluation remains the
                # separate fixed-full-map protocol.
                eval_trials_per_check=1,
                max_total_episodes=cfg.episodes,
                seed=seed,
                verbose=not quiet,
                episode_end_callback=_fixed_eval_callback,
            )
            agent, rewards, _ = runner.run()
            phase_rewards = runner.phase_rewards
            phase_success_rates = runner.phase_success_rates
            phase_history = runner.phase_history
            pd.DataFrame(phase_history).to_csv(out_dir / "curriculum_schedule.csv", index=False)
        finally:
            os.chdir(old_cwd)
        losses = [float("nan")] * len(rewards)
        episode_steps = [float("nan")] * len(rewards)
    else:
        phase_rewards = None
        phase_success_rates = None
        phase_history = None
        eps_start, eps_end = 1.0, 0.05
        eps_decay_steps = max(int(0.6 * cfg.episodes), 1)
        for ep in range(cfg.episodes):
            state, _ = env.reset(seed=seed if ep == 0 else None)
            frac = min(1.0, ep / eps_decay_steps)
            eps = eps_start + (eps_end - eps_start) * frac
            total_reward = 0.0
            steps = 0
            last_loss = None

            for _ in range(cfg.max_steps):
                action = agent.select_action(state, eps)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated) or bool(truncated)
                agent.buf.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += float(reward)
                steps += 1
                loss = agent.step_and_maybe_update()
                if loss is not None:
                    last_loss = loss
                if done:
                    break

            for _ in range(5):
                loss = agent.update()
                if loss is not None:
                    last_loss = loss

            rewards.append(total_reward)
            episode_steps.append(steps)
            losses.append(float("nan") if last_loss is None else float(last_loss))
            _fixed_eval_callback(ep + 1, agent)

            if (not quiet) and ((ep + 1) % 20 == 0 or ep == 0 or ep + 1 == cfg.episodes):
                recent = np.mean(rewards[-20:])
                print(
                    f"[{variant} seed={seed:02d}] ep={ep+1:4d}/{cfg.episodes} "
                    f"eps={eps:.3f} R20={recent:.2f}"
                )

    # Guarantee a final fixed-map evaluation if a custom episode count is not an
    # exact multiple of fixed_eval_interval.
    append_fixed_eval_record(cfg, agent, cfg.episodes, fixed_eval_records, fixed_eval_path)

    training_seconds = time.perf_counter() - t0
    ckpt_path = out_dir / "checkpoint.pth"
    torch.save(
        agent.checkpoint(
            variant=variant,
            seed=seed,
            episodes=cfg.episodes,
            rewards=list(map(float, rewards)),
            episode_steps=episode_steps,
            training_seconds=training_seconds,
            phase_rewards=phase_rewards,
            phase_success_rates=phase_success_rates,
            phase_history=phase_history,
            fixed_eval_records=fixed_eval_records,
            protocol_version=cfg.protocol_version,
            reward_protocol="paper_v2_progress_only",
            curriculum_protocol="adaptive_capped_warmup_v3",
        ),
        ckpt_path,
    )

    n = len(rewards)
    pd.DataFrame({
        "episode": np.arange(1, n + 1),
        "reward": rewards,
        "loss": losses[:n],
        "steps": episode_steps[:n],
    }).to_csv(out_dir / "training_curve.csv", index=False)

    meta = {
        "variant": variant,
        "seed": seed,
        "training_seconds": training_seconds,
        "checkpoint": str(ckpt_path),
        "uses_existing_adaptive_curriculum_runner": bool(spec["use_cl"]),
        "protocol_version": cfg.protocol_version,
        "reward_protocol": "paper_v2_progress_only",
        "curriculum_protocol": "adaptive_capped_warmup_v3" if spec["use_cl"] else "none",
        "curriculum_min_episodes": list(cfg.curriculum_min_episodes),
        "curriculum_max_episodes": list(cfg.curriculum_max_episodes),
        "curriculum_eval_interval": cfg.curriculum_eval_interval,
        "curriculum_mastery_hold": cfg.curriculum_mastery_hold,
        "fixed_eval_interval": cfg.fixed_eval_interval,
        "fixed_eval_hold": cfg.fixed_eval_hold,
        "fixed_eval_seed": cfg.fixed_eval_seed,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return agent, list(map(float, rewards)), meta


def load_agent_from_checkpoint(cfg: ExperimentConfig, checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    env = make_env(cfg)
    agent = make_agent(env, cfg, use_per=bool(payload.get("use_per", True)))
    agent.load_checkpoint(payload, load_optimizer=False)
    return agent, payload


def rollout_policy(
    env: GridWorld3DEnv,
    agent: ConfigurableDQNAgent,
    max_steps: int,
    start_pos: Tuple[int, int, int] | None = None,
    wp_reached: bool | None = None,
):
    if start_pos is None:
        state, _ = env.reset()
    else:
        env.pos = tuple(map(int, start_pos))
        env.steps = 0
        if wp_reached is not None:
            env.wp_reached = bool(wp_reached)
        env.prev_action = None
        env.last_move_dir = None
        env._refresh_padded_grid()
        state = env._norm_state(env.pos)

    path = [tuple(env.pos)]
    total_reward = 0.0
    collision_attempts = 0
    mode_switches = 0
    prev_z = int(env.pos[2])

    for _ in range(int(max_steps)):
        before = tuple(env.pos)
        action = agent.select_action(state, eps=0.0)
        dx, dy, dz = env.moves[action]
        nx, ny, nz = before[0] + dx, before[1] + dy, before[2] + dz
        blocked = (
            not (0 <= nx < env.size_x and 0 <= ny < env.size_y and 0 <= nz < env.size_z)
            or (0 <= nx < env.size_x and 0 <= ny < env.size_y and 0 <= nz < env.size_z and env.grid[nz, ny, nx] > 0)
        )
        if blocked:
            collision_attempts += 1

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        path.append(tuple(env.pos))
        z = int(env.pos[2])
        if (prev_z == 0 and z > 0) or (prev_z > 0 and z == 0):
            mode_switches += 1
        prev_z = z
        if terminated or truncated:
            break

    success = bool(getattr(env, "wp_reached", False)) and env._distance_to_target(env.pos, env.goal) <= 0.5
    arr = np.asarray(path, dtype=float)
    path_length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()) if len(arr) > 1 else 0.0
    return {
        "success": int(success),
        "reward": total_reward,
        "steps": len(path) - 1,
        "path_length": path_length,
        "collision_attempts": collision_attempts,
        "mode_switches": mode_switches,
        "final_distance": env._distance_to_target(env.pos, env.goal),
        "path": arr,
    }



def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def fixed_full_map_eval(cfg: ExperimentConfig, agent: ConfigurableDQNAgent) -> Dict:
    """Greedy evaluation on one fixed full-complexity validation environment.

    The RNG state is restored afterward so publication evaluation does not alter
    subsequent exploration/replay sampling.  Therefore all variants receive the
    same number of *training* interactions, and CL is compared to non-CL on the
    same final-complexity map throughout training.
    """
    rng_state = _capture_rng_state()
    try:
        env = make_env(cfg, seed=cfg.fixed_eval_seed)
        result = rollout_policy(env, agent, cfg.max_steps)
    finally:
        _restore_rng_state(rng_state)
    return result


def should_fixed_eval(cfg: ExperimentConfig, episode: int) -> bool:
    episode = int(episode)
    return episode == 0 or episode == cfg.episodes or (
        cfg.fixed_eval_interval > 0 and episode % cfg.fixed_eval_interval == 0
    )


def append_fixed_eval_record(
    cfg: ExperimentConfig,
    agent: ConfigurableDQNAgent,
    episode: int,
    records: List[Dict],
    out_path: Path,
):
    if not should_fixed_eval(cfg, episode):
        return
    result = fixed_full_map_eval(cfg, agent)
    row = {
        "episode": int(episode),
        "success": int(result["success"]),
        "reward": float(result["reward"]),
        "path_length": float(result["path_length"]),
        "steps": int(result["steps"]),
        "collision_attempts": int(result["collision_attempts"]),
        "mode_switches": int(result["mode_switches"]),
        "final_distance": float(result["final_distance"]),
    }
    # Avoid accidental duplicate checkpoints (e.g., episode==final and divisible
    # by interval) while preserving deterministic order.
    if records and int(records[-1]["episode"]) == int(episode):
        records[-1] = row
    else:
        records.append(row)
    pd.DataFrame(records).to_csv(out_path, index=False)


def fixed_eval_convergence(eval_curve: pd.DataFrame, hold: int, max_episode: int) -> Tuple[float, int]:
    """Return stable-convergence onset and a convergence flag.

    Non-converged runs are right-censored and return NaN for the descriptive
    convergence episode.  They are represented separately by
    ``fixed_eval_converged=0`` and by the success-AUC metric, rather than being
    incorrectly averaged as an artificial episode 401.
    """
    if eval_curve.empty:
        return float("nan"), 0
    cur = eval_curve.sort_values("episode").reset_index(drop=True)
    cur = cur[cur["episode"] > 0].reset_index(drop=True)
    h = max(1, int(hold))
    success = cur["success"].to_numpy(dtype=int)
    episodes = cur["episode"].to_numpy(dtype=int)
    for i in range(0, len(success) - h + 1):
        if np.all(success[i:i + h] == 1):
            return float(episodes[i]), 1
    return float("nan"), 0


def fixed_eval_success_auc(eval_curve: pd.DataFrame, max_episode: int) -> float:
    if eval_curve.empty:
        return 0.0
    cur = eval_curve.sort_values("episode")
    x = cur["episode"].to_numpy(dtype=float)
    y = cur["success"].to_numpy(dtype=float)
    if x[0] > 0:
        x = np.insert(x, 0, 0.0); y = np.insert(y, 0, 0.0)
    if x[-1] < max_episode:
        x = np.append(x, float(max_episode)); y = np.append(y, y[-1])
    area = float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
    return area / max(float(max_episode), 1.0)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if len(x) < window:
        return np.asarray([], dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="valid")


def convergence_episode(
    rewards: Sequence[float],
    ma_window: int,
    hold: int,
    band_fraction: float,
) -> int:
    ma = moving_average(rewards, ma_window)
    if len(ma) < hold:
        return len(rewards) + 1
    final_value = float(ma[-1])
    tol = band_fraction * max(abs(final_value), 1.0)
    lo, hi = final_value - tol, final_value + tol
    for i in range(0, len(ma) - hold + 1):
        block = ma[i:i + hold]
        if np.all((block >= lo) & (block <= hi)):
            # moving-average index i corresponds to original episode i+ma_window
            return int(i + ma_window)
    return len(rewards) + 1


def training_seed_metrics(cfg: ExperimentConfig, variant: str, seed: int, seed_dir: Path) -> Dict:
    curve = pd.read_csv(seed_dir / "training_curve.csv")
    rewards = curve["reward"].to_numpy(dtype=float)
    fixed_path = seed_dir / "fixed_eval_curve.csv"
    if not fixed_path.exists():
        raise FileNotFoundError(
            f"Missing {fixed_path}. This result was produced by the legacy protocol; "
            "retrain with protocol v3 before statistical analysis."
        )
    fixed = pd.read_csv(fixed_path).sort_values("episode").reset_index(drop=True)
    agent, payload = load_agent_from_checkpoint(cfg, seed_dir / "checkpoint.pth")
    if payload.get("protocol_version") != cfg.protocol_version:
        raise RuntimeError(
            f"Checkpoint protocol mismatch for {variant} seed={seed}: "
            f"{payload.get('protocol_version')} != {cfg.protocol_version}. Retrain this seed."
        )

    # Independent final greedy rollout retained for replanning-screening and
    # consistency checking; publication convergence comes from fixed_eval_curve.
    eval_env = make_env(cfg, seed=cfg.fixed_eval_seed)
    nominal = rollout_policy(eval_env, agent, cfg.max_steps)

    conv_ep, conv_flag = fixed_eval_convergence(
        fixed, cfg.fixed_eval_hold, cfg.episodes
    )
    auc = fixed_eval_success_auc(fixed, cfg.episodes)
    final_eval = fixed.iloc[-1]
    schedule_path = seed_dir / "curriculum_schedule.csv"
    final_phase_start = np.nan
    final_phase_episodes = np.nan
    if schedule_path.exists() and schedule_path.read_text(encoding="utf-8").strip():
        schedule = pd.read_csv(schedule_path)
        if not schedule.empty:
            last_phase = schedule.iloc[-1]
            final_phase_start = float(last_phase.get("start_episode", np.nan))
            final_phase_episodes = float(last_phase.get("episodes_used", np.nan))
    last_n = min(cfg.final_return_window, len(rewards))
    return {
        "variant": variant,
        "seed": seed,
        "final_return_mean100": float(np.mean(rewards[-last_n:])),
        "max_reward": float(np.max(rewards)),
        "reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "fixed_eval_convergence_episode": float(conv_ep),
        "fixed_eval_convergence_episode_capped": float(conv_ep) if conv_flag else float(cfg.episodes + cfg.fixed_eval_interval),
        "fixed_eval_converged": int(conv_flag),
        "fixed_eval_success_auc": float(auc),
        "fixed_eval_final_success": int(final_eval["success"]),
        "fixed_eval_final_reward": float(final_eval["reward"]),
        "fixed_eval_final_path_length": float(final_eval["path_length"]),
        "fixed_eval_final_steps": int(final_eval["steps"]),
        "fixed_eval_final_collision_attempts": int(final_eval["collision_attempts"]),
        "fixed_eval_final_distance": float(final_eval["final_distance"]),
        "nominal_success": nominal["success"],
        "nominal_reward": nominal["reward"],
        "nominal_path_length": nominal["path_length"],
        "nominal_steps": nominal["steps"],
        "nominal_collision_attempts": nominal["collision_attempts"],
        "nominal_mode_switches": nominal["mode_switches"],
        "nominal_final_distance": nominal["final_distance"],
        "curriculum_final_phase_start_episode": final_phase_start,
        "curriculum_final_phase_episodes": final_phase_episodes,
        "training_seconds": float(payload.get("training_seconds", np.nan)),
    }


def valid_cube_around_point(env: GridWorld3DEnv, point, r_xy: int, r_z: int):
    x, y, z = map(int, point)
    x0, x1 = max(0, x - r_xy), min(env.size_x, x + r_xy + 1)
    y0, y1 = max(0, y - r_xy), min(env.size_y, y + r_xy + 1)
    z0, z1 = max(0, z - r_z), min(env.size_z, z + r_z + 1)
    return {"min": (x0, y0, z0), "max": (x1, y1, z1), "value": 1}


def find_waypoint_index(path: np.ndarray, waypoint) -> int:
    if len(path) == 0:
        return 0
    d = np.linalg.norm(path - np.asarray(waypoint, dtype=float)[None, :], axis=1)
    indices = np.where(d <= 1.0)[0]
    return int(indices[0]) if len(indices) else 0


def plan_from_current_state(env, agent, max_steps: int):
    """Greedy planning rollout that restores the execution environment afterward."""
    saved = {
        "pos": tuple(env.pos),
        "wp_reached": bool(env.wp_reached),
        "steps": int(env.steps),
        "last_move_dir": env.last_move_dir,
        "prev_action": env.prev_action,
        "best_dist_to_wp": env.best_dist_to_wp,
        "best_dist_to_goal": env.best_dist_to_goal,
        "steps_since_wp": env.steps_since_wp,
    }
    try:
        state = env._norm_state(env.pos)
        path = [tuple(env.pos)]
        for _ in range(max_steps):
            action = agent.select_action(state, eps=0.0)
            state, _, terminated, truncated, _ = env.step(action)
            path.append(tuple(env.pos))
            if terminated or truncated:
                break
        return np.asarray(path, dtype=float)
    finally:
        env.pos = saved["pos"]
        env.wp_reached = saved["wp_reached"]
        env.steps = saved["steps"]
        env.last_move_dir = saved["last_move_dir"]
        env.prev_action = saved["prev_action"]
        env.best_dist_to_wp = saved["best_dist_to_wp"]
        env.best_dist_to_goal = saved["best_dist_to_goal"]
        env.steps_since_wp = saved["steps_since_wp"]
        env._refresh_padded_grid()


def build_segment_directions(path: np.ndarray):
    path = np.asarray(path, dtype=int)
    if len(path) == 0:
        return np.zeros((0, 3), dtype=int)
    dirs = np.zeros_like(path, dtype=int)
    for i in range(len(path) - 1):
        dirs[i] = np.clip(path[i + 1] - path[i], -1, 1)
    return dirs


def distance_to_path(pos, path: np.ndarray) -> float:
    pts = np.asarray(path, dtype=float)
    if len(pts) == 0:
        return 0.0
    p = np.asarray(pos, dtype=float)[None, :]
    return float(np.min(np.linalg.norm(pts - p, axis=1)))


def choose_late_aerial_tick(path: np.ndarray, air_fraction: float) -> int:
    """Choose a late aerial-path index, matching the project's wind-disturbance case.

    Triggering late in the aerial segment isolates the replanning layer: the
    nominal policy has already completed the ground waypoint and the event is
    an execution deviation, not a new global navigation problem.
    """
    path = np.asarray(path, dtype=float)
    air_idx = np.where(path[:, 2] > 0)[0]
    if len(air_idx) == 0:
        return max(1, min(len(path) - 2, int(0.8 * len(path))))
    q = float(np.clip(air_fraction, 0.0, 1.0))
    j = min(len(air_idx) - 1, max(0, int(round(q * (len(air_idx) - 1)))))
    return int(max(1, min(len(path) - 2, air_idx[j])))


def execute_deviation_replanning_benchmark(
    cfg: ExperimentConfig,
    agent: ConfigurableDQNAgent,
    seed: int,
    strategy: str,
) -> Dict:
    """Replanning ablation under a one-shot aerial path deviation.

    This reproduces the logic of the current project's wind-disturbance case:
    a deterministic displacement is applied once in the late aerial segment.
    - no_replan: continue the original open-loop segment commands;
    - periodic: replan at a fixed period irrespective of whether an event exists;
    - event: replan only when distance from the active reference path exceeds
      ``deviation_threshold``.
    """
    seed_everything(60_000 + seed)
    planning_env = make_env(cfg, seed=seed)
    initial = rollout_policy(planning_env, agent, cfg.max_steps)
    initial_path = initial["path"]
    if len(initial_path) < 4:
        raise RuntimeError(f"Initial path too short for seed {seed}")

    exec_env = make_env(cfg, seed=seed)
    exec_env.reset(seed=seed)
    exec_path = initial_path.copy()
    seg_dirs = build_segment_directions(exec_path)
    seg_idx = 0
    traj = [tuple(exec_env.pos)]
    replans = 0
    planning_seconds = 0.0
    collision_attempts = 0
    mode_switches = 0
    prev_z = int(exec_env.pos[2])
    wind_tick = choose_late_aerial_tick(initial_path, cfg.wind_air_fraction)
    wind_applied = False
    deviation_after_wind = 0.0

    for tick in range(1, cfg.max_execution_ticks + 1):
        if exec_env._distance_to_target(exec_env.pos, exec_env.goal) <= 0.5 and exec_env.wp_reached:
            break

        if (not exec_env.wp_reached) and exec_env._distance_to_target(exec_env.pos, exec_env.waypoint) <= 1.0:
            exec_env.wp_reached = True

        seg_idx = min(seg_idx, len(seg_dirs) - 1)
        dx, dy, dz = map(int, seg_dirs[seg_idx])
        cx, cy, cz = map(int, exec_env.pos)
        nx, ny, nz = cx + dx, cy + dy, cz + dz
        feasible = (
            0 <= nx < exec_env.size_x
            and 0 <= ny < exec_env.size_y
            and 0 <= nz < exec_env.size_z
            and exec_env.grid[nz, ny, nx] == 0
        )
        if feasible:
            exec_env.pos = (nx, ny, nz)
            if seg_idx < len(seg_dirs) - 1:
                seg_idx += 1
        else:
            collision_attempts += 1

        # One-shot deterministic disturbance, matching the current project.
        if (not wind_applied) and tick == wind_tick and exec_env.pos[2] > 0:
            candidate = np.asarray(exec_env.pos, dtype=int) + np.asarray(
                (cfg.wind_dx, cfg.wind_dy, cfg.wind_dz), dtype=int
            )
            candidate = np.clip(
                candidate,
                [0, 0, 0],
                [exec_env.size_x - 1, exec_env.size_y - 1, exec_env.size_z - 1],
            ).astype(int)
            cand = tuple(candidate.tolist())
            if exec_env.grid[cand[2], cand[1], cand[0]] == 0:
                exec_env.pos = cand
            wind_applied = True
            deviation_after_wind = distance_to_path(exec_env.pos, exec_path)

        do_replan = False
        if strategy == "periodic":
            do_replan = (tick % cfg.periodic_interval == 0 and replans < cfg.max_replans_periodic)
        elif strategy == "event":
            deviation = distance_to_path(exec_env.pos, exec_path)
            do_replan = (
                deviation > cfg.deviation_threshold
                and replans < cfg.max_replans_event
            )
        elif strategy != "no_replan":
            raise ValueError(strategy)

        if do_replan:
            t0 = time.perf_counter()
            new_path = plan_from_current_state(exec_env, agent, cfg.max_steps)
            planning_seconds += time.perf_counter() - t0
            replans += 1
            if len(new_path) >= 2:
                exec_path = new_path
                seg_dirs = build_segment_directions(exec_path)
                seg_idx = 0

        z = int(exec_env.pos[2])
        if (prev_z == 0 and z > 0) or (prev_z > 0 and z == 0):
            mode_switches += 1
        prev_z = z
        traj.append(tuple(exec_env.pos))

    traj_arr = np.asarray(traj, dtype=float)
    path_length = float(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1).sum()) if len(traj_arr) > 1 else 0.0
    success = bool(exec_env.wp_reached) and exec_env._distance_to_target(exec_env.pos, exec_env.goal) <= 0.5
    return {
        "strategy": strategy,
        "scenario": "path_deviation",
        "seed": seed,
        "initial_nominal_success": initial["success"],
        "success": int(success),
        "task_steps": len(traj) - 1,
        "path_length": path_length,
        "collision_attempts": collision_attempts,
        "replan_count": replans,
        "planning_time_s": planning_seconds,
        "final_distance": exec_env._distance_to_target(exec_env.pos, exec_env.goal),
        "mode_switches": mode_switches,
        "event_tick": wind_tick,
        "event_magnitude": float(np.linalg.norm([cfg.wind_dx, cfg.wind_dy, cfg.wind_dz])),
        "deviation_after_event": deviation_after_wind,
    }


def execute_replanning_benchmark(
    cfg: ExperimentConfig,
    agent: ConfigurableDQNAgent,
    seed: int,
    strategy: str,
) -> Dict:
    """Execute a preplanned path after inserting an obstacle on that path.

    The same initial path and obstacle placement are reconstructed for every
    strategy of the same seed.  No-replan follows the original path; periodic
    replanning updates at fixed intervals; event-triggered replanning updates
    only after the occupancy map differs from its baseline snapshot.
    """
    seed_everything(50_000 + seed)
    planning_env = make_env(cfg, seed=seed)
    initial = rollout_policy(planning_env, agent, cfg.max_steps)
    initial_path = initial["path"]
    if len(initial_path) < 4:
        raise RuntimeError(f"Initial path too short for seed {seed}")

    exec_env = make_env(cfg, seed=seed)
    exec_env.reset(seed=seed)
    baseline_occ = exec_env.get_static_occupancy_snapshot()

    wp_idx = find_waypoint_index(initial_path, exec_env.waypoint)
    min_inject = max(wp_idx + 5, int(len(initial_path) * cfg.dynamic_inject_fraction))
    min_inject = min(min_inject, max(1, len(initial_path) - cfg.dynamic_obstacle_lead - 2))
    inject_tick = max(1, min_inject)
    obstacle_idx = min(len(initial_path) - 2, inject_tick + cfg.dynamic_obstacle_lead)
    obstacle_point = initial_path[obstacle_idx]
    cube = valid_cube_around_point(
        exec_env,
        obstacle_point,
        cfg.dynamic_cube_radius_xy,
        cfg.dynamic_cube_radius_z,
    )

    exec_path = initial_path.copy()
    seg_dirs = build_segment_directions(exec_path)
    seg_idx = 0
    traj = [tuple(exec_env.pos)]
    map_baseline = baseline_occ.copy()
    replans = 0
    planning_seconds = 0.0
    collision_attempts = 0
    mode_switches = 0
    prev_z = int(exec_env.pos[2])
    injected = False
    last_check = -1

    for tick in range(1, cfg.max_execution_ticks + 1):
        if exec_env._distance_to_target(exec_env.pos, exec_env.goal) <= 0.5 and exec_env.wp_reached:
            break

        if (not injected) and tick == inject_tick:
            exec_env.set_cubes(cube1=cube, cube2=None, clear_before_apply=True)
            injected = True

        # Keep waypoint state consistent during open-loop path execution.
        if (not exec_env.wp_reached) and exec_env._distance_to_target(exec_env.pos, exec_env.waypoint) <= 1.0:
            exec_env.wp_reached = True

        do_replan = False
        if strategy == "periodic":
            do_replan = (tick % cfg.periodic_interval == 0 and replans < cfg.max_replans_periodic)
        elif strategy == "event":
            if (last_check < 0) or (tick - last_check >= cfg.event_check_interval):
                current_occ = exec_env.get_static_occupancy_snapshot()
                changed = current_occ.shape != map_baseline.shape or np.any(np.logical_xor(current_occ, map_baseline))
                do_replan = bool(changed and replans < cfg.max_replans_event)
                last_check = tick
        elif strategy != "no_replan":
            raise ValueError(strategy)

        if do_replan:
            t0 = time.perf_counter()
            new_path = plan_from_current_state(exec_env, agent, cfg.max_steps)
            planning_seconds += time.perf_counter() - t0
            replans += 1
            if len(new_path) >= 2:
                exec_path = new_path
                seg_dirs = build_segment_directions(exec_path)
                seg_idx = 0
            if strategy == "event":
                map_baseline = exec_env.get_static_occupancy_snapshot().copy()

        if len(seg_dirs) == 0:
            traj.append(tuple(exec_env.pos))
            continue

        seg_idx = min(seg_idx, len(seg_dirs) - 1)
        dx, dy, dz = map(int, seg_dirs[seg_idx])
        cx, cy, cz = map(int, exec_env.pos)
        nx, ny, nz = cx + dx, cy + dy, cz + dz
        feasible = (
            0 <= nx < exec_env.size_x
            and 0 <= ny < exec_env.size_y
            and 0 <= nz < exec_env.size_z
            and exec_env.grid[nz, ny, nx] == 0
        )
        if feasible:
            exec_env.pos = (nx, ny, nz)
            if seg_idx < len(seg_dirs) - 1:
                seg_idx += 1
        else:
            collision_attempts += 1

        z = int(exec_env.pos[2])
        if (prev_z == 0 and z > 0) or (prev_z > 0 and z == 0):
            mode_switches += 1
        prev_z = z
        traj.append(tuple(exec_env.pos))

    traj_arr = np.asarray(traj, dtype=float)
    path_length = float(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1).sum()) if len(traj_arr) > 1 else 0.0
    success = bool(exec_env.wp_reached) and exec_env._distance_to_target(exec_env.pos, exec_env.goal) <= 0.5
    return {
        "strategy": strategy,
        "scenario": "occupancy_change",
        "seed": seed,
        "initial_nominal_success": initial["success"],
        "success": int(success),
        "task_steps": len(traj) - 1,
        "path_length": path_length,
        "collision_attempts": collision_attempts,
        "replan_count": replans,
        "planning_time_s": planning_seconds,
        "final_distance": exec_env._distance_to_target(exec_env.pos, exec_env.goal),
        "mode_switches": mode_switches,
        "inject_tick": inject_tick,
        "obstacle_x": int(round(obstacle_point[0])),
        "obstacle_y": int(round(obstacle_point[1])),
        "obstacle_z": int(round(obstacle_point[2])),
    }


def mean_ci(values: Iterable[float], alpha: float = 0.05):
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = sd / math.sqrt(n)
        crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
        lo, hi = mean - crit * sem, mean + crit * sem
    else:
        lo = hi = mean
    return mean, sd, float(lo), float(hi)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_wilcoxon(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.nan, np.nan
    diff = x - y
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return np.nan, 1.0


def paired_binary_exact(a, b):
    """Exact McNemar-style test using discordant pairs and Binomial(0.5)."""
    x = np.asarray(a, dtype=int)
    y = np.asarray(b, dtype=int)
    b01 = int(np.sum((x == 0) & (y == 1)))
    b10 = int(np.sum((x == 1) & (y == 0)))
    discordant = b01 + b10
    if discordant == 0:
        return b01, b10, 1.0
    p = float(binomtest(min(b01, b10), n=discordant, p=0.5, alternative="two-sided").pvalue)
    return b01, b10, p


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    finite_idx = np.where(np.isfinite(p))[0]
    if len(finite_idx) == 0:
        return out.tolist()
    order = finite_idx[np.argsort(p[finite_idx])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out.tolist()


def summarize_training(df: pd.DataFrame) -> pd.DataFrame:
    continuous = [
        "final_return_mean100", "max_reward", "reward_std",
        "fixed_eval_convergence_episode", "fixed_eval_convergence_episode_capped", "fixed_eval_success_auc",
        "fixed_eval_final_reward", "fixed_eval_final_path_length",
        "fixed_eval_final_steps", "fixed_eval_final_collision_attempts",
        "fixed_eval_final_distance", "nominal_reward", "nominal_path_length",
        "nominal_steps", "nominal_collision_attempts", "nominal_mode_switches",
        "nominal_final_distance", "training_seconds",
    ]
    rows = []
    for variant, group in df.groupby("variant", sort=False):
        for metric in continuous:
            values = pd.to_numeric(group[metric], errors="coerce")
            mean, sd, lo, hi = mean_ci(values)
            rows.append({
                "variant": variant,
                "metric": metric,
                "n": int(np.isfinite(values.to_numpy(dtype=float)).sum()),
                "mean": mean, "sd": sd, "ci95_low": lo, "ci95_high": hi,
            })
        for metric in ["fixed_eval_converged", "fixed_eval_final_success", "nominal_success"]:
            succ = int(group[metric].sum())
            lo, hi = wilson_ci(succ, len(group))
            rows.append({
                "variant": variant,
                "metric": f"{metric}_rate",
                "n": len(group),
                "mean": succ / len(group),
                "sd": np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
            })
    return pd.DataFrame(rows)


def training_significance(df: pd.DataFrame) -> pd.DataFrame:
    proposed = "DDQN_PER_CL"
    comparisons = ["DDQN", "DDQN_PER", "DDQN_CL"]
    # Fixed-environment metrics are the primary cross-method inference targets.
    metrics = [
        "fixed_eval_success_auc",
        "fixed_eval_final_reward",
        "fixed_eval_final_path_length",
    ]
    rows = []
    for other in comparisons:
        pair = df[df.variant.isin([proposed, other])].pivot(index="seed", columns="variant")
        for metric in metrics:
            a = pair[metric][proposed]
            b = pair[metric][other]
            stat, p = paired_wilcoxon(a, b)
            rows.append({"comparison": f"{proposed} vs {other}", "metric": metric, "test": "paired Wilcoxon", "statistic": stat, "p_raw": p})
        for metric in ["fixed_eval_converged", "fixed_eval_final_success"]:
            a = pair[metric][proposed]
            b = pair[metric][other]
            b01, b10, p = paired_binary_exact(a, b)
            rows.append({"comparison": f"{proposed} vs {other}", "metric": metric, "test": "exact McNemar", "statistic": f"b01={b01};b10={b10}", "p_raw": p})
    p_adj = holm_adjust([float(r["p_raw"]) for r in rows])
    for row, padj in zip(rows, p_adj):
        row["p_holm"] = padj
        row["significant_0.05"] = bool(np.isfinite(padj) and padj < 0.05)
    return pd.DataFrame(rows)


def summarize_replanning(df: pd.DataFrame) -> pd.DataFrame:
    continuous = ["task_steps", "path_length", "collision_attempts", "replan_count", "planning_time_s", "final_distance", "mode_switches"]
    rows = []
    for strategy, group in df.groupby("strategy", sort=False):
        for metric in continuous:
            mean, sd, lo, hi = mean_ci(group[metric])
            rows.append({"strategy": strategy, "metric": metric, "n": len(group), "mean": mean, "sd": sd, "ci95_low": lo, "ci95_high": hi})
        succ = int(group["success"].sum())
        lo, hi = wilson_ci(succ, len(group))
        rows.append({"strategy": strategy, "metric": "success_rate", "n": len(group), "mean": succ / len(group), "sd": np.nan, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def replanning_significance(df: pd.DataFrame) -> pd.DataFrame:
    proposed = "event"
    comparisons = ["no_replan", "periodic"]
    metrics = ["task_steps", "path_length", "collision_attempts", "replan_count", "planning_time_s", "final_distance"]
    rows = []
    for other in comparisons:
        pair = df[df.strategy.isin([proposed, other])].pivot(index="seed", columns="strategy")
        for metric in metrics:
            a = pair[metric][proposed]
            b = pair[metric][other]
            stat, p = paired_wilcoxon(a, b)
            rows.append({"comparison": f"event vs {other}", "metric": metric, "test": "paired Wilcoxon", "statistic": stat, "p_raw": p})
        a = pair["success"][proposed]
        b = pair["success"][other]
        b01, b10, p = paired_binary_exact(a, b)
        rows.append({"comparison": f"event vs {other}", "metric": "success", "test": "exact McNemar", "statistic": f"b01={b01};b10={b10}", "p_raw": p})
    p_adj = holm_adjust([float(r["p_raw"]) for r in rows])
    for row, padj in zip(rows, p_adj):
        row["p_holm"] = padj
        row["significant_0.05"] = bool(np.isfinite(padj) and padj < 0.05)
    return pd.DataFrame(rows)


def collect_training_curves(results_root: Path, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in cfg.seeds:
            path = results_root / "training" / variant / f"seed_{seed:02d}" / "training_curve.csv"
            if not path.exists():
                continue
            cur = pd.read_csv(path)
            cur["variant"] = variant
            cur["seed"] = seed
            rows.append(cur)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_fixed_eval_curves(results_root: Path, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in cfg.seeds:
            path = results_root / "training" / variant / f"seed_{seed:02d}" / "fixed_eval_curve.csv"
            if not path.exists():
                continue
            cur = pd.read_csv(path)
            cur["variant"] = variant
            cur["seed"] = seed
            rows.append(cur)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_fixed_eval_curves(curves: pd.DataFrame, out_dir: Path):
    if curves.empty:
        return
    # Success fraction across seeds at each common checkpoint.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for variant in VARIANTS:
        sub = curves[curves.variant == variant]
        if sub.empty:
            continue
        pivot = sub.pivot(index="episode", columns="seed", values="success").sort_index()
        mean = pivot.mean(axis=1)
        n = pivot.notna().sum(axis=1).clip(lower=1)
        sd = pivot.std(axis=1).fillna(0.0)
        ci = 1.96 * sd / np.sqrt(n)
        line, = ax.plot(mean.index, mean.values, marker="o", ms=3, label=DISPLAY_NAMES[variant])
        ax.fill_between(mean.index, np.clip(mean-ci, 0, 1).values, np.clip(mean+ci, 0, 1).values, alpha=0.15, color=line.get_color())
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Fixed-map greedy success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_fixed_eval_success_mean95CI.png", dpi=300)
    fig.savefig(out_dir / "Fig_fixed_eval_success_mean95CI.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for variant in VARIANTS:
        sub = curves[curves.variant == variant]
        if sub.empty:
            continue
        pivot = sub.pivot(index="episode", columns="seed", values="reward").sort_index()
        mean = pivot.mean(axis=1)
        n = pivot.notna().sum(axis=1).clip(lower=1)
        sd = pivot.std(axis=1).fillna(0.0)
        ci = 1.96 * sd / np.sqrt(n)
        line, = ax.plot(mean.index, mean.values, marker="o", ms=3, label=DISPLAY_NAMES[variant])
        ax.fill_between(mean.index, (mean-ci).values, (mean+ci).values, alpha=0.15, color=line.get_color())
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Fixed-map greedy return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_fixed_eval_return_mean95CI.png", dpi=300)
    fig.savefig(out_dir / "Fig_fixed_eval_return_mean95CI.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(curves: pd.DataFrame, out_dir: Path, ma_window: int = 20):
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for variant in VARIANTS:
        sub = curves[curves.variant == variant]
        if sub.empty:
            continue
        pivot = sub.pivot(index="episode", columns="seed", values="reward").sort_index()
        smoothed = pivot.rolling(ma_window, min_periods=ma_window).mean()
        mean = smoothed.mean(axis=1)
        n = smoothed.notna().sum(axis=1).clip(lower=1)
        sd = smoothed.std(axis=1)
        sem = sd / np.sqrt(n)
        ci = 1.96 * sem
        line, = ax.plot(mean.index, mean.values, label=DISPLAY_NAMES[variant])
        ax.fill_between(mean.index, (mean-ci).values, (mean+ci).values, alpha=0.15, color=line.get_color())
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Moving-average return (window={ma_window})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "Fig_training_ablation_mean95CI.png", dpi=300)
    fig.savefig(out_dir / "Fig_training_ablation_mean95CI.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_replanning_summary(raw: pd.DataFrame, out_dir: Path):
    if raw.empty:
        return
    metrics = [
        ("success", "Success rate", True),
        ("replan_count", "Number of replanning calls", False),
        ("planning_time_s", "Cumulative replanning time (s)", False),
    ]
    for metric, ylabel, binary in metrics:
        groups = [raw[raw.strategy == s][metric].to_numpy(dtype=float) for s in ["no_replan", "periodic", "event"]]
        means = [np.mean(g) if len(g) else np.nan for g in groups]
        errs = []
        for g in groups:
            mean, sd, lo, hi = mean_ci(g)
            errs.append(max(0.0, hi-mean) if np.isfinite(mean) else np.nan)
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        x = np.arange(3)
        ax.bar(x, means, yerr=errs, capsize=4)
        ax.set_xticks(x, [DISPLAY_NAMES[s] for s in ["no_replan", "periodic", "event"]], rotation=10)
        ax.set_ylabel(ylabel)
        if binary:
            ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        stem = f"Fig_replanning_{metric}"
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def fmt_mean_sd(summary: pd.DataFrame, key_col: str, key: str, metric: str, digits: int = 2, percent: bool = False):
    row = summary[(summary[key_col] == key) & (summary.metric == metric)].iloc[0]
    if percent:
        return f"{100*row['mean']:.1f}"
    if pd.isna(row["sd"]):
        return f"{row['mean']:.{digits}f}"
    return f"{row['mean']:.{digits}f} $\\pm$ {row['sd']:.{digits}f}"


def write_training_latex(summary: pd.DataFrame, out_path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation of PER and curriculum learning using fixed full-map greedy evaluations over multiple random seeds.}",
        r"\label{tab:training_ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & $\mathrm{AUC}_{\rm SR}$ $\uparrow$ & Conv. (\%) $\uparrow$ & $E_{\rm conv}^{*}$ $\downarrow$ & Final success (\%) $\uparrow$ & Path length $\downarrow$ \\",
        r"\midrule",
    ]
    for variant in VARIANTS:
        lines.append(
            f"{DISPLAY_NAMES[variant]} & "
            f"{fmt_mean_sd(summary, 'variant', variant, 'fixed_eval_success_auc', digits=3)} & "
            f"{fmt_mean_sd(summary, 'variant', variant, 'fixed_eval_converged_rate', percent=True)} & "
            f"{fmt_mean_sd(summary, 'variant', variant, 'fixed_eval_convergence_episode', digits=1)} & "
            f"{fmt_mean_sd(summary, 'variant', variant, 'fixed_eval_final_success_rate', percent=True)} & "
            f"{fmt_mean_sd(summary, 'variant', variant, 'fixed_eval_final_path_length')} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\parbox{0.98\linewidth}{\footnotesize $E_{\rm conv}^{*}$ is averaged over converged runs only; non-converged runs are reported by the convergence-rate column and are not encoded as an artificial episode.}", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_replanning_latex(summary: pd.DataFrame, out_path: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of replanning strategies under an online occupancy change.}",
        r"\label{tab:replanning_ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Success (\%) $\uparrow$ & Replans $\downarrow$ & Planning time (s) $\downarrow$ & Final distance $\downarrow$ \\",
        r"\midrule",
    ]
    for strategy in ["no_replan", "periodic", "event"]:
        lines.append(
            f"{DISPLAY_NAMES[strategy]} & "
            f"{fmt_mean_sd(summary, 'strategy', strategy, 'success_rate', percent=True)} & "
            f"{fmt_mean_sd(summary, 'strategy', strategy, 'replan_count')} & "
            f"{fmt_mean_sd(summary, 'strategy', strategy, 'planning_time_s', digits=3)} & "
            f"{fmt_mean_sd(summary, 'strategy', strategy, 'final_distance')} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_training(cfg: ExperimentConfig, results_root: Path, resume: bool, quiet: bool):
    rows = []
    for variant in VARIANTS:
        for seed in cfg.seeds:
            seed_dir = results_root / "training" / variant / f"seed_{seed:02d}"
            checkpoint = seed_dir / "checkpoint.pth"
            curve = seed_dir / "training_curve.csv"
            fixed_curve = seed_dir / "fixed_eval_curve.csv"
            can_resume = False
            if resume and checkpoint.exists() and curve.exists() and fixed_curve.exists():
                try:
                    payload = torch.load(checkpoint, map_location="cpu")
                    can_resume = payload.get("protocol_version") == cfg.protocol_version
                except Exception:
                    can_resume = False
            if can_resume:
                print(f"[resume] {variant} seed={seed}: protocol-v3 checkpoint/curves found")
            else:
                if resume and seed_dir.exists():
                    print(f"[resume] {variant} seed={seed}: legacy/incomplete result detected -> retraining")
                    shutil.rmtree(seed_dir, ignore_errors=True)
                train_variant(cfg, variant, seed, seed_dir, quiet=quiet)
            rows.append(training_seed_metrics(cfg, variant, seed, seed_dir))
            pd.DataFrame(rows).to_csv(results_root / "training_metrics_by_seed.csv", index=False)

    metrics = pd.DataFrame(rows)
    summary = summarize_training(metrics)
    significance = training_significance(metrics)
    summary.to_csv(results_root / "training_summary.csv", index=False)
    significance.to_csv(results_root / "training_significance.csv", index=False)

    curves = collect_training_curves(results_root, cfg)
    curves.to_csv(results_root / "training_curves_all_seeds.csv", index=False)
    plot_training_curves(curves, results_root, cfg.moving_average_window)

    fixed_curves = collect_fixed_eval_curves(results_root, cfg)
    fixed_curves.to_csv(results_root / "fixed_eval_curves_all_seeds.csv", index=False)
    plot_fixed_eval_curves(fixed_curves, results_root)

    write_training_latex(summary, results_root / "table_training_ablation.tex")
    return metrics


def run_replanning(cfg: ExperimentConfig, results_root: Path):
    rows = []
    excluded = []
    for seed in cfg.seeds:
        checkpoint = results_root / "training" / "DDQN_PER" / f"seed_{seed:02d}" / "checkpoint.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing proposed checkpoint for seed {seed}: {checkpoint}. Run --stage training or --stage all first."
            )
        agent, _ = load_agent_from_checkpoint(cfg, checkpoint)

        # Event-layer ablation should not be confounded by a checkpoint that
        # already fails the nominal task. Training failures are reported in the
        # PER/CL ablation; replanning statistics use only nominally successful
        # proposed checkpoints and explicitly report excluded seeds.
        nominal_env = make_env(cfg, seed=seed)
        nominal = rollout_policy(nominal_env, agent, cfg.max_steps)
        if not nominal["success"]:
            print(f"[replanning] seed={seed:02d} excluded: proposed checkpoint fails nominal task")
            excluded.append({
                "seed": seed,
                "reason": "DDQN_PER checkpoint failed nominal waypoint+goal task",
                "nominal_final_distance": nominal["final_distance"],
                "nominal_reward": nominal["reward"],
            })
            pd.DataFrame(excluded).to_csv(results_root / "replanning_excluded_seeds.csv", index=False)
            continue

        for strategy in ["no_replan", "periodic", "event"]:
            print(f"[replanning] seed={seed:02d}, strategy={strategy}, scenario={cfg.replanning_scenario}")
            if cfg.replanning_scenario == "deviation":
                result = execute_deviation_replanning_benchmark(cfg, agent, seed, strategy)
            elif cfg.replanning_scenario == "occupancy":
                result = execute_replanning_benchmark(cfg, agent, seed, strategy)
            else:
                raise ValueError(f"Unknown replanning_scenario: {cfg.replanning_scenario}")
            rows.append(result)
            pd.DataFrame(rows).to_csv(results_root / "replanning_metrics_by_seed.csv", index=False)

    if excluded:
        pd.DataFrame(excluded).to_csv(results_root / "replanning_excluded_seeds.csv", index=False)
    else:
        pd.DataFrame(columns=["seed", "reason", "nominal_final_distance", "nominal_reward"]).to_csv(
            results_root / "replanning_excluded_seeds.csv", index=False
        )

    raw = pd.DataFrame(rows)
    if raw.empty:
            raise RuntimeError(
                "No nominally successful PER-enhanced DDQN checkpoints are available for the replanning ablation. "
                "Inspect training_metrics_by_seed.csv before rerunning this stage."
            )
    summary = summarize_replanning(raw)
    significance = replanning_significance(raw)
    summary.to_csv(results_root / "replanning_summary.csv", index=False)
    significance.to_csv(results_root / "replanning_significance.csv", index=False)
    plot_replanning_summary(raw, results_root)
    write_replanning_latex(summary, results_root / "table_replanning_ablation.tex")
    return raw


def analyze_existing(cfg: ExperimentConfig, results_root: Path):
    train_path = results_root / "training_metrics_by_seed.csv"
    if train_path.exists():
        metrics = pd.read_csv(train_path)
        summarize_training(metrics).to_csv(results_root / "training_summary.csv", index=False)
        training_significance(metrics).to_csv(results_root / "training_significance.csv", index=False)
        curves = collect_training_curves(results_root, cfg)
        if not curves.empty:
            curves.to_csv(results_root / "training_curves_all_seeds.csv", index=False)
            plot_training_curves(curves, results_root, cfg.moving_average_window)
        fixed_curves = collect_fixed_eval_curves(results_root, cfg)
        if not fixed_curves.empty:
            fixed_curves.to_csv(results_root / "fixed_eval_curves_all_seeds.csv", index=False)
            plot_fixed_eval_curves(fixed_curves, results_root)
        summary = summarize_training(metrics)
        write_training_latex(summary, results_root / "table_training_ablation.tex")
    replan_path = results_root / "replanning_metrics_by_seed.csv"
    if replan_path.exists():
        raw = pd.read_csv(replan_path)
        summary = summarize_replanning(raw)
        summary.to_csv(results_root / "replanning_summary.csv", index=False)
        replanning_significance(raw).to_csv(results_root / "replanning_significance.csv", index=False)
        plot_replanning_summary(raw, results_root)
        write_replanning_latex(summary, results_root / "table_replanning_ablation.tex")


def parse_seeds(value: str) -> Tuple[int, ...]:
    value = value.strip()
    if "-" in value and "," not in value:
        a, b = value.split("-", 1)
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def load_config(path: Path | None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "seeds" in raw:
        raw["seeds"] = tuple(raw["seeds"])
    return ExperimentConfig(**raw)


def main():
    parser = argparse.ArgumentParser(description="PER/CL/Event-trigger ablation with multi-seed statistics")
    parser.add_argument("--stage", choices=["training", "replanning", "all", "analyze"], default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_ablation.json")
    parser.add_argument("--results", type=Path, default=ROOT / "paper_ablation_results")
    parser.add_argument("--seeds", type=str, default=None, help="Override seeds, e.g. 0-19 or 0,1,2")
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes")
    parser.add_argument("--device", type=str, default=None, help="auto/cpu/cuda")
    parser.add_argument("--resume", action="store_true", help="Skip completed seed training runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config if args.config.exists() else None)
    if args.seeds is not None:
        cfg.seeds = parse_seeds(args.seeds)
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.device is not None:
        cfg.device = args.device

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print("=" * 72)
    print("Paper ablation configuration")
    print(json.dumps(asdict(cfg), indent=2))
    print("=" * 72)

    if args.stage in ("training", "all"):
        run_training(cfg, args.results, resume=args.resume, quiet=args.quiet)
    if args.stage in ("replanning", "all"):
        run_replanning(cfg, args.results)
    if args.stage == "analyze":
        analyze_existing(cfg, args.results)

    print(f"\nDone. Paper-ready results are in: {args.results}")


if __name__ == "__main__":
    main()
