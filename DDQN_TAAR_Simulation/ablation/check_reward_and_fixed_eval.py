#!/usr/bin/env python3
"""Quick protocol checks for the revised paper-ablation suite."""
from pathlib import Path
import tempfile

from experiments.paper_ablation import ExperimentConfig, make_env, make_agent, append_fixed_eval_record


def main():
    cfg = ExperimentConfig(episodes=10, fixed_eval_interval=5, fixed_eval_hold=2, device="cpu")
    env = make_env(cfg, seed=0)

    # Reward-hacking regression test: put the robot one grid step from the goal,
    # block the goal cell, and command the blocked move. Reward must stay negative.
    env.reset(seed=0)
    gx, gy, gz = env.goal
    pos = (gx - 1, gy, gz)
    env.pos = pos
    env.wp_reached = True
    env.grid[gz, gy, gx] = 1
    env._refresh_padded_grid()
    action = env.moves.index((1, 0, 0))
    _, reward, _, _, info = env.step(action)
    assert info.get("collision") is True
    assert reward < 0.0, f"blocked near-goal action produced positive reward: {reward}"
    assert env.pos == pos, "collision unexpectedly changed position"

    # Fixed-evaluation file generation check.
    agent = make_agent(env, cfg, use_per=False)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fixed_eval_curve.csv"
        records = []
        append_fixed_eval_record(cfg, agent, 0, records, out)
        append_fixed_eval_record(cfg, agent, 5, records, out)
        assert out.exists() and len(records) == 2

    print("[OK] reward exploit regression test")
    print("[OK] fixed full-map evaluation writer")
    print("Protocol:", cfg.protocol_version)


if __name__ == "__main__":
    main()
