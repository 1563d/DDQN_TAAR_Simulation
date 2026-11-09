#!/usr/bin/env python3
"""Fast dependency/import check before launching long experiments."""
import importlib
import sys
from pathlib import Path

mods = ["numpy", "pandas", "scipy", "matplotlib", "torch", "gymnasium", "tqdm"]
failed = []
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(mod, '__version__', 'installed')}")
    except Exception as exc:
        failed.append((name, exc))
        print(f"[FAIL] {name}: {exc}")

if failed:
    print("\nInstall missing dependencies before running the experiment suite.")
    sys.exit(1)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from path_planner.env import GridWorld3DEnv
from experiments.configurable_agent import ConfigurableDQNAgent

env = GridWorld3DEnv()
agent = ConfigurableDQNAgent(env.observation_space.shape[0], env.action_space.n, use_per=False)
state, _ = env.reset(seed=0)
action = agent.select_action(state, eps=1.0)
next_state, reward, done, truncated, info = env.step(action)
print("[OK] Project import + one environment step succeeded.")

# Reward-protocol sanity check: blocked action near the goal must not receive positive shaping.
env = GridWorld3DEnv()
env.reset(seed=0)
gx, gy, gz = env.goal
env.pos = (gx - 1, gy, gz)
env.wp_reached = True
env.grid[gz, gy, gx] = 1
env._refresh_padded_grid()
action = env.moves.index((1, 0, 0))
_, reward, _, _, info = env.step(action)
if not (info.get("collision") and reward < 0):
    print(f"[FAIL] Reward protocol regression: collision reward={reward}, info={info}")
    sys.exit(1)
print("[OK] Reward protocol v2 collision/proximity regression check succeeded.")
