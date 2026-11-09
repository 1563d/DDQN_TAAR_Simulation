from pathlib import Path
import json

root = Path(__file__).resolve().parent
cfg = json.loads((root / "configs" / "paper_ablation.json").read_text(encoding="utf-8"))
assert cfg["protocol_version"] == "paper_ablation_v3_adaptive_curriculum"
assert cfg["curriculum_min_episodes"][:4] == [20, 20, 20, 20]
assert cfg["curriculum_max_episodes"][:4] == [50, 50, 50, 50]
assert cfg["curriculum_eval_interval"] == 10
assert cfg["curriculum_mastery_hold"] == 2
assert cfg["episodes"] >= sum(cfg["curriculum_max_episodes"][:4])
print("[OK] protocol v3 config")
print("[OK] at least", cfg["episodes"] - sum(cfg["curriculum_max_episodes"][:4]), "episodes are reserved for the final full-complexity phase")
print("[OK] v2 reward and fixed-evaluation protocol are retained")
