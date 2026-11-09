from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SUITE_VERSION = "real_scale_randomized_v1"


def _add_jitter(base, jitter, rng):
    return [int(b + rng.integers(-int(j), int(j)+1)) for b, j in zip(base, jitter)]


def scenario_signature(scenario):
    key = {
        'start_cell': scenario['start_cell'],
        'ground_waypoint_cell': scenario['ground_waypoint_cell'],
        'inspection_goal_cell': scenario['inspection_goal_cell'],
        'ground_obstacles': [
            {'min_cell': o['min_cell'], 'max_cell': o['max_cell']}
            for o in scenario['ground_obstacles']
        ],
    }
    return json.dumps(key, sort_keys=True, separators=(',', ':'))


def _candidate(cfg, source_seed: int, attempt: int, scenario_id: str):
    rcfg = cfg['scenario_randomization']
    # local RNG: does not perturb training / PER RNG streams
    ss = np.random.SeedSequence([int(source_seed), int(attempt), 20260812])
    rng = np.random.default_rng(ss)

    start = _add_jitter(cfg['task']['start_cell'], rcfg['start_jitter_cells'], rng)
    waypoint = _add_jitter(cfg['task']['ground_waypoint_cell'], rcfg['waypoint_jitter_cells'], rng)
    goal = _add_jitter(cfg['task']['inspection_goal_cell'], rcfg['goal_jitter_cells'], rng)
    start[2] = 0
    waypoint[2] = 0

    obstacles = []
    for spec in rcfg['ground_obstacles']:
        base = spec['base_min_cell']
        jitter = spec['jitter_cells']
        size = spec['size_cells']
        omin = _add_jitter(base, jitter, rng)
        omin[2] = int(base[2])
        omax = [int(omin[i] + size[i]) for i in range(3)]
        obstacles.append({
            'name': spec['name'],
            'min_cell': omin,
            'max_cell': omax,
        })

    return {
        'suite_version': SUITE_VERSION,
        'scenario_id': str(scenario_id),
        'source_seed': int(source_seed),
        'start_cell': list(map(int, start)),
        'ground_waypoint_cell': list(map(int, waypoint)),
        'inspection_goal_cell': list(map(int, goal)),
        'ground_obstacles': obstacles,
    }


def generate_feasible_scenario(cfg, env, source_seed: int, scenario_id: str,
                               reserved_signatures=None, require_3d_path=False,
                               max_attempts=500):
    reserved = set(reserved_signatures or [])
    for attempt in range(int(max_attempts)):
        s = _candidate(cfg, int(source_seed), attempt, scenario_id)
        sig = scenario_signature(s)
        if sig in reserved:
            continue
        try:
            env.apply_scenario(s, validate_ground=True)
        except ValueError:
            continue
        if require_3d_path and not env._has_3d_path(env.waypoint, env.goal):
            continue
        s['generation_attempt'] = int(attempt)
        return s
    raise RuntimeError(f'Could not generate a feasible unique scenario for seed={source_seed}, id={scenario_id}')


def build_suite(cfg, env, seeds, prefix, reserved_signatures=None, require_3d_path=True):
    suite = []
    reserved = set(reserved_signatures or [])
    for i, seed in enumerate(seeds, start=1):
        sid = f'{prefix}_{i:02d}'
        s = generate_feasible_scenario(
            cfg, env, int(seed), sid,
            reserved_signatures=reserved,
            require_3d_path=require_3d_path,
        )
        suite.append(s)
        reserved.add(scenario_signature(s))
    return suite


def save_suite(path, suite, description=''):
    payload = {
        'suite_version': SUITE_VERSION,
        'description': description,
        'num_scenarios': len(suite),
        'scenarios': suite,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def load_suite(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data['scenarios']


def scenario_row(s):
    obs = s['ground_obstacles']
    row = {
        'scenario_id': s['scenario_id'],
        'scenario_seed': s['source_seed'],
        'start_x': s['start_cell'][0], 'start_y': s['start_cell'][1], 'start_z': s['start_cell'][2],
        'waypoint_x': s['ground_waypoint_cell'][0], 'waypoint_y': s['ground_waypoint_cell'][1], 'waypoint_z': s['ground_waypoint_cell'][2],
        'goal_x': s['inspection_goal_cell'][0], 'goal_y': s['inspection_goal_cell'][1], 'goal_z': s['inspection_goal_cell'][2],
    }
    for i, o in enumerate(obs, start=1):
        for k, v in enumerate(o['min_cell']): row[f'obs{i}_min_{"xyz"[k]}'] = v
        for k, v in enumerate(o['max_cell']): row[f'obs{i}_max_{"xyz"[k]}'] = v
    return row
