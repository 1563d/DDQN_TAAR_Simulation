from pathlib import Path
import shutil
import numpy as np
from sim_common import *

# -----------------------------------------------------------------------------
# Case 2A: fixed online obstacle used for the paper-style representative figure
# -----------------------------------------------------------------------------
SEED = 10

# The new obstacle is fixed near the ground path before the prescribed waypoint.
# Bounds follow the environment convention: min is inclusive, max is exclusive.
# Its location is chosen to match the representative obstacle region in the
# paper figure while still leaving a feasible detour for the trained planner.
NEW_OBSTACLE = {
    'min': (10, 61, 0),
    'max': (16, 66, 6),
    'value': 1,
}

# Insert the obstacle immediately before the robot reaches it. At tick 39 the
# nominal trajectory is at approximately (16, 61, 0), adjacent to the obstacle.
OBSTACLE_INJECT_TICK = 39
EVENT_CHECK_INTERVAL = 1
MAX_REPLANS_EVENT = 3

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / 'checkpoint' / 'proposed_PER_DDQN_seed10.pth'
RESULTS = ROOT / 'results'


def path_intersects_cube(path, cube):
    """Return True if any path point lies inside the fixed obstacle cuboid."""
    P = np.asarray(path, dtype=float)
    if len(P) == 0:
        return False
    lo = np.asarray(cube['min'], dtype=float)
    hi = np.asarray(cube['max'], dtype=float)
    inside = np.all((P >= lo) & (P < hi), axis=1)
    return bool(np.any(inside))


def execute(agent, use_behavior_management: bool):
    """Execute the same nominal plan with/without the behavior-management layer.

    Both cases use the same final PER-enhanced DDQN checkpoint, initial path,
    fixed obstacle, and obstacle-insertion time.

    With behavior management:
        occupancy change -> replanning request -> new collision-free path.

    Without behavior management:
        no path update; the stored final reference path remains the original
        nominal path and therefore geometrically passes through the new obstacle.
    """
    seed_everything(50_000 + SEED)

    # 1) Generate the common nominal path before the obstacle appears.
    planning_env = make_env(SEED)
    initial = rollout_policy(planning_env, agent)
    if not initial['success']:
        raise RuntimeError('The selected PER-enhanced DDQN checkpoint fails the nominal task.')
    initial_path = np.asarray(initial['path'], dtype=float)

    # Sanity check: the fixed obstacle must invalidate the nominal path.
    if not path_intersects_cube(initial_path, NEW_OBSTACLE):
        raise RuntimeError('NEW_OBSTACLE does not intersect the nominal path; adjust its fixed location.')

    # 2) Common execution environment.
    env = make_env(SEED)
    env.reset(seed=SEED)
    baseline_occ = env.get_static_occupancy_snapshot().copy()

    reference_path = initial_path.copy()
    # This is the complete route used for plotting after the final planning event.
    final_reference_path = initial_path.copy()
    replanned_segment = np.empty((0, 3), dtype=float)

    seg_dirs = build_segment_directions(reference_path)
    seg_idx = 0
    traj = [tuple(env.pos)]
    injected = False
    last_check = -1
    replans = 0
    collision_attempts = 0
    replan_point = None
    replan_tick = None

    for tick in range(1, MAX_EXEC_TICKS + 1):
        if env._distance_to_target(env.pos, env.goal) <= 0.5 and env.wp_reached:
            break

        # Fixed obstacle appears at the same time/location in both cases.
        if (not injected) and tick == OBSTACLE_INJECT_TICK:
            env.set_cubes(cube1=NEW_OBSTACLE, cube2=None, clear_before_apply=True)
            injected = True

        if (not env.wp_reached) and env._distance_to_target(env.pos, env.waypoint) <= 1.0:
            env.wp_reached = True

        # 3) Behavior-management branch: detect occupancy change and request replan.
        if use_behavior_management:
            do_replan = False
            if (last_check < 0) or (tick - last_check >= EVENT_CHECK_INTERVAL):
                current_occ = env.get_static_occupancy_snapshot()
                changed = (
                    current_occ.shape != baseline_occ.shape
                    or np.any(np.logical_xor(current_occ, baseline_occ))
                )
                do_replan = bool(changed and replans < MAX_REPLANS_EVENT)
                last_check = tick

            if do_replan:
                new_path = np.asarray(plan_from_current_state(env, agent), dtype=float)
                replans += 1

                if replan_point is None:
                    replan_point = np.asarray(env.pos, dtype=float)
                    replan_tick = tick

                if len(new_path) >= 2:
                    replanned_segment = new_path.copy()
                    reference_path = new_path.copy()
                    seg_dirs = build_segment_directions(reference_path)
                    seg_idx = 0

                    # Full final planned route = already executed prefix + new plan.
                    prefix = np.asarray(traj, dtype=float)
                    if len(prefix) and np.allclose(prefix[-1], new_path[0]):
                        final_reference_path = np.vstack((prefix, new_path[1:]))
                    else:
                        final_reference_path = np.vstack((prefix, new_path))

                baseline_occ = env.get_static_occupancy_snapshot().copy()

        # 4) Execute the current reference path.
        if len(seg_dirs) == 0:
            traj.append(tuple(env.pos))
            continue

        seg_idx = min(seg_idx, len(seg_dirs) - 1)
        dx, dy, dz = map(int, seg_dirs[seg_idx])
        cx, cy, cz = map(int, env.pos)
        nx, ny, nz = cx + dx, cy + dy, cz + dz
        feasible = (
            0 <= nx < env.size_x
            and 0 <= ny < env.size_y
            and 0 <= nz < env.size_z
            and env.grid[nz, ny, nx] == 0
        )

        if feasible:
            env.pos = (nx, ny, nz)
            if seg_idx < len(seg_dirs) - 1:
                seg_idx += 1
        else:
            collision_attempts += 1

        traj.append(tuple(env.pos))

    traj = np.asarray(traj, dtype=float)
    success = (
        bool(env.wp_reached)
        and env._distance_to_target(env.pos, env.goal) <= 0.5
        and collision_attempts == 0
    )

    # Enforce the intended representative comparison.
    planned_intersection = path_intersects_cube(final_reference_path, NEW_OBSTACLE)
    if use_behavior_management and planned_intersection:
        raise RuntimeError('Replanned path still intersects the new obstacle.')
    if (not use_behavior_management) and (not planned_intersection):
        raise RuntimeError('No-management reference path no longer passes through the new obstacle.')

    return dict(
        initial_path=initial_path,
        final_reference_path=final_reference_path,
        replanned_segment=replanned_segment,
        executed_trajectory=traj,
        env=env,
        inject_tick=int(OBSTACLE_INJECT_TICK),
        obstacle_cube=NEW_OBSTACLE,
        replan_tick=replan_tick,
        replan_point=replan_point,
        replan_count=int(replans),
        collision_attempts=int(collision_attempts),
        success=int(success),
        final_distance=float(env._distance_to_target(env.pos, env.goal)),
        planned_path_intersects_obstacle=int(planned_intersection),
    )


def save_case(name, data):
    out = RESULTS / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    save_path(out, 'initial_nominal_path', data['initial_path'])
    save_path(out, 'final_reference_path', data['final_reference_path'])
    save_path(out, 'replanned_segment', data['replanned_segment'])
    save_path(out, 'executed_trajectory', data['executed_trajectory'])
    save_grid(out, data['env'])

    meta = {
        'seed': SEED,
        'behavior_management': name == 'with_behavior_management',
        'success': data['success'],
        'replan_count': data['replan_count'],
        'collision_attempts': data['collision_attempts'],
        'final_distance': data['final_distance'],
        'initial_path_length': path_length(data['initial_path']),
        'final_reference_path_length': path_length(data['final_reference_path']),
        'executed_path_length': path_length(data['executed_trajectory']),
        'obstacle_inject_tick': data['inject_tick'],
        'obstacle_cube_min': list(data['obstacle_cube']['min']),
        'obstacle_cube_max': list(data['obstacle_cube']['max']),
        'replan_tick': data['replan_tick'],
        'replan_point': None if data['replan_point'] is None else data['replan_point'].tolist(),
        'planned_path_intersects_new_obstacle': data['planned_path_intersects_obstacle'],
    }
    write_metrics(out, meta)


def main():
    RESULTS.mkdir(exist_ok=True)
    agent = load_proposed_agent(CKPT)

    with_mgr = execute(agent, True)
    without_mgr = execute(agent, False)

    save_case('with_behavior_management', with_mgr)
    save_case('without_behavior_management', without_mgr)

    print('\n=== Fixed new-obstacle case ===')
    print('Obstacle cuboid:', NEW_OBSTACLE)
    print('Insertion tick:', OBSTACLE_INJECT_TICK)
    for label, d in [
        ('With behavior management', with_mgr),
        ('Without behavior management', without_mgr),
    ]:
        print(
            f'{label}: success={d["success"]}, replans={d["replan_count"]}, '
            f'collisions={d["collision_attempts"]}, final_distance={d["final_distance"]:.3f}, '
            f'planned_path_intersects_obstacle={d["planned_path_intersects_obstacle"]}'
        )

    print('\nFor plotting, use final_reference_path.csv/.mat:')
    print('  with behavior management    -> detours around the new obstacle')
    print('  without behavior management -> nominal path passes through the new obstacle')
    print('Saved results under:', RESULTS)


if __name__ == '__main__':
    main()
