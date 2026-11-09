# Case 2A — Fixed New Obstacle / Behavior-Management Comparison

This project uses the final PER-enhanced DDQN checkpoint (`seed=10`) in both cases. The lower-level planner is identical; the only difference is whether the event-triggered behavior-management layer is enabled.

## Fixed obstacle

The online obstacle is inserted at a fixed paper-style location:

```python
NEW_OBSTACLE = {
    "min": (10, 61, 0),
    "max": (16, 66, 6),
    "value": 1,
}
OBSTACLE_INJECT_TICK = 39
```

The upper bound is exclusive. The obstacle is placed directly on the original nominal ground path and appears when the robot is adjacent to it.

## Expected comparison

- **With behavior management:** the occupancy change is detected at the insertion event; the upper layer requests replanning from the current state. The saved final path detours around the obstacle.
- **Without behavior management:** no replanning is allowed. The final stored reference path remains the nominal path and therefore geometrically passes through the newly inserted obstacle. During execution, collision checking prevents the simulated robot from physically entering occupied voxels, so the executed trajectory stalls at the obstacle. For the paper path-comparison figure, plot `final_reference_path`, not `executed_trajectory`.

## Run

```powershell
python -m pip install -r ../../requirements.txt
python run_simulation.py
```

## Main outputs

Each result folder contains:

- `initial_nominal_path.csv/.mat`
- `final_reference_path.csv/.mat` — **use this for the path-comparison figure**
- `replanned_segment.csv/.mat`
- `executed_trajectory.csv/.mat`
- `grid_world_with_new_obstacle.mat/.npy`
- `metrics.csv/.json`

The metrics include `planned_path_intersects_new_obstacle`, which should be `0` with behavior management and `1` without behavior management.
