# Case 2B: aerial path-deviation simulation

This project uses the final **PER-enhanced DDQN** checkpoint (seed 10) for both compared strategies. During late aerial execution the same controlled displacement `[2, 2, -2]` is applied to both strategies.

- **With behavior management:** when path deviation exceeds the threshold, the upper layer requests replanning from the disturbed state.
- **Without behavior management:** no replanning is requested and the robot continues following the original reference-path directions.

This is a controlled path-deviation disturbance, not an aerodynamic wind-field model.

Run on Windows:

```powershell
python -m pip install -r ../../requirements.txt
python run_simulation.py
```

Outputs are saved to `results/with_behavior_management/` and `results/without_behavior_management/`:
- `initial_nominal_path.csv/.mat`
- `final_reference_path.csv/.mat`
- `executed_trajectory.csv/.mat`
- `grid_world.mat/.npy`
- `metrics.csv/.json`
