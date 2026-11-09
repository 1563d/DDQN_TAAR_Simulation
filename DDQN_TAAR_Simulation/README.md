# TAAR cross-domain path planning

Simulation code accompanying *Deep Reinforcement Learning-Based Cross-Domain Path Planning for Terrestrial-Aerial Amphibious Robots in Wind Turbine Inspections*, by Hai Wang, Qingxiang Wu, Zhi-Xin Yang, and Ning Sun.

| Directory | Manuscript experiment |
| --- | --- |
| [ablation](ablation) | DDQN/PER/curriculum training ablation (Table I) and event-triggered replanning ablation (Table II) |
| [examples](examples) | Controlled occupancy change and aerial path deviation (Fig. 4) |
| [real_scale](real_scale) | DDQN+PER training, deployment, and comparison on 20 realistic-scale scenarios (Table III) |
| [reference_results](reference_results) | Archived per-seed or per-scenario metrics for checking the reported results |

## Installation

Use a virtual environment and install the dependencies from the repository root.

    python -m venv .venv

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell, then run:

    python -m pip install -r requirements.txt

The release was checked with Python 3.13.1 and PyTorch 2.8.0. The quick runs below use CPU. For full training, pass `--device cuda` with a CUDA-enabled PyTorch installation, or use `--device auto` to select the available device. For deterministic CUDA training, set `CUBLAS_WORKSPACE_CONFIG` to `:4096:8` before launching Python. Reported wall-clock times depend on the machine and runtime.

## Training and replanning ablations

From the repository root:

    cd ablation
    python check_ablation_environment.py
    python check_reward_and_fixed_eval.py
    python check_curriculum_protocol_v3.py

Run the training ablation (four variants, seeds 10–29, 400 episodes each) and the replanning ablation using the resulting DDQN+PER checkpoints:

    python run_paper_ablation.py --stage all --device auto

All variants are evaluated with a greedy rollout on the same fixed, full-complexity map every 20 episodes. The curriculum changes the training environment schedule. For the replanning ablation, each DDQN+PER checkpoint is evaluated under no replanning, periodic replanning, and event-triggered replanning with the same imposed path deviation. Checkpoints that fail the nominal task are recorded in the excluded-seeds CSV.

To run the replanning ablation without retraining, copy the supplied DDQN+PER checkpoints into a separate results directory first:

    python -c "from shutil import copytree; copytree('reference_checkpoints/training', 'results/replanning/training', dirs_exist_ok=True)"
    python run_paper_ablation.py --stage replanning --device cpu --results results/replanning

The first command copies only the checkpoints; training curves and reference metrics are available separately under `reference_results/ablation`. The ablation runner generates CSV summaries, paired tests, plots, and LaTeX tables. Use a fresh `--results` directory for runs with changed settings.

## Representative replanning cases

From the repository root, run the occupancy-change and path-deviation cases using the included seed-10 policy:

    python examples/occupancy_change/run_simulation.py
    python examples/path_deviation/run_simulation.py

Each script writes reference paths, executed paths, and metrics to its own `results/` directory. The occupancy change is a supplied map update, and the path deviation is an imposed displacement of `[2, 2, -2]` voxels. See each example's README for the output used in Fig. 4.

## Realistic-scale comparison

Run each command block from the repository root in a separate shell, or return to the repository root after each block. The benchmark uses a shared, frozen manifest of 20 scenarios (scenario seeds 1001–1020) at 5 m per voxel. Keep the manifests unchanged when comparing planners.

### DDQN+PER

    cd real_scale/ddqn_per
    python validate_scene.py
    python plan_real_scale.py --checkpoint checkpoints/proposed_validation_selected.pt --device cpu --scenario-id test_01
    python plan_real_scale.py --checkpoint checkpoints/proposed_validation_selected.pt --device cpu

The first planning command runs a single scenario. The second runs all 20 and writes the full paired results. For training from scratch, run the following from `real_scale/training`:

    python train_real_scale.py --seed 10 --device auto
    python train_real_scale.py --seed 11 --device auto
    python train_real_scale.py --seed 12 --device auto
    python train_real_scale.py --seed 13 --device auto
    python train_real_scale.py --seed 14 --device auto
    python select_deployment_checkpoint.py

From `real_scale/ddqn_per`, pass `--checkpoint ../training/deployment_checkpoint/proposed_validation_selected.pt` to evaluate the newly selected model. The selector ranks checkpoints using the fixed validation suite. The feasibility mask was introduced after an earlier evaluation on the frozen 20-scenario benchmark. This manifest supports paired comparisons and should be described as a fixed benchmark.

### Fan et al. and Zhang et al.

    cd real_scale/fan_2019
    python validate_fan_project.py
    python run_fan_20_scenarios.py --scenario-id test_01
    python run_fan_20_scenarios.py

    cd real_scale/zhang_2022
    python validate_project.py
    python run_zhang2022_20_scenarios.py --scenario-id test_01
    python run_zhang2022_20_scenarios.py

These are Python benchmark adaptations. Fan et al. refers to *Autonomous Hybrid Ground/Aerial Mobility in Unknown Environments* (IROS 2019). Zhang et al. refers to *Autonomous and Adaptive Navigation for Terrestrial-Aerial Bimodal Vehicles* (RA-L 2022). The Zhang adaptation was informed by the [ZJU-FAST-Lab TABV project](https://github.com/ZJU-FAST-Lab/Terrestrial-Aerial-Navigation), whose upstream ROS/C++ source is released under GPL-3.0. The corresponding implementation notes and source provenance are included alongside both adaptations.

### Summarize paired results

After all three planners have run the full benchmark, use the following from `real_scale`:

    python compare_three_methods.py --proposed ddqn_per/results/proposed_ddqn_per_20_scenarios/paired_results_proposed.csv --fan fan_2019/results/fan_hybrid_20_scenarios/paired_results_fan.csv --zhang zhang_2022/results/zhang2022_tabv_20_scenarios/paired_results_zhang2022.csv --output results/three_method_comparison

The same script accepts the archived CSVs in `../reference_results/real_scale`. Those CSVs preserve the benchmark metrics and omit the original machine's checkpoint path.

## Citation

Use [CITATION.cff](CITATION.cff) to cite the simulation code and cite the accompanying manuscript when using the results.
