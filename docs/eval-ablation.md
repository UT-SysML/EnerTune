# Ablation Studies

We run two ablations, both on EnerTune. Since both use MIG, make sure MIG is enabled and the run-time variables are exported first (see [`environment-setup.md`](./environment-setup.md)):
```
$ export USE_SUDO=1
$ export VENV=<venv-path>
$ source helper.sh
```

### 1. Colocation Power Estimator

This isolates how much our estimator helps by swapping it for a naive additive baseline while keeping the rest of EnerTune fixed. There is an energy variant and a power variant; run every load level (25% to 125%) for each.
```
$ cd systems/ener-tune/estimator-energy-ablation/load-<load_level>   # load_level: 25, 50, 75, 100, 125
$ ./load-<load_level>-et-energy.sh
```
```
$ cd systems/ener-tune/estimator-power-ablation/load-<load_level>
$ ./load-<load_level>-et-power.sh
```
If running smoothly, you will see the per-mix progress on `stdout`, like below: 
```
System: {system} | Running mix: {job_mix} | Device: {device_id}
Logs at /tmp/print_outs-{random-id}.txt
Running point experiment for mig with round #0
Made it before while loop for duration arg
Duration is 0
Run success: results are stored in results/a100/{job_mix_path}
Exiting with error_code=0 (0 is clean exit)
Examine /tmp/print_outs-{random-id}.txt for logs
Completed {job_mix} on GPU {device_id} with {frequency} MHz for system {system}.
```
Results for each system will be written under `results/`, with a directory for each: `et-energy-estimator-ablation-results` and `et-pwr-estimator-ablation-results`. 

### 2. Resource Planner

This isolates the planner by running placement only, without the rest of EnerTune's frequency scaling. We evaluate it at 100% load. Pick the metric you are optimizing for (`energy`, `power`, or `carbon`):
```
$ cd systems/ener-tune/placement-only-<metric>/load-100
$ ./load-100-et-placement-only-<metric>.sh
```
If running smoothly, you will see the per-mix progress on `stdout`, like below: 
```
System: {system} | Running mix: {job_mix} | Device: {device_id}
Logs at /tmp/print_outs-{random-id}.txt
Running point experiment for mig with round #0
Made it before while loop for duration arg
Duration is 0
Run success: results are stored in results/a100/{job_mix_path}
Exiting with error_code=0 (0 is clean exit)
Examine /tmp/print_outs-{random-id}.txt for logs
Completed {job_mix} on GPU {device_id} with {frequency} MHz for system {system}.
```
Results for each system will be written under `results/`, with a directory for each: `et-placement-only-power-results`, `et-placement-only-energy-results`, and `et-placement-only-carbon-results`. 

Once the runs finish, aggregate them and plot with `ablation_estimator()` (estimator study) and `ablation_placement_only()` (planner study) — see [Plotting](../README.md#reproducibility) in the README.

