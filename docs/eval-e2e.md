# End-to-End Evaluation

We reproduce EnerTune's end-to-end energy savings and SLO attainment against four baselines. Every system has its own scripts under `systems/`, one directory per load level (`load-25` through `load-125`, i.e. 25% to 125% load).

The systems share GPUs two different ways:
- **MPS:** Usher, GPULets
- **MIG:** EnerTune, FGD, ParvaGPU

MPS needs no reboot but MIG does, so **start with the MPS systems (Usher and GPULets) while MIG is off, then enable MIG once and run the rest.** This keeps you to a single reboot.

First export the run-time variables and load the helpers (see [`environment-setup.md`](./environment-setup.md)):
```
$ export USE_SUDO=1
$ export VENV=<venv-path>
$ source helper.sh
```

### Usher and GPULets (MPS)

Make sure MIG is off and MPS is up (the scripts call `enable_mps_if_needed` for you; see [`environment-setup.md`](./environment-setup.md) if you want to do it by hand). Then run every load level for each baseline:
```
$ cd systems/<baseline>/load-<load_level>   # baseline: usher, gpulets   load_level: 25, 50, 75, 100, 125
$ ./load-<load_level>-<baseline>.sh
```

### FGD and ParvaGPU (MIG)

Enable MIG (this needs the reboot from [`environment-setup.md`](./environment-setup.md)), then run each baseline the same way:
```
$ cd systems/<baseline>/load-<load_level>   # baseline: fgd, parva   load_level: 25, 50, 75, 100, 125
$ ./load-<load_level>-<baseline>.sh
```

### EnerTune (MIG)

EnerTune also uses MIG, so keep MIG enabled. Pick the metric you are optimizing for (`energy`, `power`, or `carbon`) and run each load level:
```
$ cd systems/ener-tune/<metric>/load-<load_level>
$ ./load-<load_level>-et-<metric>.sh
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
Results for each system will be written under `results/`, with a directory per system. 

Once the runs finish, aggregate them and plot with `e2e_plot()` — see [Plotting](../README.md#reproducibility) in the README.
