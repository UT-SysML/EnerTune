# Profiling Evaluation

We measure two things about EnerTune's profiling: its **accuracy** and its **cost**. Analyzing both requires tens of hours of data collection. Hence, to relieve the user, we include all the data for these experiments in `data/`. Our analysis lives in `analysis/analysis.py`, and each figure is just the single function call at the bottom of that file. Set that call to the function you want and run it with the venv from [`environment-setup.md`](./environment-setup.md):
```
$ source <venv-path>/bin/activate
$ cd analysis
$ mkdir -p plots
$ python analysis.py     # runs whichever function is called at the bottom of the file
```

### Profiling accuracy

We include the ground-truth data from brute-force profiling individual models and whole model sets so you don't have to spend the hours it takes to collect it; it's all under `data/`. The function `estimation_accuracy()` in `anslysis/analysis.py` then uses our analytical formulations to estimate each model's power draw under different deployment configurations (GPU allocation, batch size, GPU frequency), and our Colocation Power Estimator to estimate the power draw of different model sets. The latter we also compare against the ground truth and against a naive Additive Power Draw baseline.

Set the bottom call to `estimation_accuracy()` and run `analysis.py`. The figure is written to `analysis/plots/estimation-accuracy.pdf`.

### Profiling cost

`profiling_cost()` computes how long, and how much energy, it takes each system to profile its search space. EnerTune's energy-conscious profiling samples a small fraction of configurations instead of brute-forcing the full grid, which cuts profiling time and energy over brute force and prior systems (Usher, GPULets, ParvaGPU). It reuses the same per-model power profiles plus the GPU kernel traces in `data/gputraces/`, so again no reprofiling is needed.

Set the bottom call to `profiling_cost()` and run `analysis.py`. The figure is written to `analysis/plots/profiling-cost.pdf`.
