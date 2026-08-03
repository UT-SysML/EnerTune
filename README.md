# Beyond Utilization: Energy-Conscious GPU Sharing for Inference Serving
This is the official repository for [EnerTune]() at SOSP'26. EnerTune is an inference serving system that reduces energy consumption while meeting performance SLOs. EnerTune introduces analytical models to estimate per-model performance and power, and the power draw of colocated models on shared GPUs, and uses them in an energy-aware bin-packing algorithm to jointly determine model placement and configuration. EnerTune meets performance SLOs while reducing energy consumption by 1.4-2.3× and power draw by 1.3-2.6× over state-of-the-art baselines, including Usher, FGD, GPUlets, and ParvaGPU.

Our contributions include:
- We develop analytical models that capture the impact of GPU allocation size, frequency, and batch size on per-model power. Using these models, our energy-conscious profiling reduces profiling time by 7.3× on average compared to prior systems while retaining high accuracy. 
- We introduce a novel analytical model that accurately estimates the power draw of spatially multiplexed GPUs from individual model power profiles. Our methodology eliminates tens of hours of joint profiling overhead and enables EnerTune to reduce power and energy consumption by 1.3× over prior work.
- We develop an energy-aware bin-packing algorithm and build EnerTune on top of PyTorch. Compared to four state-of-the-art performance-driven baselines, EnerTune reduces energy and power draw by 1.4-2.3× and 1.3-2.6×, respectively, without violating SLOs, across load.

## Reproducibility

We conduct our experiments on an A100 80GB GPU (CUDA 12.9, Driver 575.57.08). The docs below walk through reproducing each result in the paper. Start with the environment setup, run the experiments you care about, then plot.

**Setup.** Install CUDA, the python requirements, and the monitoring daemon, and enable GPU sharing (MIG/MPS): [`docs/environment-setup.md`](./docs/environment-setup.md).

**Experiments.** Each doc explains what it reproduces, which scripts to run, and where results land:
- [`docs/eval-e2e.md`](./docs/eval-e2e.md) — end-to-end energy savings and SLO attainment vs. the baselines.
- [`docs/eval-ablation.md`](./docs/eval-ablation.md) — Colocation Power Estimator and Resource Planner ablations.
- [`docs/eval-profiling.md`](./docs/eval-profiling.md) — profiling cost and estimation accuracy.
- [`docs/eval-robustness.md`](./docs/eval-robustness.md) — robustness to bursty (Poisson) arrivals.
- [`docs/eval-overheads.md`](./docs/eval-overheads.md) — EnerTune's runtime overheads.

**Plotting.** The experiment scripts write raw per-run output under `results/<system>-results/`. First aggregate it into per-system CSVs:
```
$ python results/results-aggregator.py   # writes results/eval-results/<system>-results.csv
```
Then `analysis/analysis.py` turns those into the paper figures — `e2e_plot()`, `ablation_estimator()`, `ablation_placement_only()`, `estimation_accuracy()`, `profiling_cost()`, and `robustness_arrival()` (overhead figures live in `analysis/overhead_analysis.py`). Uncomment the figure you want at the bottom of the file, then:
```
$ cd analysis && python analysis.py
```
Each function reads its aggregated results and writes a PDF to `analysis/plots/`.

