# Measuring EnerTune's Minimal Overheads

We evalutae the overheads of EnerTune:
1. Time for the Colocation Power Estimator to make its power estimates
2. Time to change GPU frequency
3. Time to change GPU allocation size per model
4. Time for the Resource Planner to make planning decisions
5. Time to load models (not an artifact of EnerTune)

Ensure you have an A100 80GB GPU with `sudo` access (refer to [`environment-setup.md`](./environment-setup.md). Overhead analysis can be reproduced using `analysis/overhead_analysis.py`: 
```
$ source <venv-path>/bin/activate
$ cd analysis
$ python overhead_analysis.py
```
If running smoothly, you will see the output directly on `stdout`. 
