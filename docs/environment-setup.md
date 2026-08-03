# Setting Up EnerTune

One-time setup on the machine that runs the experiments. We assume an A100 80GB GPU with CUDA 12.9 and Driver 575.57.08.

### 1. CUDA

EnerTune talks to the GPU through CUDA and NVML, so the CUDA runtime and its Python bindings need to be installed and matched to your toolkit.
```
$ python -m pip uninstall -y cuda || true
$ python -m pip install --no-cache-dir "cuda-python>=12,<13"
```

### 2. Python environment

All of EnerTune's Python (profiling, scheduling, serving) runs out of a single venv.
```
$ sudo apt install python3-venv
$ python3 -m venv <venv-path>
$ source <venv-path>/bin/activate
$ pip3 install -r requirements.txt
```

### 3. Monitoring daemon

`gmonitor` is a small C++ daemon that samples per-GPU power and utilization through NVML. EnerTune runs it during experiments to measure energy. Build it once:
```
$ cd profiler
$ make all
```
It links against `libnvidia-ml` and the CUDA headers. If your CUDA is not under `/usr/local/cuda-12.3`, update the `INCLUDES` path in `profiler/Makefile` first.

### 4. GPU sharing (MIG and MPS)

Our experiments share the GPU two different ways: some partition it with MIG, others multiplex it with MPS. You will need to turn each on and off as you move between experiments, so we ship bash helpers in `helper.sh` to do it. Source it first:
```
$ source helper.sh
```

**Enabling / disabling MIG.** MIG mode is a device-level setting that requires a reboot, so flip it once before running any MIG experiment:
```
$ is_mig_feature_available            # nonzero = MIG-capable (4 on our 4-GPU box)
$ assert_mig_status mig <device-id>   # if "MIG mode not enabled", continue below
$ sudo nvidia-smi -mig 1
$ sudo reboot now
```
After rebooting, confirm it stuck (no output means enabled):
```
$ assert_mig_status mig <device-id>
```
The run scripts then carve out and clean up the actual slices per run via `setup_mig_if_needed` and `cleanup_mig_if_needed`. To turn MIG back off at the device level:
```
$ sudo nvidia-smi -mig 0
$ sudo reboot now
```

**Enabling / disabling MPS.** MPS needs no reboot. `enable_mps_if_needed` puts the GPU in `EXCLUSIVE_PROCESS` mode and starts the MPS control daemon; `disable_mps_if_needed` stops the daemon and resets the GPU. The run scripts call these for you:
```
$ enable_mps_if_needed <mode> <device-id>
$ disable_mps_if_needed <mode> <device-id>
```

### 5. Export run-time variables

Our scripts need `sudo` to set MIG slices, MPS, and GPU frequency, and they source your venv on every run.
```
$ export USE_SUDO=1
$ export VENV=<venv-path>
```

You're set. See the other docs in `docs/` for each experiment.
