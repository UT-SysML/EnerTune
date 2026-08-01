# Beyond Utilization: Energy-Conscious GPU Sharing for Inference Serving
This is the official repository for [EnerTune]() at SOSP'26. EnerTune is an inference serving system that reduces energy consumption while meeting performance SLOs. EnerTune introduces analytical models to estimate per-model performance and power, and the power draw of colocated models on shared GPUs, and uses them in an energy-aware bin-packing algorithm to jointly determine model placement and configuration. EnerTune meets performance SLOs while reducing energy consumption by 1.4-2.3× and power draw by 1.3-2.6× over state-of-the-art baselines, including Usher, FGD, GPUlets, and ParvaGPU.

Our contributions include:
- We develop analytical models that capture the impact of GPU allocation size, frequency, and batch size on per-model power. Using these models, our energy-conscious profiling reduces profiling time by 7.3× on average compared to prior systems while retaining high accuracy. 
- We introduce a novel analytical model that accurately estimates the power draw of spatially multiplexed GPUs from individual model power profiles. Our methodology eliminates tens of hours of joint profiling overhead and enables EnerTune to reduce power and energy consumption by 1.3× over prior work.
- We develop an energy-aware bin-packing algorithm and build EnerTune on top of PyTorch. Compared to four state-of-the-art performance-driven baselines, EnerTune reduces energy and power draw by 1.4-2.3× and 1.3-2.6×, respectively, without violating SLOs, across load.

### Hardware and CUDA Requirements
We conduct our experiments on an A100 80GB GPU. CUDA Version is 12.9 and Driver Version is 575.57.08. Ensure CUDA is installed: 
```
$ python -m pip uninstall -y cuda || true
$ python -m pip install --no-cache-dir "cuda-python>=12,<13"
```

### Setup Environment

Install all python requirements in a python virtual environment:
```
$ sudo apt install python3-venv
$ python3 -m venv <venv-path>
$ source <venv-path>/bin/activate
$ pip3 install -r requirements.txt
```
Compile the C++ per-GPU monitoring daemon
```
$ cd profiler
$ make all
```

Enable Multi-Instance GPUs (MIG). This is the hardware mechanism that EnerTune uses to share GPUs. 
```
$ source helper.sh
$ is_mig_feature_available # should see 4
$ assert_mig_status mig <device id (e.g., 0, 1, 2, 3) # if "MIG mode not enabled", continue below
$ sudo nvidia-smi -mig 1 # Might need to insert sudo before
$ sudo reboot now
```
Once rebooted, confirm MIG is enabled
```
$ assert_mig_status mig <device id (e.g., 0, 1, 2, 3) # if working correctly, there should be no output
```

### How to run experiments

We require SUDO to set MIG slices, MPS, and adjust GPU frequency.
```
$ export USE_SUDO=1
````

Indicate to our scripts where the python virtual environment lives
```
$ export VENV=<venv-path>
```
Run scripts for each baseline to reproduce E2E results. For easier reproducability efforts, we have adapted our scripts to use just a single GPU, running each set of models that would be placed on a GPU one at a time. This reduces the number of GPUs required for reproducability efforts. 

Run each system at 5 load levels (25% to 125% load). In `systems/`, there is a directory per system: EnerTune, FGD, GPULets, ParvaGPU, and Usher. 

For each baseline, run the following: 
```
$ cd systems/{baseline} # either fgd, gpulets, parva, usher
$ cd load-{load_level} # either 25, 50, 75, 100, or 125
$ load-{load_level}-{baseline}.sh
```

For EnerTune, run the following:

```
$ cd systems/ener-tune/{optimization_metric} # power, energy, or carbon
$ cd load-{load_level} # either 25, 50, 75, 100, or 125
$ load-{load_level}-et-{optimization_metric}.sh # power, energy, or carbon
```

If running smoothly, you will see the following
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

