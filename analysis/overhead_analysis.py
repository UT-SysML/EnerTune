#!/usr/bin/env python3

import platform
import statistics
import timeit


def power_estimation_overhead():
    STATIC_PWR_DRAWS = {
        210: 22.0, 240: 22.1, 330: 22.5, 420: 22.9, 510: 23.2,
        600: 23.5, 690: 21.7, 780: 21.8, 870: 23.4, 960: 24.4,
        1050: 25.4, 1140: 27.1, 1230: 32.1, 1320: 37.56, 1410: 44.7,
    }
    FAN_PWR = 20.4
    IDLE_PWR_DRAWS = {
        210: 1.3904760350877268, 240: 1.4434391811286673, 330: 1.4701607823834166,
        420: 1.420075588744595, 510: 1.6442643070866154, 600: 1.6466100275229195,
        690: 1.660922408376825, 780: 2.0857701904761882, 870: 2.1179469517543907,
        960: 2.1617198775510076, 1050: 3.2587632598039136, 1140: 3.8781596666666474,
        1230: 4.781902733333311, 1320: 7.284372394821938, 1410: 8.333536788990841,
    }
    REPRESENTATIVE_POWERS = [22.83, 3.78, 16.65, 4.10]
    FREQUENCY = 690
    MODEL_COUNTS = [2, 3, 4]
    ITERS = 100_000
    REPEATS = 7

    def power_estimation(frequency, power_list):
        return STATIC_PWR_DRAWS[frequency] + FAN_PWR + IDLE_PWR_DRAWS[frequency] + sum(power_list)

    print("EnerTune power-estimation timing")
    print("-" * 60)
    print(f"Python       : {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform     : {platform.platform()}")
    print(f"Processor    : {platform.processor() or 'unknown'}")
    print(f"Frequency    : {FREQUENCY} MHz")
    print(f"iters/measure: {ITERS:,}   repeats: {REPEATS}")
    print("-" * 60)
    print(f"{'Models':>6} | {'min (us)':>10} | {'median (us)':>11} | {'max (us)':>9} | {'est power (W)':>13}")
    print("-" * 60)
    for k in MODEL_COUNTS:
        power_list = REPRESENTATIVE_POWERS[:k]
        for _ in range(1000):
            power_estimation(FREQUENCY, power_list)
        runs = timeit.repeat(
            lambda: power_estimation(FREQUENCY, power_list),
            number=ITERS,
            repeat=REPEATS,
        )
        per_call_us = [(t / ITERS) * 1e6 for t in runs]
        est = power_estimation(FREQUENCY, power_list)
        print(f"{k:>6} | {min(per_call_us):>10.4f} | {statistics.median(per_call_us):>11.4f} | "
              f"{max(per_call_us):>9.4f} | {est:>13.3f}")
    print("-" * 60)
    print("Reported time = per single power_estimation() call. 'min' is the most stable estimate.")


def model_loading_overhead():
    import time
    import torch
    from transformers import AutoTokenizer, GPTJForCausalLM, pipeline
    from diffusers import DiffusionPipeline

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_cnn(model_name):
        def _load():
            model = torch.hub.load(
                "pytorch/vision:v0.14.1",
                model_name,
                verbose=False,
                pretrained=True,
            )
            model.eval()
            model.to(device)
            return model
        return _load

    def load_stable_diffusion():
        return DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)

    def load_whisper():
        return pipeline(
            "automatic-speech-recognition",
            "openai/whisper-small",
            device=device,
        )

    def load_gpt_j_6b():
        model_path = "EleutherAI/gpt-j-6B"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model = GPTJForCausalLM.from_pretrained(
            model_path,
            revision="float16",
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True,
        ).to(device)
        model.resize_token_embeddings(len(tokenizer))
        return model

    models = [
        ("resnet50", load_cnn("resnet50")),
        ("mobilenet_v3_large", load_cnn("mobilenet_v3_large")),
        ("vgg11", load_cnn("vgg11")),
        ("resnet152", load_cnn("resnet152")),
        ("densenet169", load_cnn("densenet169")),
        ("densenet161", load_cnn("densenet161")),
        ("vgg19", load_cnn("vgg19")),
        ("diffusion_1024_1024", load_stable_diffusion),
        ("diffusion_256_256", load_stable_diffusion),
        ("whisper", load_whisper),
        ("gpt", load_gpt_j_6b),
    ]

    def free(model):
        del model
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    print("EnerTune model-loading timing")
    print("-" * 60)
    print(f"Torch        : {torch.__version__}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print("-" * 60)
    print(f"{'Model':>20} | {'load (s)':>10}")
    print("-" * 60)
    for name, loader in models:
        warm = loader()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        free(warm)
        start = time.perf_counter()
        model = loader()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        print(f"{name:>20} | {elapsed:>10.3f}")
        free(model)
    print("-" * 60)
    print("load (s) = warm-cache time for load_model() as in src/inference.py (one untimed warm-up pass precedes each measurement).")


def frequency_change_overhead():
    import subprocess
    import statistics
    import time

    device_id = 0
    frequencies = [1410, 210, 690, 1050, 900, 300]
    repeats = 5

    def set_frequency(freq):
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(device_id), "-lgc", f"{freq},{freq}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    def reset_frequency():
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(device_id), "-rgc"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    print("EnerTune GPU frequency-change timing")
    print("-" * 60)
    print(f"Device       : cuda:{device_id}")
    print(f"Command      : nvidia-smi -i {device_id} -lgc <f>,<f>")
    print(f"repeats      : {repeats}")
    print("-" * 60)
    print(f"{'freq (MHz)':>10} | {'min (ms)':>10} | {'median (ms)':>11} | {'max (ms)':>10}")
    print("-" * 60)
    set_frequency(frequencies[0])
    for freq in frequencies:
        samples_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            set_frequency(freq)
            samples_ms.append((time.perf_counter() - start) * 1e3)
        print(f"{freq:>10} | {min(samples_ms):>10.1f} | {statistics.median(samples_ms):>11.1f} | {max(samples_ms):>10.1f}")
    reset_frequency()
    print("-" * 60)
    print(f"time (ms) = one nvidia-smi -lgc call as in helper.sh set_gpu_freq(); target is < 100 ms.")


def mig_allocation_overhead():
    import subprocess
    import statistics
    import time

    device_id = 0
    slice_to_gi = {1: 19, 2: 14, 3: 9, 4: 5, 7: 0}
    allocations = [[7], [3, 3], [2, 2, 2], [1, 1, 1, 1, 1, 1, 1]]
    repeats = 3

    def mig(args):
        return subprocess.run(
            ["sudo", "nvidia-smi", "mig", "-i", str(device_id), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def mig_enabled():
        out = subprocess.run(
            ["nvidia-smi", "-i", str(device_id),
             "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.stdout.strip() == "Enabled"

    def create_allocation(slices):
        gi_string = ",".join(str(slice_to_gi[s]) for s in slices)
        mig(["-cgi", gi_string])
        gi_ids = []
        for line in mig(["-lgi"]).stdout.splitlines():
            fields = line.split()
            if len(fields) >= 6 and fields[5].isdigit():
                gi_ids.append(fields[5])
        for gi_id in gi_ids:
            ci_profile = None
            for line in mig(["-gi", gi_id, "-lcip"]).stdout.splitlines():
                if "*" in line:
                    fields = line.split()
                    if len(fields) >= 6:
                        ci_profile = fields[5].strip("*")
                        break
            if ci_profile is not None:
                mig(["-gi", gi_id, "-cci", ci_profile])

    def destroy_allocation():
        gi_ids = []
        ci_ids = []
        for line in mig(["-lci"]).stdout.splitlines():
            fields = line.split()
            if len(fields) >= 7 and fields[2].isdigit() and fields[6].isdigit():
                gi_ids.append(fields[2])
                ci_ids.append(fields[6])
        for ci_id, gi_id in zip(ci_ids, gi_ids):
            mig(["-dci", "-ci", ci_id, "-gi", gi_id])
        mig(["-dgi"])

    print("EnerTune MIG allocation-change timing")
    print("-" * 72)
    print(f"Device       : cuda:{device_id}")
    print(f"Commands     : nvidia-smi mig -cgi/-cci (create), -dci/-dgi (destroy)")
    print(f"repeats      : {repeats}")
    print("-" * 72)
    if not mig_enabled():
        print(f"MIG mode is not Enabled on GPU {device_id}; enable it first:")
        print(f"    sudo nvidia-smi -i {device_id} -mig ENABLED")
        print("-" * 72)
        return
    print(f"{'allocation':>18} | {'create min':>11} | {'create med':>11} | {'destroy min':>12} | {'destroy med':>12}")
    print(f"{'(GPU slices)':>18} | {'(ms)':>11} | {'(ms)':>11} | {'(ms)':>12} | {'(ms)':>12}")
    print("-" * 72)
    destroy_allocation()
    for slices in allocations:
        create_ms = []
        destroy_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            create_allocation(slices)
            create_ms.append((time.perf_counter() - start) * 1e3)
            start = time.perf_counter()
            destroy_allocation()
            destroy_ms.append((time.perf_counter() - start) * 1e3)
        label = "+".join(str(s) for s in slices)
        print(f"{label:>18} | {min(create_ms):>11.1f} | {statistics.median(create_ms):>11.1f} | "
              f"{min(destroy_ms):>12.1f} | {statistics.median(destroy_ms):>12.1f}")
    print("-" * 72)
    print("time (ms) = create/destroy of a MIG allocation as in helper.sh setup/cleanup_mig_if_needed(); target is < 200 ms.")


if __name__ == "__main__":
    # power_estimation_overhead()
    # print()
    # model_loading_overhead()
    # print()
    frequency_change_overhead()
    print()
    mig_allocation_overhead()
    print()
