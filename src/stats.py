import argparse
import os
import sys
import pickle
import atexit
import fcntl
import numpy as np
import pandas as pd

TPUT = "tput"
TOTAL_PREFIX = "total"
QUEUED_PREFIX = "queued"
PROCESSING_PREFIX = "processing"
PERCENTILES = [0, 10, 25, 50, 75, 90, 95, 99, 100]

METRIC_NAMES= [TPUT]
for prefix in [TOTAL_PREFIX, QUEUED_PREFIX, PROCESSING_PREFIX]:
    for percentile in PERCENTILES:
        METRIC_NAMES.append(f"{prefix}_p{percentile}")

def acquire_lock():
    print("Attempting to acquire stat lock")
    global lock_fd
    lock_fd = open(sys.argv[0], 'r+')
    fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Acquire an exclusive lock
    print("Acquired stat lock")

def release_lock():
    fcntl.flock(lock_fd, fcntl.LOCK_UN)  # Release the lock
    lock_fd.close()

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def populate_stats(id, array, tid, metrics):
    # Calculate percentiles
    percentile_metrics = np.percentile(array, PERCENTILES)
    for i, percentile in enumerate(PERCENTILES):
        metrics[f"{id}_p{percentile}"][tid] = percentile_metrics[i] * 1000

def create_dir(directory):
    try:
        os.makedirs(directory)
    except:
        pass

if __name__=="__main__":
    acquire_lock()
    atexit.register(release_lock)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--load", type=float, required=True)
    parser.add_argument("--min-freq", type=int, required=True)
    parser.add_argument("--max-freq", type=int, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument(
        "pickle_files",
        metavar="<pkl_file>",
        type=str,
        nargs="+",
        help="List of model and details"
    )
    opt = parser.parse_args()

    # Create result directory
    create_dir(opt.result_dir)

    models = [None] * len(opt.pickle_files)
    metrics = {}
    for metric_name in METRIC_NAMES:
        metrics[metric_name] = [None] * len(opt.pickle_files)
    

    for pickle_file in opt.pickle_files:
        # Validate file paths
        if not os.path.isfile(pickle_file):
            print(f"File '{pickle_file}' does not exist.")
            sys.exit(1)
        
        # Load arrays from pickle files
        tid, mig_slice, pid, infer_stats = load_pickle_file(pickle_file)
        model, start_time, end_time, rps, reqs_completed, tput, total_times, queued_times = infer_stats
        print(model)
        print(start_time)
        print(end_time)
        print(rps)
        print(reqs_completed)
        print(tput)
        print(total_times)
        print(queued_times)
        populate_stats(TOTAL_PREFIX, total_times, tid, metrics)
        populate_stats(QUEUED_PREFIX, queued_times, tid, metrics)
        processing_times = [x - y for x, y in zip(total_times, queued_times)]
        populate_stats(PROCESSING_PREFIX, processing_times, tid, metrics)
        metrics[TPUT][tid] = tput
        if models[tid] is None:
            models[tid] = f"{tid}-{model}-{start_time}-{end_time}-{mig_slice}-{reqs_completed}-{pid}-{rps}"
        else:
            assert models[tid] == f"{tid}-{model}-{start_time}-{end_time}-{mig_slice}-{reqs_completed}-{pid}-{rps}"

    # Create dataframe for each metric type
    models = [x for x in models if x is not None]
    for metric_type, metrics_list in metrics.items():
        modes = [opt.mode] * len(models)
        loads = [opt.load] * len(models)
        jobs = [x.split("-")[0] for x in models]
        mig_slices = [x.split("-")[5] for x in models]
        model_names = [x.split("-")[1] for x in models]
        batch_sizes = [x.split("-")[2] for x in models]
        metric_values = metrics_list
        metric_types = [metric_type] * len(models)
        start_times = [x.split("-")[3] for x in models]
        end_times = [x.split("-")[4] for x in models]
        completed_reqs = [x.split("-")[6] for x in models]
        pids = [x.split("-")[7] for x in models]
        rpss = [x.split("-")[8] for x in models]
        min_freqs = [opt.min_freq] * len(models)
        max_freqs = [opt.max_freq] * len(models)

        df = pd.DataFrame()
        df['mode'] = modes
        df['mig_slices'] = mig_slices
        df['load'] = loads
        df['job_no'] = jobs
        df['model'] = model_names
        df['batch_size'] = batch_sizes
        df['metric_value'] = metric_values
        df['metric_type'] = metric_types
        df['start_time'] = start_times
        df['end_time'] = end_times
        df['reqs_completed'] = completed_reqs
        df['pkl_file_key'] = pids
        df['min_freq'] = min_freqs
        df['max_freqs'] = max_freqs
        df['rps'] = rpss
        csv_file = os.path.join(opt.result_dir, f"{metric_type}.csv")
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)
