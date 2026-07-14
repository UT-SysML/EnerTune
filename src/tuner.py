import pandas as pd
import numpy as np
import csv
import copy
import itertools

STATIC_PWR_DRAWS = {
    210: 42.4,
    240: 42.5,
    330: 42.9,
    420: 43.3,
    510: 43.6,
    600: 43.9,
    690: 42.1,
    780: 42.2,
    870: 43.8,
    960: 44.8,
    1050: 45.8,
    1140: 47.5,
    1230: 52.5,
    1320: 57.96,
    1410: 65.1,
}
IDLE_PWR_DRAWS = {
    210: 1.3904760350877268, 
    240: 1.4434391811286673, 
    330: 1.4701607823834166, 
    420: 1.420075588744595, 
    510: 1.6442643070866154, 
    600: 1.6466100275229195, 
    690: 1.660922408376825, 
    780: 2.0857701904761882, 
    870: 2.1179469517543907, 
    960: 2.1617198775510076, 
    1050: 3.2587632598039136, # 3.2587632598039136
    1140: 3.8781596666666474, 
    1230: 4.781902733333311,
    1320: 7.284372394821938, # 5.9404681052631361
    1410: 8.333536788990841,
}

PUE = 1.1
CI = 380
CI = 50
LT = 4 * 352 * 24 * 60 * 60
C_M = 1/0.9 * (583 * 1.52 + 350 + 500) * 8.26 + 66.60 * 80 + 2 * 150

BASE_DIR = f"/home/ps35324/sus-gpus/motivation/batch-size/aggregated-power-data"
MODEL_DATABASE = f'{BASE_DIR}/model_datastore.csv'
FREQ_SETTING = 1410

def cost_estimator(frequency, power_list, latency_list, cost_metric='power'):
    """
    Function to calclulate deployment cost based on cost metric

    Input:
        - frequency (int): Slected Frequency
        - power_list (list): List of power draw per model
        - latency_list (list): List of latencies per model
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])

    Output:
        - (float): Cost Estimation
    """
    if cost_metric == 'power':
        return STATIC_PWR_DRAWS[frequency] + IDLE_PWR_DRAWS[frequency] + sum(power_list)
    elif cost_metric == 'energy':
        pairs = sorted(zip(latency_list, power_list))
        lat_sorted, pwr_sorted = zip(*pairs)
        lat_sorted = list(lat_sorted)
        pwr_sorted = list(pwr_sorted)

        energy = 0.0
        for i in range(len(pwr_sorted)):
            remaining_sum_pwr = STATIC_PWR_DRAWS[frequency] + IDLE_PWR_DRAWS[frequency] + sum(pwr_sorted[i:])
            dt = lat_sorted[0] if i==0 else lat_sorted[i] - lat_sorted[i-1]
            energy += remaining_sum_pwr * dt
        return energy
    elif cost_metric == 'carbon':
        E = cost_estimator(frequency, power_list, latency_list, cost_metric='energy')
        T = max(latency_list)

        C_O = PUE * CI * E
        C_E = (T / LT) * C_M
        return C_O + C_E

def serch_config_tuning(model_name, latency_slo, throughput_slo, freq_target, alloc, direction, csv_path=MODEL_DATABASE):
    """
    Function to find all feasible configurations for a model, subject to:
      - given model_name
      - latency <= latency_slo_ms
      - throughput >= throughput_slo
      - GPU Frequency == freq_target based on 'direction' (above/below/all)
      - GPU Allocation == max_alloc

    Input:
        - model_name (str): Model's Name
        - latency_slo (float): Specifies Latency SLO
        - throughput_slo (float): Specifies Throughput SLO
        - freq_target (int): Current Frequency
        - alloc (int): GPU Allocation
        - direction (str): Direction of search
        - csv_path (str): Path to database file

    Output:
        - (list): List of feasible configurations
    """
    candidates = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by model
            if row["model"] != model_name:
                continue

            batch_size = int(row["batch_size"])
            frequency = int(row["frequency"])
            allocation = int(row["mig_slices"])
            latency = float(row["latency"])
            throughput = float(row["throughput"])
            power = float(row["dynamic_power_draw"])

            # Check Frequency, SLOs, Allocations
            if ((direction == 'above') and (frequency < freq_target)) or ((direction == 'below') and (frequency > freq_target)):
                continue
            if latency > latency_slo:
                continue
            if throughput < throughput_slo:
                continue
            if allocation != alloc:
                continue

            candidates.append({"model_name": model_name,
                    "latency_slo": latency_slo,
                    "throughput_slo": throughput_slo,
                    "batch_size": batch_size,
                    "gpu_frequency": frequency,
                    "gpu_allocation": allocation,
                    "latency": latency,
                    "throughput": throughput,
                    "power": power,})

    return candidates

def model_tuner(prev_value, new_value, trigger_event, model_state, cost_metric="power", csv_path=MODEL_DATABASE, bypass_dir=None):
    """
    Function to change model configuration based on trigger

    Input:
        - prev_value (float): Previous value before triggering event
        - new_value (float): New value after triggering event
        - trigger_desc (str): Event that triggers reconfiguration ['load', 'carbon_intensity']
        - model_state (dictionary): Model's configuration
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])
        - csv_path (str): Path to database file
    Ouput:
         - (list): List of feasible configurations after tunning
    """
    if ((trigger_event == 'load') and (prev_value < new_value)) or ((trigger_event == 'carbon_intensity') and (prev_value > new_value)):
        direction = 'above' # load increase OR CI decrese => frequency increase
    elif ((trigger_event == 'load') and (prev_value > new_value)) or ((trigger_event == 'carbon_intensity') and (prev_value < new_value)):
        direction = 'below' # load decrease OR CI increase => frequency decrease
    else:
        direction = 'all' # load OR CI unchanged => frequency non-specified

    if bypass_dir:
        direction = bypass_dir

    model_name = model_state['model_name']
    latency_slo = model_state['latency_slo']
    throughput_slo = new_value if trigger_event == 'load' else model_state['throughput_slo']
    gpu_frequency = model_state['gpu_frequency']
    gpu_allocation = model_state['gpu_allocation']

    return serch_config_tuning(model_name, latency_slo, throughput_slo, gpu_frequency, gpu_allocation, direction, csv_path)

def best_config_tuning(model_name, latency_slo, throughput_slo, freq_target, alloc, cost_metric="power", csv_path=MODEL_DATABASE):
    """
    Function to find the configurations that minimizes cost for a model, subject to:
      - given model_name
      - latency <= latency_slo_ms
      - throughput >= throughput_slo
      - GPU Frequency == freq_target
      - GPU Allocation == max_alloc

    Input:
        - model_name (str): Model's Name
        - latency_slo (float): Specifies Latency SLO
        - throughput_slo (float): Specifies Throughput SLO
        - freq_target (int): Current Frequency
        - alloc (int): GPU Allocation
        - cost_metric (str): Specified Cost Metric (power/energy/carbon)
        - csv_path (str): Path to database file

    Output:
        - (list): List of feasible configurations
    """
    best_config = None
    best_cost = float("inf")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by model
            if row["model"] != model_name:
                continue

            batch_size = int(row["batch_size"])
            frequency = int(row["frequency"])
            allocation = int(row["mig_slices"])
            latency = float(row["latency"])
            throughput = float(row["throughput"])
            power = float(row["dynamic_power_draw"])
            cost = float(cost_estimator(frequency, [power], [latency], cost_metric))

            # Check Frequency, SLOs, Allocations
            if frequency != freq_target:
                # print("frequency if statement")
                continue
            if latency > latency_slo:
                # print("latency slo if statement")
                continue
            if throughput < throughput_slo:
                # print("throughput if statement")
                continue
            if allocation != alloc:
                # print("allocation if statement")
                continue
            
            # Minimize cost
            if cost < best_cost:
                best_cost = cost
                best_config = {
                    "model_name": model_name,
                    "latency_slo": latency_slo,
                    "throughput_slo": throughput_slo,
                    "batch_size": batch_size,
                    "gpu_frequency": frequency,
                    "gpu_allocation": allocation,
                    "latency": latency,
                    "throughput": throughput,
                    "power": power,
                }

    return best_config

def local_tuner(current_values, trigger_values, trigger_desc, gpu_state, cost_metric="power", csv_path=MODEL_DATABASE):
    """
    Function to change gpu configuration based on trigger

    Input:
        - current_values (list): Exisitng value before reconfiguration (one per model) e.g.: [old_value_1, old_value_2, ...]
        - trigger_values (list): Value that trigger reconfiguration (one per model) e.g.: [new_value_1, new_value_2, ...]
        - trigger_desc (str): Event that triggers reconfiguration ['load', 'carbon_intensity']
        - gpu_state (dictionary): Content of GPU
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])
        - csv_path (str): Path to database file
    Ouput:
         - (dictionary): Dictionary of tuned deployment configs
    """
    # Step 1: Find frequency candidates based on triggering
    canditates_freq = []
    for i, bin in enumerate(gpu_state['configs']):
        if current_values[i] != trigger_values[i]: # if trigger value changes
            candidates_per_model_tmp = model_tuner(current_values[i], trigger_values[i], trigger_desc, bin, cost_metric, csv_path)
            if not candidates_per_model_tmp:
                print(f'Model {bin["model_name"]} cannot be tuned')
                return None
            canditates_freq.append(min([item["gpu_frequency"] for item in candidates_per_model_tmp]))
    

    # Step 2: Specify max(min freq across models)
    freq_cutoff = max(canditates_freq)

    # Step 3: For each model in bin prune search space (frequency > cut-off frequency) and get BS that minimizes cost (cost monotonically increases with BS)
    tuned_configs = []
    for i, model in enumerate(gpu_state['configs']):
        throughput_slo = trigger_values[i] if trigger_desc == 'load' else model['throughput_slo']
        tmp_config = best_config_tuning(model["model_name"], model["latency_slo"], throughput_slo, freq_cutoff, model["gpu_allocation"], cost_metric=cost_metric, csv_path=MODEL_DATABASE)
        if not tmp_config:
            break
        tuned_configs.append(tmp_config)

    merged_cost = None
    if tuned_configs:
        freq_working = tuned_configs[0]["gpu_frequency"]
        merged_powers = [c["power"] for c in tuned_configs]
        merged_lats = [c["latency"] for c in tuned_configs]
        merged_cost = cost_estimator(freq_working, merged_powers, merged_lats, cost_metric)

    tuned_bin = {"configs": tuned_configs,
                  "total_cost": merged_cost,
                  "total_alloc_size": sum(c["gpu_allocation"] for c in tuned_configs)}



    return tuned_bin


gpu_state = {'configs': [{'model_name': 'mobilenet_v3_large',
                          'latency_slo': 25.970196723937992,
                          'throughput_slo': 981.59,
                          'batch_size': 4,
                          'gpu_frequency': 1230,
                          'gpu_allocation': 1,
                          'latency': 4.04047966003418,
                          'throughput': 999.3660180328412,
                          'power': 16.095429253125133},
                        {'model_name': 'densenet169',
                          'latency_slo': 93.69852542877194,
                          'throughput_slo': 254.07,
                          'batch_size': 4,
                          'gpu_frequency': 1230,
                          'gpu_allocation': 2,
                          'latency': 12.794256210327148,
                          'throughput': 315.23921491222643,
                          'power': 25.545024240859775},
                        {'model_name': 'resnet50',
                          'latency_slo': 47.57630825042726,
                          'throughput_slo': 1391.97,
                          'batch_size': 16,
                          'gpu_frequency': 1230,
                          'gpu_allocation': 4,
                          'latency': 11.397123336791992,
                          'throughput': 1416.976201728885,
                          'power': 81.48249965182572}],
             'total_cost': 2053.41,
             'total_alloc_size': 7}

tuned_bin = local_tuner([981.59, 254.07, 1391.97], [600, 189, 830], 'load', gpu_state, cost_metric='energy')
# tuned_bin = local_tuner([380, 380, 380], [50, 50, 50], 'CI', gpu_state, cost_metric='carbon')
print(tuned_bin)