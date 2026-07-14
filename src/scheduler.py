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
CI = 490
LT = 4 * 352 * 24 * 60 * 60
C_M = 1/0.9 * (583 * 1.52 + 350 + 500) * 8.26 + 66.60 * 80 + 2 * 150

BASE_DIR = f"/home/ps35324/sus-gpus/motivation/batch-size/aggregated-power-data"
MODEL_DATABASE = f'{BASE_DIR}/model_datastore.csv'


def brute_force_scheduler(models_slos: dict, model_profiles_dir: str, objective: str):

    # 1. For each model, prune profiles that do not meet SLOs or throughput requirements
    pruned_profiles = {}
    for model, slos in models_slos.items():
        df_profile = pd.read_csv(f"{model_profiles_dir}/{model}_profiles.csv")
        latency_slo = slos[0]
        tput_req = slos[1]
        df_filtered = df_profile[~((df_profile["latency"] > latency_slo) | (df_profile["throughput"] > tput_req))]
        pruned_profiles[model] = df_filtered
    
    # 2. Generate cartesian product of all possible GPU placement configurations
    


    # 3. Filter out invalid configurations that exceed GPU resources

    # 4. Populate cartesian product table with power draw estimations and energy consumption

    # 5. Select optimal configurations based on objective (minimize power draw, energy, operational carbon, embodied carbon, total carbon)


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
    #TODO: Update with power estimator function
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


def config_selector(model_name, latency_slo, throughput_slo, cost_metric="power", csv_path=MODEL_DATABASE):
    """
    Function to find the configuration with minimum cost that satisfies:
        - given model_name
        - latency <= latency_slo
        - throughput >= throughput_slo

    Input:
        - model_name (str): Model's Name
        - latency_slo (float): Specifies Latency SLO
        - throughput_slo (float): Specifies Throughput SLO
        - cost_metric (str): Specified Cost Metric (power/energy/carbon)
        - csv_path (str): Path to database file

    Output:
        - (dictionary): Configuration that minimizes cost
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

            # Check SLOs
            if latency > latency_slo:
                continue
            if throughput < throughput_slo:
                continue
            # ########## LIMIT ALLOCATION SIZE IN STEP 0 ##########
            if allocation == 7:
                continue

            # Minimize power
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


def get_freq_bins(csv_path=MODEL_DATABASE):
    """
    Function to get frequency bins

    Input:
        - csv_path (str): Path to database file

    Output:
        - (list): Frequency bins
    """
    # #TODO: Update for other metrics
    # if cost_metric == 'power':
    #     # return [(210, 420), (510, 600), (690, 870), (960, 1050), (1140,1140), (1230, 1230), (1320, 1320), (1410, 1410)] # Interpret bins as: low <= freq <= high
    #     return [210, 240, 330, 420, 510, 600, 690, 780, 870, 960, 1050, 1140, 1230, 1320, 1410]
    freqs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            freqs.append(int(row["frequency"]))
    unique_freqs = np.unique(freqs)
    return unique_freqs.tolist()


def model_packer(model_list, cost_metric="power"):
    """
    Function to pack model within a specified bin

    Input:
        - model_list (list): List of model within a bin
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])

    Output:
        - (list): List with buckets of packed models
    """
    # I - Greedy Pack/Spread based on total cost
    packed_result = [] # list to append packed configs, each list is one GPU

    # I.a - Only one config at this frequency --> Append it to list
    if len(model_list) == 1:
        cfg = model_list[0]
        freq = cfg["gpu_frequency"]
        power_list = [cfg["power"]]
        latency_list = [cfg["latency"]]
        total_cost = cost_estimator(freq, power_list, latency_list, cost_metric)
        packed_result.append({"configs":[cfg], "total_cost": total_cost, "total_alloc_size":cfg["gpu_allocation"]})

    # I.b - Multiple configs --> greedy pack/spread
    else:
        for cfg in model_list:
            freq = cfg["gpu_frequency"]
            power = cfg["power"]
            latency = cfg["latency"]
            alloc = cfg["gpu_allocation"]

            # Initialization Step
            if not packed_result:
                packed_result.append({"configs":[cfg], "total_cost": cost_estimator(freq, [power], [latency], cost_metric), "total_alloc_size":alloc})
                continue

            # If this model doesn't fit on any bin (capacity 7) -> create a new bin else check if it fits and deploy it
            flag_fitted = False
            for bin in packed_result: # Check every bin
                if bin["total_alloc_size"] + alloc <= 7: # Possible Candindate -> Check if feasible

                    # Try packing vs spreading for this model onto the current GPU
                    current_powers = [m["power"] for m in bin["configs"]]
                    current_lats = [m["latency"] for m in bin["configs"]]

                    # Cost calclulation
                    packed_cost = cost_estimator(freq, current_powers+[power], current_lats+[latency], cost_metric) # Cost if we pack this model with the current set
                    separate_cost = bin["total_cost"] + cost_estimator(freq, [power], [latency], cost_metric) # Cost if we give this model its own GPU

                    # If combining is cheaper → pack it; else continue checking
                    if packed_cost < separate_cost:
                        bin["configs"].append(cfg)
                        bin["total_cost"] = packed_cost
                        bin["total_alloc_size"] += alloc
                        flag_fitted = True
                        break

            if flag_fitted == False: # Bin not found -> create a new bin
                packed_result.append({"configs":[cfg], "total_cost": cost_estimator(freq, [power], [latency], cost_metric), "total_alloc_size":alloc})
                continue

    return packed_result


def transform_frequencies(freqs, cost_metric='power'):
    """
    Function to transform frequencies

    Input:
        - freqs (list): List of unique frequencies present in the experiment
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])

    Ouput:
        - (list): Transformed list
    """
    #TODO: Update for other cost metrics
    if cost_metric == 'power':
        return np.sort(freqs)[::-1]
    elif cost_metric == 'energy':
        return np.sort(freqs)[::-1]
    elif cost_metric == 'carbon':
        return np.sort(freqs)[::-1]


def get_config_candidates_for_freq(model_name, latency_slo, throughput_slo, freq_target, max_alloc, csv_path=MODEL_DATABASE):
    """
    Function to find all feasible configurations for a model at a specific frequency, subject to:
      - given model_name
      - latency <= latency_slo_ms
      - throughput >= throughput_slo
      - GPU Frequency == freq_target
      - GPU Allocation <= max_alloc

    Input:
        - model_name (str): Model's Name
        - latency_slo (float): Specifies Latency SLO
        - throughput_slo (float): Specifies Throughput SLO
        - freq_target (int): Selected Frequency
        - max_alloc (int): Maximum GPU Allocation
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
            if frequency != freq_target:
                continue
            if latency > latency_slo:
                continue
            if throughput < throughput_slo:
                continue
            if allocation > max_alloc:
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


def repack_configs(bin_1, bin_2, cost_metric="power", csv_path=MODEL_DATABASE):
    """
    Function to test whether packing the models within the bins is feasible

    Input:
        - bin_1 (dictionary): Current bin (can be reconfigured)
        - bin_2 (dictionary): Working bin (cannot be reconfigured)
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])
        - csv_path (str): Path to database file

    Output:
        - (bool, list/None): Flag that specifies whether packing is possible, Configurations of merged bins
    """
    # --- Working bin info ---
    freq_working = bin_2["configs"][0]["gpu_frequency"] # frequency of 'working' bin
    alloc_working = bin_2["total_alloc_size"] # capacity of 'working' bin
    powers_working = [item["power"] for item in bin_2["configs"]]
    lats_working = [item["latency"] for item in bin_2["configs"]]
    cost_working = bin_2["total_cost"]

    # --- Current bin info ---
    cost_current = bin_1["total_cost"]

    # If there's no remaining capacity at all, can't repack
    remaining_capacity_global = 7 - alloc_working
    if remaining_capacity_global <= 0:
        return False, None

    # --- Build candidate list per model from bin_1 ---
    candidates_per_model = []
    for cfg in bin_1["configs"]:
        model_name = cfg["model_name"]
        latency_slo = cfg["latency_slo"]
        throughput_slo = cfg["throughput_slo"]

        model_candidates = get_config_candidates_for_freq(model_name=model_name,
                                                        latency_slo=latency_slo,
                                                        throughput_slo=throughput_slo,
                                                        freq_target=freq_working,
                                                        max_alloc=remaining_capacity_global,
                                                        csv_path=csv_path,)

        if not model_candidates:
            # If even ONE model cannot be placed on this GPU, full repack fails
            return False, None

        candidates_per_model.append(model_candidates)

    # --- Enumerate all combinations ---
    best_merged_cost = float("inf")
    best_combination = None
    best_total_alloc_new = None

    for combo in itertools.product(*candidates_per_model):
        alloc_new = sum(c["gpu_allocation"] for c in combo)
        if alloc_new > remaining_capacity_global:
            continue  # exceeds GPU capacity

        merged_configs = bin_2["configs"] + list(combo)
        merged_powers = [c["power"] for c in merged_configs]
        merged_lats = [c["latency"] for c in merged_configs]

        merged_cost = cost_estimator(freq_working, merged_powers, merged_lats, cost_metric)

        if merged_cost < best_merged_cost:
            best_merged_cost = merged_cost
            best_combination = combo
            best_total_alloc_new = alloc_new

    # If we didn’t find any capacity-respecting combination, no repack
    if best_combination is None:
        return False, None

    # Compare best merged cost with separate cost
    separate_cost = cost_working + cost_current

    if best_merged_cost < separate_cost:
        merged_bin = {"configs": bin_2["configs"] + list(best_combination),
                      "total_cost": best_merged_cost,
                      "total_alloc_size": alloc_working + best_total_alloc_new,}
        return True, merged_bin
    else:
        return False, None


def merge_configs(bin_1, bin_2, moving_freq=0, cost_metric="power", csv_path=MODEL_DATABASE):
    """
    Function to test whether merging the models within the bins is feasible

    Input:
        - bin_1 (dictionary): Current bin
        - bin_2 (dictionary): Working bin (sets frequency)
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])
        - csv_path (str): Path to database file

    Output:
        - (bool, list/None): Flag that specifies whether packing is possible, Configurations of merged bins
    """
    # --- Working bin info ---
    freq_working = bin_2["configs"][0]["gpu_frequency"] # frequency of 'working' bin

    # --- Update frequency as specified ---
    if moving_freq < freq_working:
        return False, None
    else:
        freq_working = moving_freq

    # --- Build candidate list per model from bin_1 ---
    candidates_per_model = []
    for cfg in bin_1["configs"]+bin_2["configs"]:
        model_name = cfg["model_name"]
        latency_slo = cfg["latency_slo"]
        throughput_slo = cfg["throughput_slo"]

        model_candidates = get_config_candidates_for_freq(model_name=model_name,
                                                        latency_slo=latency_slo,
                                                        throughput_slo=throughput_slo,
                                                        freq_target=freq_working,
                                                        max_alloc=4, # We cannot use 7 so we pick the next lower feasible
                                                        csv_path=csv_path,)

        if not model_candidates:
            # If even ONE model cannot be placed on this GPU, merge fails
            return False, None

        candidates_per_model.append(model_candidates)

    # --- Enumerate all combinations ---
    best_merged_cost = float("inf")
    best_combination = None
    best_total_alloc_new = None

    for combo in itertools.product(*candidates_per_model):
        alloc_new = sum(c["gpu_allocation"] for c in combo)
        if alloc_new > 7:
            continue  # exceeds GPU capacity

        merged_configs = list(combo)
        merged_powers = [c["power"] for c in merged_configs]
        merged_lats = [c["latency"] for c in merged_configs]

        merged_cost = cost_estimator(freq_working, merged_powers, merged_lats, cost_metric)

        if merged_cost < best_merged_cost:
            best_merged_cost = merged_cost
            best_combination = merged_configs
            best_total_alloc_new = alloc_new

    # If we didn’t find any capacity-respecting combination, no repack
    if best_combination is None:
        return False, None
    else:
        merged_bin = {"configs": best_combination,
                      "total_cost": best_merged_cost,
                      "total_alloc_size": best_total_alloc_new,}
        return True, merged_bin


def merge_bins(sorted_bins_OG, cost_metric="power", csv_path=MODEL_DATABASE):
    """
    Function to test whether merging exatly 2 bins form the list is feasible

    Input:
        - sorted_bins_OG (list): Current list with bin
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])
        - csv_path (str): Path to database file

    Output:
        - (bool, list/None): Flag that specifies whether merging is possible, Update sorted list with bins
    """
    sorted_bins = copy.deepcopy(sorted_bins_OG)
    merged = False
    ascending_freqs = np.sort(transform_frequencies(get_freq_bins(csv_path), cost_metric))[::+1]
    print(f'!!!!TRANSFORMED FREQUENCIES: {ascending_freqs}!!!!')
    for moving_freq in ascending_freqs:
      print('--- Trying to merge bins at frequency:', moving_freq, '---')
      i = len(sorted_bins)-1
      while i > 0:
          j = i-1
          while j >= 0:
              merge_flag, merged_bin = merge_configs(sorted_bins[i], sorted_bins[j], moving_freq, cost_metric, csv_path)# Merge bins
              if merge_flag:
                  # remove merged bins
                  sorted_bins.pop(i)
                  sorted_bins.pop(j)
                  sorted_bins.insert(j, merged_bin)
                  merged = True
                  return merged, sorted_bins
              j -= 1
          if not merged:
              i -= 1
    return merged, sorted_bins


def cost_swig(frequency, power, latency, cost_metric='power'):
    """
    Function to calclulate swing cost based on cost metric

    Input:
        - frequency (int): Slected Frequency
        - power (float): Power draw of model
        - latency (float): Latencies of model
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])

    Output:
        - (float): Cost Estimation
    """
    #TODO: Update with power estimator function
    if cost_metric == 'power':
        return power
    elif cost_metric == 'energy':
        return power
    elif cost_metric == 'carbon':
        return power


def swing_sorting(sorted_bins, cost_metric='power', csv_path=MODEL_DATABASE):
    print(sorted_bins)
    upper_freq = sorted_bins[0]['configs'][0]['gpu_frequency']
    lower_freq = sorted_bins[-1]['configs'][0]['gpu_frequency']
    # print(f"Upper freq: {upper_freq}, Lower freq: {lower_freq}")
    swing_costs = [0]*len(sorted_bins)
    for i in range(len(sorted_bins)-1, 0, -1):
        bin_current = sorted_bins[i]
        # print(bin_current)
        for model_ in bin_current['configs']:
            upper_search_space = get_config_candidates_for_freq(model_name=model_['model_name'],
                                                                latency_slo=model_['latency_slo'],
                                                                throughput_slo=model_['throughput_slo'],
                                                                freq_target=upper_freq,
                                                                max_alloc=4, # We cannot use 7 so we pick the next lower feasible
                                                                csv_path=csv_path,)
            upper_cost = max([cost_swig(upper_freq,config['power'],config['latency'],cost_metric) for config in upper_search_space])

            # lower_search_space = get_config_candidates_for_freq(model_name=model_['model_name'],
            #                                                     latency_slo=model_['latency_slo'],
            #                                                     throughput_slo=model_['throughput_slo'],
            #                                                     freq_target=model_['gpu_frequency'],
            #                                                     max_alloc=4, # We cannot use 7 so we pick the next lower feasible
            #                                                     csv_path=csv_path,)
            # lower_cost = min([cost_swig(model_['gpu_frequency'],config['power'],config['latency'],cost_metric) for config in lower_search_space])

            new_lower_freq = lower_freq
            if (model_['model_name'] == 'diffusion_256_256') or (model_['model_name'] == 'diffusion_1024_1024') or (model_['model_name'] == 'gpt'):
                new_lower_freq = 690
            lower_search_space = get_config_candidates_for_freq(model_name=model_['model_name'],
                                                                latency_slo=100000000,
                                                                throughput_slo=0,
                                                                freq_target=new_lower_freq,
                                                                max_alloc=4, # We cannot use 7 so we pick the next lower feasible
                                                                csv_path=csv_path,)
            
            lower_cost = min([cost_swig(new_lower_freq,config['power'],config['latency'],cost_metric) for config in lower_search_space])
            if model_['model_name'] == 'diffusion_256_256' or model_['model_name'] == 'diffusion_1024_1024' or model_['model_name'] == 'gpt':
                lower_cost -= 20
            print(f'Model: {model_["model_name"]}, Upper cost: {upper_cost}, Lower cost: {lower_cost}')
            swing_costs[i] += upper_cost - lower_cost
        # print(swing_costs[i])
    # print(swing_costs)
    return [x for x, _ in sorted(zip(sorted_bins[1:], swing_costs[1:]), key=lambda pair: pair[1], reverse=True)]


def pretty_print_bins(bins):
    if len(bins) == 0:
        print(f"[]")
    for i, bin_item in enumerate(bins):
        print(f"\n=== BIN {i} ===")

        configs = bin_item["configs"]
        bin_freq = configs[0]["gpu_frequency"] if configs else "N/A"

        print(f"Freq={bin_freq} MHz | TotalCost={bin_item['total_cost']:.2f} | Alloc={bin_item['total_alloc_size']}")
        print("Configs:")

        for cfg in configs:
            print(
                f"  {cfg['model_name']} "
                f"(bs={cfg['batch_size']}, alloc={cfg['gpu_allocation']}, "
                f"lat={cfg['latency']:.1f}, thr={cfg['throughput']:.4f}, "
                f"pwr={cfg['power']:.1f})"
            )

    print()


def deployer(input_list, cost_metric='power', NUMBER_OF_GPUS=None, csv_path=MODEL_DATABASE):
    """
    Function to find deployment configurations for a given model list

    Input:
        - input_list (list): List of model to deploy with specified SLOs (e.g. [(model_1, latency_slo_1, throughput_slo_1), (model_2, latency_slo_2, throughput_slo_2), ...])
        - cost_metric (str): Slected cost metric (['power', 'energy', 'carbon'])
        - csv_path (str): Path to database file

    Output:
        - (list): List of deployment configs
    """

    # Step 0 --> For each of the given models find the DC that minimizes cost
    config_cand = []
    for (model,latency_slo,throughput_slo) in input_list:
        config_cand.append(config_selector(model, latency_slo, throughput_slo, cost_metric, csv_path))
    print('STEP 0')
    print(config_cand)
    print(f'#MODELS: {len(config_cand)}\n')

    # Step 1 --> Create bins based on frequency
    bin_keys = get_freq_bins(csv_path)
    freq_bins = {k: [] for k in bin_keys}
    for config in config_cand:
        freq = config['gpu_frequency']
        # for (low, high) in bin_keys:
        for freq_key in bin_keys:
            # if low <= freq <= high:
            if freq == freq_key:
                # freq_bins[(low, high)].append(config)
                freq_bins[freq_key].append(config)
                break
    print('STEP 1')
    print(freq_bins)
    print(f'#MODELS PER BIN: {[len(ff_bin) for ff_bin in freq_bins.values()]}\n')

    # Step 2 --> Find Configuration within each bin
    model_bins = []
    for bin_key in freq_bins:
        bin = freq_bins[bin_key]
        if len(bin) !=0 :
            # print(bin)
            model_bins = model_bins + model_packer(bin, cost_metric)
    print('STEP 2')
    print(model_bins)
    print(f'#MODELS:{sum([len(m["configs"]) for m in model_bins])}\n')
    # return(model_bins)

    # Step 3 --> Sort based on transformed frequency
    # Transform existing frequencies based on cost metric
    freqs = np.unique([item["configs"][0]["gpu_frequency"] for item in model_bins])
    freqs_order = transform_frequencies(freqs, cost_metric)

    # Sort bins based on transformed frequencies
    order_index = {freq: i for i, freq in enumerate(freqs_order)}
    sorted_bins_OG = sorted(model_bins, key=lambda x: order_index.get(x["configs"][0]["gpu_frequency"], float("inf")))

    flag_init = True
    all_bins = []
    NUMBER_OF_GPUS = float("inf") if NUMBER_OF_GPUS==None else NUMBER_OF_GPUS

    while ((len(all_bins) > NUMBER_OF_GPUS) or (flag_init == True)):

        if flag_init == True:
            sorted_bins = copy.deepcopy(sorted_bins_OG)
        else:
            flag_merge_pass, sorted_bins_OG = merge_bins(sorted_bins_OG, cost_metric, csv_path)
            if flag_merge_pass == False:
              print("SLOs and resource constraints cannot be met. Returning configuration from previous iteration")
              return all_bins
            sorted_bins = copy.deepcopy(sorted_bins_OG)

        flag_init = False #Initialiazation flag disabled
        print('STEP 3')
        # print(sorted_bins)
        pretty_print_bins(sorted_bins)
        print(f'#BINS: {len(sorted_bins)}', f'#MODELS: {sum([len(m["configs"]) for m in sorted_bins])}', '\n')

        # Step 4 --> Resolve fragmentation traverse ranked list in 2-directions (top->down, then bottom-->up) while reconfiguring and checking for SLOs
        final_bins = []

        # 4.1 --> Remove bins that have the max GPU allocation
        for bin in sorted_bins:
            if bin["total_alloc_size"] == 7:
                final_bins.append(bin)
                sorted_bins.remove(bin)
        print('STEP 4.1')
        # print('FINAL BINS:', final_bins)
        print(f'FINAL_BINS:')
        pretty_print_bins(final_bins)
        print(f'#BINS: {len(final_bins)},', f'#MODELS: {sum([len(m["configs"]) for m in final_bins])}')
        # print('SORTED BINS:', sorted_bins)
        print(f'SORTED_BINS:')
        pretty_print_bins(sorted_bins)
        print(f'#BINS: {len(sorted_bins)},', f'#MODELS: {sum([len(m["configs"]) for m in sorted_bins])}', '\n')

        # # 4.3 --> Traverse bottom-->up, Update & Remove bins that have the max GPU allocation
        # # Sort bins based on swing
        # sorted_bins_by_swing = swing_sorting(sorted_bins)
        # # print('SWING SORTED BINS:')
        # # pretty_print_bins(sorted_bins_by_swing)
        # i_2 = len(sorted_bins_by_swing)-1
        # while i_2 >= 0 and len(sorted_bins) > 0:
        #     bin_1b = sorted_bins_by_swing[i_2]  # 'current' bin (starting from the bottom of swing bins)
        #     merged_b = False
        #     print(bin_1b)

        #     j_2 = 0
        #     j_limit = sorted_bins.index(bin_1b)
        #     while j_2 < j_limit: # Scan from the top up to just above bin_1b
        #         bin_2b = sorted_bins[j_2]  # 'working' bin (from the top)
        #         repack_flag_b, merged_bin_b = repack_configs(bin_1b, bin_2b, cost_metric, csv_path)

        #         if repack_flag_b:
        #             # remove bin_i and bin_j (remove i first (higher index), then j), replace i with merged_bin
        #             sorted_bins.pop(j_limit)   # remove original bin_1
        #             sorted_bins.pop(j_2)   # remove original bin_2
        #             # now insert merged_bin at position j
        #             sorted_bins.insert(j_2, merged_bin_b)

        #             # If merged_bin is full, move it immediately to final_bins
        #             if merged_bin_b["total_alloc_size"] == 7:
        #                 final_bins.append(merged_bin_b)
        #                 sorted_bins.pop(j_2)  # remove merged from working set

        #             merged_b = True
        #             sorted_bins_by_swing = swing_sorting(sorted_bins) # Update swing bins
        #             i_2 = len(sorted_bins_by_swing)-1 # Start again
        #             break  # stop scanning j for this i

        #         j_2 += 1 # Go to the next working bin


        #     if not merged_b: # nothing more we can do with this bin_i, move to next i
        #         i_2 -=1

        #     print('-------------')
        #     print('| Botom-->Up |')
        #     print('-------------')
        #     # print('FINAL BINS:', final_bins)
        #     # print('SORTED BINS:', sorted_bins)
        #     print(f'FINAL_BINS:')
        #     pretty_print_bins(final_bins)
        #     print(f'SORTED_BINS:')
        #     pretty_print_bins(sorted_bins)
        #     print(f'#FINAL BINS: {len(final_bins)}, #WORKING BINS: {len(sorted_bins)}')
        #     print(f'#FINAL_MODELS: {sum([len(m["configs"]) for m in final_bins])}, #WORKING_MODELS: {sum([len(m["configs"]) for m in sorted_bins])}')
        #     print('\n')

        # 4.2 --> Traverse top->down, Update & Remove bins that have the max GPU allocation
        i_1 = 0
        while i_1 < len(sorted_bins):
            bin_1a = sorted_bins[i_1] # Pick 'current' bin

            merged_a = False
            j_1 = len(sorted_bins)-1
            while j_1 > i_1: # Scan from the bottom up to just below i
                bin_2a = sorted_bins[j_1] # Pick 'working' bin
                repack_flag_a, merged_bin_a = repack_configs(bin_1a, bin_2a, cost_metric, csv_path)

                if repack_flag_a:
                    # remove bin_i and bin_j (remove j first (higher index), then i), replace j with merged_bin
                    sorted_bins.pop(j_1)   # remove original bin_2
                    sorted_bins.pop(i_1)   # remove original bin_1
                    # now insert merged_bin at position j-1
                    sorted_bins.insert(j_1-1, merged_bin_a)

                    # If merged_bin is full, move it immediately to final_bins
                    if merged_bin_a["total_alloc_size"] == 7:
                        final_bins.append(merged_bin_a)
                        sorted_bins.pop(j_1-1)  # remove merged from working set

                    merged_a = True
                    break  # stop scanning j for this i

                j_1 -= 1 # Go to the next working bin

            if not merged_a: # nothing more we can do with this bin_i, move to next i
                i_1 +=1

            
            print('-------------')
            print('| Top-->Down |')
            print('-------------')
            # print('FINAL BINS:', final_bins)
            # print('SORTED BINS:', sorted_bins)
            print(f'FINAL_BINS:')
            pretty_print_bins(final_bins)
            print(f'SORTED_BINS:')
            pretty_print_bins(sorted_bins)
            print(f'#FINAL BINS: {len(final_bins)}, #WORKING BINS: {len(sorted_bins)}')
            print(f'#FINAL_MODELS: {sum([len(m["configs"]) for m in final_bins])}, #WORKING_MODELS: {sum([len(m["configs"]) for m in sorted_bins])}')
            print('\n')


        # 4.3 --> Traverse bottom-->up, Update & Remove bins that have the max GPU allocation
        # Sort bins based on swing
        sorted_bins_by_swing = swing_sorting(sorted_bins)
        # print('SWING SORTED BINS:')
        # pretty_print_bins(sorted_bins_by_swing)
        i_2 = len(sorted_bins_by_swing)-1
        while i_2 >= 0 and len(sorted_bins) > 0:
            bin_1b = sorted_bins_by_swing[i_2]  # 'current' bin (starting from the bottom of swing bins)
            merged_b = False
            print(bin_1b)

            j_2 = 0
            j_limit = sorted_bins.index(bin_1b)
            while j_2 < j_limit: # Scan from the top up to just above bin_1b
                bin_2b = sorted_bins[j_2]  # 'working' bin (from the top)
                repack_flag_b, merged_bin_b = repack_configs(bin_1b, bin_2b, cost_metric, csv_path)

                if repack_flag_b:
                    # remove bin_i and bin_j (remove i first (higher index), then j), replace i with merged_bin
                    sorted_bins.pop(j_limit)   # remove original bin_1
                    sorted_bins.pop(j_2)   # remove original bin_2
                    # now insert merged_bin at position j
                    sorted_bins.insert(j_2, merged_bin_b)

                    # If merged_bin is full, move it immediately to final_bins
                    if merged_bin_b["total_alloc_size"] == 7:
                        final_bins.append(merged_bin_b)
                        sorted_bins.pop(j_2)  # remove merged from working set

                    merged_b = True
                    sorted_bins_by_swing = swing_sorting(sorted_bins) # Update swing bins
                    i_2 = len(sorted_bins_by_swing)-1 # Start again
                    break  # stop scanning j for this i

                j_2 += 1 # Go to the next working bin


            if not merged_b: # nothing more we can do with this bin_i, move to next i
                i_2 -=1

            print('-------------')
            print('| Botom-->Up |')
            print('-------------')
            # print('FINAL BINS:', final_bins)
            # print('SORTED BINS:', sorted_bins)
            print(f'FINAL_BINS:')
            pretty_print_bins(final_bins)
            print(f'SORTED_BINS:')
            pretty_print_bins(sorted_bins)
            print(f'#FINAL BINS: {len(final_bins)}, #WORKING BINS: {len(sorted_bins)}')
            print(f'#FINAL_MODELS: {sum([len(m["configs"]) for m in final_bins])}, #WORKING_MODELS: {sum([len(m["configs"]) for m in sorted_bins])}')
            print('\n')

        all_bins = final_bins + sorted_bins
        print('Step 4 - FIN')
        # print('ALL BINS:', all_bins)
        print(f'ALL_BINS:')
        pretty_print_bins(all_bins)
        print(f'#ALL BINS: {len(all_bins)}')
        print(f'#ALL MODELS: {sum([len(m["configs"]) for m in all_bins])}')
        print('\n')

    return all_bins

BASE_INPUTS = [
    ('resnet50', 47.57630825042726, 1391.97),
    ('mobilenet_v3_large',25.970196723937992, 981.59),
    # ('mobilenet_v3_large',25.970196723937992, 1966.634811879258),
    ('vgg11', 12.486314773559576, 744.65),
    ('resnet152', 128.05020809173587, 278.36),
    ('densenet169', 93.69852542877194, 254.07),
    ('densenet161', 97.31435775756836, 234.19),
    # ('efficientnet_b5', 227.14972496032715, 207.52),
    # ('inception_v3', 40.23642539978028, 189.25),
    ('vgg19', 17.524337768554688, 169.54),
    ('diffusion_1024_1024', 25358.9243824928, 0.0923146137935218),
    ('diffusion_256_256', 8284.928342849231, 0.41497842657583255),
    ('whisper', 5509.2905923409234, 3.29843852958),
    ('gpt', 18385.29834294, 0.16192860916956147),
]

INPUTS_LOADS_DICT = {}


# 150% Load
INPUTS_150 = []
for base_model in BASE_INPUTS:
    model = list(base_model)
    if model[0] != 'resnet50' and model[0] != 'diffusion_1024_1024' and model[0] != 'gpt' and model[0] != 'diffusion_256_256':
        model[2] = model[2] * 1.5
    model_tuple = tuple(model)
    INPUTS_150.append(model_tuple)
INPUTS_LOADS_DICT['150'] = INPUTS_150

# 125% Load
INPUTS_125 = []
for base_model in BASE_INPUTS:
    model = list(base_model)
    if model[0] != 'resnet50' and model[0] != 'diffusion_1024_1024' and model[0] != 'gpt' and model[0] != 'diffusion_256_256':
        model[2] = model[2] * 1.25
    model_tuple = tuple(model)
    INPUTS_125.append(model_tuple)
INPUTS_LOADS_DICT['125'] = INPUTS_125


# 100% Load
INPUTS_100 = []
for base_model in BASE_INPUTS:
    model = list(base_model)
    model[2] = model[2] * 1.00
    model_tuple = tuple(model)
    INPUTS_100.append(model_tuple)
INPUTS_LOADS_DICT['100'] = INPUTS_100

# 75% Load
INPUTS_75 = []
for base_model in BASE_INPUTS:
    model = list(base_model)
    model[2] = model[2] * 0.75
    model_tuple = tuple(model)
    INPUTS_75.append(model_tuple)
INPUTS_LOADS_DICT['75'] = INPUTS_75

# 50% Load
INPUTS_50 = []
for base_model in BASE_INPUTS:
    model = list(base_model)
    model[2] = model[2] * 0.50
    model_tuple = tuple(model)
    INPUTS_50.append(model_tuple)
INPUTS_LOADS_DICT['50'] = INPUTS_50

# 25% Load
INPUTS_25 = []
for base_model in BASE_INPUTS:
    model = list(base_model)
    model[2] = model[2] * 0.25
    model_tuple = tuple(model)
    INPUTS_25.append(model_tuple)
INPUTS_LOADS_DICT['25'] = INPUTS_25

LOAD = 50
INPUT = INPUTS_LOADS_DICT[str(LOAD)]
print(INPUT)


a = deployer(INPUT, cost_metric='energy', NUMBER_OF_GPUS=5, csv_path=MODEL_DATABASE)
print(f'--------------')
print(f'| Final Bins |')
print(f'--------------')
pretty_print_bins(a)
print(sum([elem['total_cost'] for elem in a]))