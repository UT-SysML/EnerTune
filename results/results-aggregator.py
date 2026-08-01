import pandas as pd
import glob
import os
import numpy as np
import re
import math

SYSTEMS=['usher', 'parva', 'fgd', 'gpulets', 'et-power', 'et-energy', 'et-carbon']
MODEL_SLOS = {
    'resnet50': [47.57630825042726, 1391.97],
    'mobilenet_v3_large': [25.970196723937992, 981.59],
    'densenet169': [93.69852542877194, 254.07],
    'gpt': [18385.29834294, 0.16192860916956147],
    'diffusion_256_256': [8284.928342849231, 0.41497842657583255],
    'vgg19': [17.524337768554688, 169.54],
    'vgg11': [12.486314773559576, 744.65],
    'diffusion_1024_1024': [25358.9243824928, 0.0923146137935218],
    'densenet161': [97.31435775756836, 234.19],
    'resnet152': [128.05020809173587, 278.36],
    'whisper': [5509.2905923409234, 3.29843852958],
    'efficientnet_b5': [227.14972496032715, 207.52],
    'inception_v3': [40.23642539978028, 189.25],
}
PUE = 1.1
CI = 380
C_M = 1/0.9 * (583 * 1.52 + 350 + 500) * 8.26 + 66.60 * 80 + 2 * 150
LT = 4 * 352 * 24 * 60 * 60

for system in SYSTEMS:
    rows = []
    for d in glob.glob(f"{system}-results/"):
        if os.path.isdir(d):
            print(f"Processing {system} results in path {d}")
            for load_path in glob.glob(f"{d}/*/"):
                print(load_path)
                curr_load = int(load_path.split('/')[-2].split('-')[-1])
                print(curr_load)
                if os.path.isdir(load_path):
                    for sd in glob.glob(f"{load_path}/*/"):
                        if os.path.isdir(sd):
                            for f in glob.glob(f"{sd}/*/"):
                                if os.path.isdir(f):
                                    job_mix_info = f.split('/')[-2]
                                    jobs = re.split(r"-(?:point|poisson)_", job_mix_info)
                                    models = []
                                    bss = []
                                    for job in jobs:
                                        models.append(job.split('-')[0])
                                        bss.append(int(job.split('-')[1]))
                                    distribution = job_mix_info.split('-')[-1]

                                    df_tput = pd.read_csv(f"{f}/tput.csv")
                                    df_lat = pd.read_csv(f"{f}/total_p99.csv")

                                    row = []
                                    start_time = -1
                                    end_time = -1
                                    for (i, tput_row) in df_tput.iterrows():
                                        load = math.ceil((tput_row['rps'] * bss[tput_row['job_no']]) / MODEL_SLOS[models[tput_row['job_no']]][1] * 100)
                                        if load >= 20 and load <= 30:
                                            load = 25
                                        elif load >= 45 and load <= 55:
                                            load = 50
                                        elif load >= 70 and load <= 80:
                                            load = 75
                                        elif load >= 95 and load <= 105:
                                            load = 100
                                        elif load >= 120 and load <= 130:
                                            load = 125
                                        elif load >= 145 and load <= 155:
                                            load = 150
                                        
                                        freq = tput_row['max_freqs']
                                        tput = tput_row['metric_value']
                                        batch_size = int(tput_row['batch_size'])
                                        allocation_size = tput_row['mig_slices'] / 7 * 100
                                        if tput_row['mode'] == 'mps-manual-cap':
                                            allocation_size = tput_row['mig_slices']
                                        latency = df_lat.iloc[i]['metric_value']
                                        curr_start_time = tput_row['start_time']
                                        curr_end_time = tput_row['end_time']


                                        if tput_row['job_no'] == 0:
                                            start_time = -1
                                            end_time = -1

                                        if start_time == -1 or curr_start_time < start_time:
                                            start_time = curr_start_time
                                        if curr_end_time > end_time:
                                            end_time = curr_end_time

                                        avg_pwr = 0
                                        max_pwr = 0
                                        total_energy = 0
                                        operational_carbon = 0
                                        embodied_carbon = 0
                                        if tput_row['job_no'] == (len(models)-1):
                                            for pwr_csv in glob.glob(f"{f}/energy-*-{freq}.csv"):
                                                df_pwr = pd.read_csv(pwr_csv)
                                                df_pwr['timestamp'] = df_pwr['timestamp'] / 1e9
                                                df_pwr = df_pwr[(df_pwr['timestamp'] >= start_time) & (df_pwr['timestamp'] <= end_time)]
                                                avg_pwr = df_pwr['power_draw_mW'].mean() / 1000
                                                max_pwr = df_pwr['power_draw_mW'].max() / 1000
                                                energy_list = df_pwr['energy_draw_j'].tolist()
                                                total_energy = energy_list[-1] - energy_list[0]
                                                operational_carbon = total_energy * PUE * CI / (3.6 * 10e6)
                                                embodied_carbon = C_M * (end_time - start_time) / LT * 1000
                                            start_time = -1
                                            end_time = -1
                                        total_carbon = operational_carbon + embodied_carbon

                                        row = [
                                            system,
                                            freq,
                                            curr_load,
                                            distribution,
                                            models[tput_row['job_no']],
                                            batch_size,
                                            allocation_size,
                                            latency,
                                            tput,
                                            avg_pwr,
                                            max_pwr,
                                            total_energy,
                                            operational_carbon,
                                            embodied_carbon,
                                            total_carbon,
                                        ]
                                        rows.append(row)
    df = pd.DataFrame(rows, columns=['system', 'frequency', 'load', 'distribution', 'model', 'batch_size', 'allocation_size', 'latency', 'tput', 'avg_pwr', 'max_pwr', 'total_energy', 'operational_carbon', 'embodied_carbon', 'total_carbon'])
    df = df.sort_values(by='load')
    df.to_csv(f'./eval-results/{system}-results.csv', index=False)
    print(df.to_string())
