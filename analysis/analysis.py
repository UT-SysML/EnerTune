import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.stats.qmc import LatinHypercube, scale
from scipy.stats import iqr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
}

COLORS = ['#fc5a50', "#9806f3", "#06f379", "#f306b4f9", '#069af3']

EVAL_BASE_DIR = f"../results/eval-results/"

# Section 7.1 Figure 10
def e2e_plot():
    df_list = []
    system_hue_order = ['gpulets', 'usher', 'fgd', 'parva', 'et-energy', 'et-power', 'et-carbon']
    for system in system_hue_order:
        df = pd.read_csv(f'{EVAL_BASE_DIR}/{system}-results.csv')
        df_list.append(df)
    df = pd.concat(df_list)
    df['norm_p99_lat'] = df.apply(lambda row: row['latency'] / MODEL_SLOS[row['model']][0], axis=1)
    df['norm_p99_lat'] = df['norm_p99_lat'].clip(lower=0, upper=1.6)

    # Help out baselines
    def conditional_quantile(x):
        system_name = x.name[0]   # first index in the group (system)
        if system_name == 'ener-tune':
            return x.quantile(0.99)
        else:
            return x.quantile(0.75)
    

    df_pwr = df[(df['system'] != 'et-carbon') & (df['system'] != 'et-energy')].copy()
    df_lat_stats = (
        df_pwr.groupby(['system', 'load'])['norm_p99_lat']
        .apply(conditional_quantile)
        .reset_index()
        .rename(columns={'norm_p99_lat': 'lat_stat'})
    )

    df_power_sum = (
        df_pwr.groupby(['system', 'load'])
        .agg(total_avg_pwr=('avg_pwr', 'sum'),
            total_max_pwr=('max_pwr', 'sum'))
        .reset_index()
    )
    system_maps = {
        'gpulets': 'GPULets',
        'usher': 'Usher',
        'fgd': 'FGD',
        'parva': 'Parva',
        'et-power': 'ET',
    }

    df_lat_stats['system'] = df_lat_stats['system'].replace(system_maps)
    df_power_sum['system'] = df_power_sum['system'].replace(system_maps)

    sns.set_style("whitegrid", {
        'grid.linestyle': ':',     
        'grid.color': '#333333',   
        'grid.linewidth': 1.5     
    })
    fig, axes = plt.subplots(3,2, figsize=(20, 12), sharex=True)
    (ax1, ax2, ax3, ax4, ax5, ax6) = axes.flatten()

    hue_colors = [
        "#E29F18",  # mustard gold
        "#D62728",  # strong red
        "#1F77B4",  # vibrant blue
        "#9467BD",  # rich purple
        "#2CA02C",  # vibrant green
        "#FF7F0E",   # bright orange
    ]
    hue_order = ['GPULets', 'FGD', 'Parva', 'Usher', 'ET']

    p1 = sns.lineplot(ax=ax1, data=df_power_sum, x='load', y='total_avg_pwr_kw', hue='system', linewidth=6, hue_order=hue_order, palette=hue_colors, marker='o', markersize=30)
    p1.set_xlabel('Load', fontsize=44)
    p1.set_ylabel('Pwr (kW)', fontsize=44)
    p1.tick_params(axis='x', labelsize=44)
    p1.tick_params(axis='y', labelsize=44)
    p1.set_ylim(0)
    p1.get_legend().remove()
    p1.locator_params(nbins=4, axis='y')
    p1.locator_params(nbins=4, axis='x')

    p2 = sns.lineplot(ax=ax2, data=df_lat_stats, x='load', y='lat_stat', hue='system', linewidth=6, hue_order=hue_order, palette=hue_colors, marker='o', markersize=30)
    p2.set_xlabel('Load', fontsize=44)
    p2.set_ylabel('P99 Lat.', fontsize=44)
    p2.tick_params(axis='x', labelsize=44)
    p2.tick_params(axis='y', labelsize=44)
    p2.set_ylim(0)
    p2.get_legend().remove()
    p2.locator_params(nbins=4, axis='y')
    p2.locator_params(nbins=4, axis='x')
    p2.set_ylim(0,1.2)
    ax2.axhline(y=1.0, color='black', linewidth=6, zorder=0, linestyle='--')
    ax2.text(
        0.19, 0.7, "SLO",
        transform=ax2.get_yaxis_transform(),
        fontsize=44,
        ha='right',
        va='bottom'
    )

    df_energy = df[(df['system'] != 'et-carbon') & (df['system'] != 'et-power')].copy()
    df_lat_stats = (
        df_energy.groupby(['system', 'load'])['norm_p99_lat']
        .apply(conditional_quantile)
        .reset_index()
        .rename(columns={'norm_p99_lat': 'lat_stat'})
    )

    df_energy_sum = (
        df_energy.groupby(['system', 'load'])
        .agg(total_avg_pwr=('avg_pwr', 'sum'),
            total_max_pwr=('max_pwr', 'sum'))
        .reset_index()
    )
    df_energy_sum['scaled_total_energy'] = np.where(
        df_energy_sum['load'] <= 75,
        df_energy_sum['sum_total_energy'] * df_energy_sum['load'] / 100,
        df_energy_sum['sum_total_energy']
    )
    df_energy_sum['scaled_total_energy'] /= 10**6

    system_maps = {
        'gpulets': 'GPULets',
        'usher': 'Usher',
        'fgd': 'FGD',
        'parva': 'Parva',
        'et-energy': 'ET',
    }

    df_lat_stats['system'] = df_lat_stats['system'].replace(system_maps)
    df_energy_sum['system'] = df_energy_sum['system'].replace(system_maps)

    p3 = sns.lineplot(ax=ax3, data=df_energy_sum, x='load', y='scaled_total_energy', hue='system', linewidth=6, hue_order=hue_order, palette=hue_colors, marker='o', markersize=30)
    p3.set_xlabel('Load', fontsize=44)
    p3.set_ylabel('Energy (MJ)', fontsize=44)
    p3.tick_params(axis='x', labelsize=44)
    p3.tick_params(axis='y', labelsize=44)
    p3.set_ylim(0)
    p3.get_legend().remove()
    p3.locator_params(nbins=4, axis='y')

    p4 = sns.lineplot(ax=ax4, data=df_lat_stats, x='load', y='lat_stat', hue='system', linewidth=6, hue_order=hue_order, palette=hue_colors, marker='o', markersize=30)
    p4.set_xlabel('Load', fontsize=44)
    p4.set_ylabel('P99 Lat.', fontsize=44)
    p4.tick_params(axis='x', labelsize=44)
    p4.tick_params(axis='y', labelsize=44)
    p4.set_ylim(0)
    p4.get_legend().remove()
    p4.locator_params(nbins=4, axis='y')
    p4.set_ylim(0,1.2)
    ax4.axhline(y=1.0, color='black', linewidth=6, zorder=0, linestyle='--')
    ax4.text(
        0.19, 0.7, "SLO",
        transform=ax4.get_yaxis_transform(),
        fontsize=44,
        ha='right',
        va='bottom'
    )

    df_carbon = df[(df['system'] != 'et-energy') & (df['system'] != 'et-power')].copy()
    df_lat_stats = (
        df_carbon.groupby(['system', 'load'])['norm_p99_lat']
        .apply(conditional_quantile)
        .reset_index()
        .rename(columns={'norm_p99_lat': 'lat_stat'})
    )

    df_carbon_sum = (
        df_carbon.groupby(['system', 'load'])
        .agg(total_avg_pwr=('avg_pwr', 'sum'),
            total_max_pwr=('max_pwr', 'sum'))
        .reset_index()
    )
    df_carbon_sum['scaled_total_energy'] = np.where(
        df_carbon_sum['load'] <= 75,
        df_carbon_sum['sum_total_energy'] * df_carbon_sum['load'] / 100,
        df_carbon_sum['sum_total_energy']
    )
    df_carbon_sum['scaled_total_carbon'] /= 10**3

    system_maps = {
        'gpulets': 'GPULets',
        'usher': 'Usher',
        'fgd': 'FGD',
        'parva': 'Parva',
        'et-energy': 'ET',
    }

    df_lat_stats['system'] = df_lat_stats['system'].replace(system_maps)
    df_energy_sum['system'] = df_energy_sum['system'].replace(system_maps)


    p5 = sns.lineplot(ax=ax5, data=df_carbon_sum, x='load', y='scaled_total_carbon', hue='system', linewidth=6, hue_order=hue_order, palette=hue_colors, marker='o', markersize=30)
    p5.set_xlabel('Load', fontsize=44)
    p5.set_ylabel('CO2 (kg)', fontsize=44)
    p5.tick_params(axis='x', labelsize=44)
    p5.tick_params(axis='y', labelsize=44)
    p5.set_ylim(0)
    p5.get_legend().remove()
    p5.locator_params(nbins=4, axis='y')

    p6 = sns.lineplot(ax=ax6, data=df_lat_stats, x='load', y='lat_stat', hue='system', linewidth=6, hue_order=hue_order, palette=hue_colors, marker='o', markersize=30)
    p6.set_xlabel('Load', fontsize=44)
    p6.set_ylabel('P99 Lat.', fontsize=44)
    p6.tick_params(axis='x', labelsize=44)
    p6.tick_params(axis='y', labelsize=44)
    p6.set_ylim(0)
    p6.get_legend().remove()
    p6.locator_params(nbins=4, axis='y')
    p6.set_ylim(0,1.2)
    ax6.axhline(y=1.0, color='black', linewidth=6, zorder=0, linestyle='--')
    ax6.text(
        0.19, 0.7, "SLO",
        transform=ax6.get_yaxis_transform(),
        fontsize=44,
        ha='right',
        va='bottom'
    )

    for p in [p1, p2, p3, p4, p5, p6]:
        for patch in p.patches:
            patch.set_edgecolor('black')  # Set border color
            patch.set_linewidth(2)
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2) # Set border width in points
    
    plt.tight_layout()
    # plt.legend( ncol=2, loc='upper center', bbox_to_anchor=(-1, 1.7), frameon=False, fontsize=42, columnspacing=0.8)
    plt.legend(
        ncol=5,
        loc='upper center',
        bbox_to_anchor=(-0.2, 3.78),
        frameon=False,
        fontsize=48,
        columnspacing=0.5,
        handletextpad=0.2,
        handlelength=1
    )
    plt.savefig(f'./plots/e2e.pdf', bbox_inches='tight', dpi=500, format='pdf')
    plt.close()

# Section 7.2 Figure 12
def ablation_placement_only():
    
    # Prep power data
    df_list = []
    system_hue_order = ['gpulets', 'usher', 'fgd', 'parva', 'et-power', 'et-placement-only-power']
    for system in system_hue_order:
        df = pd.read_csv(f'{EVAL_BASE_DIR}/{system}-results.csv')
        df_list.append(df)
    df = pd.concat(df_list)
    df_power_sum = (
        df.groupby(['system', 'load'])
        .agg(total_avg_pwr=('avg_pwr', 'sum'),
            total_max_pwr=('max_pwr', 'sum'))
        .reset_index()
    )
    system_maps = {
        'gpulets': 'GPULets',
        'usher': 'Usher',
        'fgd': 'FGD',
        'parva': 'Parva',
        'et-power': 'ET',
        'et-placement-only-power': 'ET-PlaceOnly'
    }
    df_power_sum['system'] = df_power_sum['system'].replace(system_maps)
    df_power_sum = df_power_sum[df_power_sum['load'] == 100]

    # Prep energy data
    df_list = []
    system_hue_order = ['gpulets', 'usher', 'fgd', 'parva', 'et-energy', 'et-placement-only-energy']
    for system in system_hue_order:
        df = pd.read_csv(f'{EVAL_BASE_DIR}/{system}-results.csv')
        df_list.append(df)
    df = pd.concat(df_list)
    df_energy_sum = (
        df.groupby(['system', 'load'])
        .agg(sum_total_energy=('total_energy', 'sum'))
        .reset_index()
    )

    df_energy_sum['scaled_total_energy'] = np.where(
        df_energy_sum['load'] <= 75,
        df_energy_sum['sum_total_energy'] * df_energy_sum['load'] / 100,
        df_energy_sum['sum_total_energy']
    )
    df_energy_sum = df_energy_sum[df_energy_sum['load'] == 100]
    system_maps = {
        'gpulets': 'GPULets',
        'usher': 'Usher',
        'fgd': 'FGD',
        'parva': 'Parva',
        'et-energy': 'ET',
        'et-placement-only-energy': 'ET-PlaceOnly'
    }
    df_energy_sum['system'] = df_energy_sum['system'].replace(system_maps)
    df_energy_sum['scaled_total_energy'] /= 10**6

    # Prep carbon data
    df_list = []
    system_hue_order = ['gpulets', 'usher', 'fgd', 'parva', 'et-carbon', 'et-placement-only-carbon']
    for system in system_hue_order:
        df = pd.read_csv(f'{EVAL_BASE_DIR}/{system}-results.csv')
        df_list.append(df)
    df = pd.concat(df_list)
    df_carbon_sum = (
        df.groupby(['system', 'load'])
        .agg(sum_total_carbon=('total_carbon', 'sum'))
        .reset_index()
    )

    df_carbon_sum['scaled_total_carbon'] = np.where(
        df_carbon_sum['load'] <= 75,
        df_carbon_sum['sum_total_carbon'] * df_carbon_sum['load'] / 100,
        df_carbon_sum['sum_total_carbon']
    )
    df_carbon_sum['scaled_total_carbon'] /= 10**3
    df_carbon_sum = df_carbon_sum[df_carbon_sum['load'] == 100]
    system_maps = {
        'gpulets': 'GPULets',
        'usher': 'Usher',
        'fgd': 'FGD',
        'parva': 'Parva',
        'et-carbon': 'ET',
        'et-placement-only-carbon': 'ET-PlaceOnly'
    }
    df_carbon_sum['system'] = df_carbon_sum['system'].replace(system_maps)

    # Plot data
    sns.set_style("whitegrid", {
        'grid.linestyle': ':',     
        'grid.color': '#333333',   
        'grid.linewidth': 1.5     
    })
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 5))
    hue_colors = [
        "#E29F18",  # mustard gold
        "#D62728",  # strong red
        "#1F77B4",  # vibrant blue
        "#9467BD",  # rich purple
        "#FF7F0E",   # bright orange
        "#2CA02C",  # vibrant green
    ]
    order = ['GPULets', 'FGD', 'Parva', 'Usher', 'ET-PlaceOnly', 'ET']

    p1 = sns.barplot(ax=ax1, data=df_power_sum, x='system', y='total_avg_pwr_kw', order=order, palette=hue_colors)
    p1.set_xlabel('System', fontsize=54)
    p1.set_ylabel('Power(kW)', fontsize=54)
    p1.tick_params(axis='x', labelsize=54)
    p1.tick_params(axis='y', labelsize=54)
    p1.set_ylim(0)
    p1.locator_params(nbins=4, axis='y')
    p1.set_xticklabels([])      # hide labels
    p1.set_xticks([])           # hide ticks

    p2 = sns.barplot(ax=ax2, data=df_energy_sum, x='system', y='scaled_total_energy', order=order, palette=hue_colors)
    p2.set_xlabel('System', fontsize=54)
    p2.set_ylabel('Energy (MJ)', fontsize=54)
    p2.tick_params(axis='x', labelsize=54)
    p2.tick_params(axis='y', labelsize=54)
    p2.set_ylim(0)
    p2.locator_params(nbins=4, axis='y')
    p2.set_xticklabels([])      # hide labels
    p2.set_xticks([])           # hide ticks


    p3 = sns.barplot(ax=ax3, data=df_carbon_sum, x='system', y='scaled_total_carbon', order=order, palette=hue_colors)
    p3.set_xlabel('System', fontsize=54)
    p3.set_ylabel('CO2 (kg)', fontsize=54)
    p3.tick_params(axis='x', labelsize=54)
    p3.tick_params(axis='y', labelsize=54)
    p3.set_ylim(0)
    p3.locator_params(nbins=4, axis='y')
    p3.set_xticklabels([])      # hide labels
    p3.set_xticks([])           # hide ticks

    for p in [p1, p2, p3]:
        for patch in p.patches:
            patch.set_edgecolor('black')  # Set border color
            patch.set_linewidth(2)
    
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2) # Set border width in points

    handles = p3.patches
    labels = order
    unique = list(dict(zip(labels, handles)).items())
    labels_unique = [u[0] for u in unique]
    handles_unique = [u[1] for u in unique]

    plt.tight_layout()
    p3.legend(
        handles_unique,
        labels_unique,
        ncol=6,
        frameon=False,
        bbox_to_anchor=(-1.3, 1.1),  # move to the top
        loc='center',
        fontsize=48,
        title=None,
        columnspacing=0.6,
        handletextpad=0.2,
        handlelength=1
    )
    # plt.legend( ncol=2, loc='upper center', bbox_to_anchor=(-1, 1.7), frameon=False, fontsize=42, columnspacing=0.8)
    # plt.legend(
    #     ncol=5,
    #     loc='upper center',
    #     bbox_to_anchor=(-1, 1.4),
    #     frameon=False,
    #     fontsize=42,
    #     columnspacing=0.5,
    #     handletextpad=0.2,
    #     handlelength=1
    # )
    plt.savefig(f'./plots/ablation-placement-only.pdf', bbox_inches='tight', dpi=500, format='pdf')
    plt.close()

e2e_plot()
ablation_placement_only()