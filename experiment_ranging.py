import os
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Optional

# Speed of light for physical delay conversions
ELECTROMAGNETIC_SPEED = 299792458.0

# DATA PREPROCESSING HELPERS
def counters_with_timestamps(csv_path: str, timestamp_col: str = "timestamp_seconds") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["message_id", "transmitter_id"])
    out = df.pivot(index="message_id", columns="transmitter_id", values=timestamp_col).reset_index()
    out.columns = ["message_id"] + [f"node_{c}" for c in out.columns[1:]]
    return out

def drop_nan_rows(df: pd.DataFrame, cols_to_check: Optional[List[str]] = None, return_dropped: bool = False) -> pd.DataFrame:
    if cols_to_check is None:
        cols_to_check = [c for c in df.columns if c.startswith("node_")]
    mask_nan = df[cols_to_check].isna().any(axis=1)
    df_clean = df.loc[~mask_nan].reset_index(drop=True)
    if return_dropped:
        return df_clean, df.loc[mask_nan].reset_index(drop=True)
    return df_clean

def filter_common_counters(dfs: List[pd.DataFrame], counter_col: str = "message_id") -> List[pd.DataFrame]:
    if not dfs: return []
    common_counters = set(dfs[0][counter_col])
    for df in dfs[1:]:
        common_counters &= set(df[counter_col])
    common_counters = sorted(common_counters)
    return [df[df[counter_col].isin(common_counters)].reset_index(drop=True) for df in dfs]

def flatten_timestamps(df: pd.DataFrame, node_A: int, node_B: int) -> np.ndarray:
    node_cols = [f"node_{n}" for n in [node_A, node_B]]
    return df[node_cols].to_numpy().flatten().reshape(-1, 1)

def generate_e(K: int) -> np.ndarray:
    return np.array([(-1)**(i+1) for i in range(2*K)]).reshape(-1, 1)

def load_data(experiment_date, experiment_run, experiment_try, base_dir="data", filter=True):
    experiment_filepath = f"{base_dir}/{experiment_date}/{experiment_date}-{experiment_run}"
    pattern = os.path.join(experiment_filepath, f"{experiment_date}-{experiment_run}-{experiment_try}_ttyACM*.csv")
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found with pattern: {pattern}")
    index_to_file = {}
    for f in all_files:
        base = os.path.basename(f)
        try:
            idx = int(base.split("ttyACM")[1].split(".csv")[0])
            index_to_file[idx] = f
        except ValueError:
            continue
    indices = sorted(index_to_file.keys())
    datas = []
    filepaths = []
    for idx in indices:
        path = index_to_file[idx]
        data = counters_with_timestamps(path)
        data = drop_nan_rows(data)
        datas.append(data)
        filepaths.append(path)
    if filter:
        datas = filter_common_counters(datas)
    return datas, indices, filepaths

def load_parameters(experiment_date, experiment_run, experiment_try, base_dir='data', suffix='parameters.yaml'):
    experiment_filepath = f"{base_dir}/{experiment_date}/{experiment_date}-{experiment_run}/{suffix}"
    with open(experiment_filepath) as file:
        return yaml.safe_load(file)

def construct_topological_E_matrix(N: int, K: int) -> np.ndarray:
    E_base = np.array([
        [ 0, -1, -1,  0],  # Link 1->2
        [ 1,  0,  0,  1]   # Link 2->1
    ], dtype=float)
    E_scaled = np.zeros((K, 2 * N), dtype=float)
    for k in range(K):
        if k % 2 == 0:
            E_scaled[k, :] = E_base[0, :]
        else:
            E_scaled[k, :] = E_base[1, :]
    return E_scaled

def perform_ranging_for_link(datas, K_min, K_max, pair, delta_calib):
    i, j = pair
    K = K_max - K_min
    
    t_ij = flatten_timestamps(datas[i], node_A=i, node_B=j)[K_min:K_max]
    t_ji = flatten_timestamps(datas[j], node_A=i, node_B=j)[K_min:K_max]
    
    N = 2
    T = np.zeros((K, N))
    E_1 = np.zeros((K, N))
    ones = np.ones((K, 1))
    
    T[:, i] = t_ij.ravel()
    T[:, j] = -t_ji.ravel()
    E_1[:, i] = ones.ravel()
    E_1[:, j] = -ones.ravel()
    
    A_bar = np.hstack([T[:, 1:], E_1[:, 1:]])
    t_1 = -T[:, 0]
    
    theta = np.linalg.pinv(A_bar.T @ A_bar) @ A_bar.T @ t_1
    residuals = t_1.ravel() - (A_bar @ theta).ravel()
    
    minus_e = np.array([(-1)**k for k in range(K)]).reshape(-1, 1)
    tau_ij_raw = (np.linalg.pinv(minus_e) @ residuals.reshape(-1, 1)).item()
    
    d_i_tx, d_i_rx = delta_calib[2 * i], delta_calib[2 * i + 1]
    d_j_tx, d_j_rx = delta_calib[2 * j], delta_calib[2 * j + 1]
    
    calibrated_tau = np.abs(tau_ij_raw) - (d_i_tx + d_i_rx + d_j_tx + d_j_rx) / 2.0
    est_distance = np.abs(calibrated_tau) * ELECTROMAGNETIC_SPEED
    return est_distance


if __name__ == '__main__':

    experiment_date = '2026-01-23'
    runs = ['01', '02', '03']
    tries = ['01', '02', '03']
    K_min, K_max = 0, 100

    delta_ls = [2.5740469136689183e-07, 2.5714502038475544e-07, 2.572897939459782e-07, 2.5745502828102594e-07]

    results_records = []
    processed_runs = set()

    print("=" * 90)
    print(f" RUNNING MULTI-RUN MULTI-TRY RANGING ANALYSIS")
    print("=" * 90)

    for run in runs:
        for try_idx in tries:
            try:
                parameters = load_parameters(experiment_date=experiment_date, experiment_run=run, experiment_try=try_idx)
                datas, _, _ = load_data(experiment_date=experiment_date, experiment_run=run, experiment_try=try_idx, filter=True)
                
                node_pairs_list = parameters.get('node_pairs', [])
                for p_idx, pair_info in enumerate(node_pairs_list):
                    raw_nodes = pair_info['nodes']
                    i_node = 0 if raw_nodes[0] == 0x00 else 1
                    j_node = 1 if raw_nodes[1] == 0x01 else 0
                    pair = (i_node, j_node)
                    
                    true_dist = pair_info['distance']
                    est_dist_ls = perform_ranging_for_link(datas, K_min, K_max, pair, delta_ls)
                    
                    results_records.append({
                        'Run': run,
                        'Try': try_idx,
                        'Pair': f"N{i_node+1}-N{j_node+1}",
                        'True_Distance': true_dist,
                        'Est_Distance_LS': est_dist_ls,
                        'Error_LS': est_dist_ls - true_dist
                    })
                    processed_runs.add(run)
                    print(f"Run {run} | Try {try_idx} | {true_dist:6.3f}m | LS Err: {est_dist_ls - true_dist:+6.3f}m")
            except Exception as e:
                print(f"Skipping configuration Run {run} Try {try_idx} due to exception: {e}")

    df_results = pd.DataFrame(results_records)
    summary_by_run = df_results.groupby(['True_Distance', 'Run', 'Pair']).agg(
        Mean_Error_LS=('Error_LS', 'mean'),
        Std_Error_LS=('Error_LS', 'std')
    ).reset_index()

    print("\n" + "=" * 90)
    print(" METRIC SUMMARY PER EXPERIMENTAL RUN (ERROR DISTRIBUTIONS)")
    print("=" * 90)
    for _, row in summary_by_run.iterrows():
        print(f"Run: {row['Run']} | Pair: {row['Pair']} | True Dist: {row['True_Distance']:5.2f}m | "
            f"Mean Error: {row['Mean_Error_LS']:+7.4f}m | Std Dev: {row['Std_Error_LS']:6.4f}m")
    print("=" * 90 + "\n")

    # IEEE Publication Style Settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.5,
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # Standard single-column size for IEEE: 3.5 inches width
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    unique_distances = sorted(summary_by_run['True_Distance'].unique())
    sorted_runs = sorted(list(processed_runs))

    # Shift configurations slightly left/right per run variant to minimize visual overlapping
    marker_offsets = {run: (-0.03 + idx * 0.03) for idx, run in enumerate(sorted_runs)}
    marker_styles = { '01': 'o', '02': 's', '03': '^' }  
    run_colors = ["#1f77b4", '#1f77b4', '#1f77b4', '#1f77b4']

    for _, row in summary_by_run.iterrows():
        dist = row['True_Distance']
        run_id = row['Run']
        offset = marker_offsets.get(run_id, 0)
        
        ax.errorbar(dist + offset, row['Mean_Error_LS'], yerr=row['Std_Error_LS'],
                    fmt=marker_styles.get(run_id, 'o'), color=run_colors[sorted_runs.index(run_id) % len(run_colors)], 
                    markersize=3.5, capsize=2, elinewidth=0.8, markeredgecolor='black', markeredgewidth=0.4)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_xlabel('Ground Truth Distance (m)', labelpad=3)
    ax.set_ylabel('Localization Error (m)', labelpad=3)

    ax.set_xticks(unique_distances)
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker=marker_styles.get(r, 'o'), color='white', markerfacecolor=run_colors[idx % len(run_colors)], 
            markeredgecolor='black', markeredgewidth=0.5, markersize=5, label=f'Run {r}')
        for idx, r in enumerate(sorted_runs)
    ]
    legend_elements.append(Line2D([0], [0], color='black', linestyle='--', linewidth=0.7, label='Zero-Error'))

    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
            fancybox=False, edgecolor='gainsboro', framealpha=0.9, labelspacing=0.25)

    plt.tight_layout()
    plt.savefig("ranging_performance_ieee.pdf", format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()