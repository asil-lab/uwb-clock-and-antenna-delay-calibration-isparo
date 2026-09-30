import os
import sys
import yaml
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture

# IEEE Publication Style Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "patch.force_edgecolor": True,
    "patch.facecolor": "none"
})

# High-contrast color definitions
CLR_PAIRWISE = "#1f77b4"
CLR_CENTRAL  = "#f47724"

# Constants
DWT_TIME_UNITS        = 1.0 / 499.2e6 / 128.0
ELECTROMAGNETIC_SPEED = 299_792_458.0

# Data Helpers

def counters_with_timestamps(csv_path, timestamp_col="timestamp_seconds"):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["message_id", "message_transmitter_id"])
    out = (df.pivot(index="message_id", columns="message_transmitter_id", values=timestamp_col)
             .reset_index())
    out.columns = ["message_id"] + [f"node_{c}" for c in out.columns[1:]]
    return out


def drop_nan_rows(df, cls_to_check=None, return_dropped=False):
    if cls_to_check is None:
        cls_to_check = [c for c in df.columns if c.startswith("node_")]
    mask = df[cls_to_check].isna().any(axis=1)
    clean = df.loc[~mask].reset_index(drop=True)
    if return_dropped:
        print(f"Dropped {mask.sum()} rows containing NaN.")
        return clean, df.loc[mask].reset_index(drop=True)
    print(f"Dropped {mask.sum()} rows containing NaN.")
    return clean


def filter_common_counters(dfs, counter_col="message_id"):
    if not dfs:
        return []
    common = set(dfs[0][counter_col])
    for df in dfs[1:]:
        common &= set(df[counter_col])
    common = sorted(common)
    print(f"Common counters kept: {len(common)}")
    return [df[df[counter_col].isin(common)].reset_index(drop=True) for df in dfs]


def flatten_timestamps(df, node_A, node_B):
    cls = [f"node_{n}" for n in [node_A, node_B]]
    missing = [c for c in cls if c not in df.columns]
    if missing:
        raise ValueError(f"Missing node columns: {missing}")
    return df[cls].to_numpy().flatten().reshape(-1, 1)


def generate_e(K):
    return np.array([(-1) ** (i + 1) for i in range(2 * K)]).reshape(-1, 1)


def load_data(experiment_date, experiment_run, experiment_try,
              base_dir="data", tty_indices=None, filter=True):
    fp = f"{base_dir}/{experiment_date}/{experiment_date}-{experiment_run}"
    pattern = os.path.join(fp,
        f"{experiment_date}-{experiment_run}-{experiment_try}_ttyACM*.csv")
    all_files = glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    index_to_file = {}
    for f in all_files:
        base = os.path.basename(f)
        try:
            idx = int(base.split("ttyACM")[1].split(".csv")[0])
            index_to_file[idx] = f
        except ValueError:
            continue
    indices = sorted(index_to_file) if tty_indices is None else list(tty_indices)
    datas, filepaths = [], []
    for idx in indices:
        d = counters_with_timestamps(index_to_file[idx])
        d = drop_nan_rows(d)
        datas.append(d)
        filepaths.append(index_to_file[idx])
    if filter:
        datas = filter_common_counters(dfs=datas) 
    return datas, indices, filepaths


def load_parameters(experiment_date, experiment_run, experiment_try,
                    base_dir="data", suffix="parameters.yaml"):
    path = f"{base_dir}/{experiment_date}/{experiment_date}-{experiment_run}/{suffix}"
    with open(path) as fh:
        return yaml.safe_load(fh)


# Mathematical Processing Engines

def build_system(datas, K_min, K_max, pairs, mus, ranging):
    N = len(datas)
    K = K_max - K_min
    P = len(pairs)
    
    if mus is None:
        mus = np.zeros((N, 2))
    mus = np.asarray(mus)
    
    e = generate_e(K // 2)
    ones = np.ones_like(e)
    T = np.zeros((K * P, N))
    E_1 = np.zeros((K * P, N))
    for p, (i, j) in enumerate(pairs):
        t_ij = flatten_timestamps(datas[i], i, j)[K_min:K_max]
        t_ji = flatten_timestamps(datas[j], i, j)[K_min:K_max]
        t_ij = t_ij - np.tile(mus[i] * np.array([-1, 1]), K // 2).reshape(-1, 1)
        t_ji = t_ji + np.tile(mus[j] * np.array([-1, 1]), K // 2).reshape(-1, 1)
        rs, re = p * K, p * K + K
        T[rs:re, i] = t_ij.ravel()
        T[rs:re, j] = -t_ji.ravel()
        E_1[rs:re, i] = ones.ravel()
        E_1[rs:re, j] = -ones.ravel()
    T_bar = T[:, 1:]
    E_1_bar = E_1[:, 1:]
    if ranging:
        A_bar = np.hstack([T_bar, E_1_bar, np.kron(-np.eye(P), e)])
    else:
        A_bar = np.hstack([T_bar, E_1_bar])
    return A_bar, -T[:, 0]


def solve_pairwise_ls(datas, K_min, K_max, pairs, mus, ranging):
    K = K_max - K_min
    P = len(pairs)
    pairwise_residuals = np.zeros(K * P)
    for p, (i, j) in enumerate(pairs):
        sub_datas = [datas[i], datas[j]]
        sub_pairs = [(0, 1)]
        sub_mus = np.array([mus[i], mus[j]])
        A_sub, t_sub = build_system(sub_datas, K_min, K_max, sub_pairs, sub_mus, ranging)
        theta_sub = np.linalg.pinv(A_sub) @ t_sub
        pairwise_residuals[p * K : (p + 1) * K] = A_sub @ theta_sub - t_sub
    return pairwise_residuals


def solve_centralized_ls(datas, K_min, K_max, pairs, mus, ranging):
    A_bar, t_1 = build_system(datas, K_min, K_max, pairs, mus, ranging)
    theta_ls = np.linalg.pinv(A_bar) @ t_1
    return theta_ls, A_bar, t_1

def solve_centralized_antenna_delays(theta_vector, n_nodes, pairs_list, true_geometric_delays):
    P = len(pairs_list)
    delta_1_ref = 2.5740469136689183e-07
    
    start_idx = 2 * (n_nodes - 1)
    tau_tilde_hat = theta_vector[start_idx : start_idx + P].reshape(-1, 1)
    tau = np.asarray(true_geometric_delays).reshape(-1, 1)
    y = tau_tilde_hat - tau  
    
    B_full = np.zeros((P, n_nodes))
    for p, (i, j) in enumerate(pairs_list):
        B_full[p, i] = 1.0
        B_full[p, j] = 1.0
        
    B_ref_column = B_full[:, 0].reshape(-1, 1)
    y_adjusted = y - (B_ref_column * delta_1_ref)
    B_reduced = B_full[:, 1:]
    
    delta_reduced, _, _, _ = np.linalg.lstsq(B_reduced, y_adjusted, rcond=None)
    
    delta_hat = np.zeros(n_nodes)
    delta_hat[0] = delta_1_ref
    delta_hat[1:] = delta_reduced.flatten()
    
    return delta_hat


# Plotting Helpers

def _fit_gmm(data_ns):
    gmm = GaussianMixture(n_components=2, random_state=42).fit(data_ns.reshape(-1, 1))
    return gmm


def plot_per_link_histograms(res_p_ns, res_c_ns, pairs, K, node_labels=None):
    P = len(pairs)
    if node_labels is None:
        node_labels = [f"N{i+1}–N{j+1}" for i, j in pairs]

    fig, axes = plt.subplots(P, 2, figsize=(3.25, 0.575 * P), sharex=True, sharey=False)
    if P == 1:
        axes = np.array([axes])

    print(f"\n{'Link Pair':<12} | {'Method':<14} | {'Mean (ns)':>9} | {'Std (ns)':>8} | {'Max (ns)':>8}")
    print("-" * 65)

    for p in range(P):
        ax_p = axes[p, 0]  
        ax_c = axes[p, 1]  
        
        data_p = res_p_ns[p * K : (p + 1) * K]
        data_c = res_c_ns[p * K : (p + 1) * K]
        n_samples = len(data_c)

        combined_min = min(np.min(data_p), np.min(data_c))
        combined_max = max(np.max(data_p), np.max(data_c))
        edges = np.linspace(combined_min, combined_max, 25)
        bin_width = edges[1] - edges[0]

        mean_p, std_p = np.mean(data_p), np.std(data_p)
        mean_c, std_c = np.mean(data_c), np.std(data_c)

        # Left Column: Pairwise LS
        ax_p.hist(data_p, bins=edges, density=False, color=CLR_PAIRWISE, alpha=0.5, edgecolor=CLR_PAIRWISE, lw=0.6)
        ax_p.set_ylabel(node_labels[p], fontsize=7, fontweight="normal", labelpad=2)
        ax_p.tick_params(labelsize=6, direction="in", pad=1)
        ax_p.grid(True, alpha=0.1, linestyle=":")

        gmm_p = _fit_gmm(data_p)
        x_p = np.linspace(edges[0], edges[-1], 200)
        gmm_counts_p = np.exp(gmm_p.score_samples(x_p.reshape(-1, 1))) * n_samples * bin_width
        ax_p.plot(x_p, gmm_counts_p, color="#000000", lw=0.7, linestyle="-")
        ax_p.axvline(mean_p, color="#000000", lw=0.7, linestyle="--")

        text_p = f"$\\bar{{r}}={mean_p:+.1f}$ ns\n$\\sigma={std_p:.1f}$ ns"
        ax_p.text(0.95, 0.92, text_p, transform=ax_p.transAxes, fontsize=5.5,
                  va="top", ha="right", color="#333333")

        # Right Column: Centralized LS
        ax_c.hist(data_c, bins=edges, density=False, color=CLR_CENTRAL, alpha=0.5, edgecolor=CLR_CENTRAL, lw=0.6)
        ax_c.tick_params(labelsize=6, direction="in", pad=1)
        ax_c.grid(True, alpha=0.1, linestyle=":")

        gmm_c = _fit_gmm(data_c)
        x_c = np.linspace(edges[0], edges[-1], 200)
        gmm_counts_c = np.exp(gmm_c.score_samples(x_c.reshape(-1, 1))) * n_samples * bin_width
        ax_c.plot(x_c, gmm_counts_c, color="#000000", lw=0.7, linestyle="-")
        ax_c.axvline(mean_c, color="#000000", lw=0.7, linestyle="--")

        text_c = f"$\\bar{{r}}={mean_c:+.1f}$ ns\n$\\sigma={std_c:.1f}$ ns"
        ax_c.text(0.95, 0.92, text_c, transform=ax_c.transAxes, fontsize=5.5,
                  va="top", ha="right", color="#333333")

        if p == 0:
            ax_p.set_title("Pairwise LS", fontsize=8, fontweight="normal", pad=4)
            ax_c.set_title("Centralized LS", fontsize=8, fontweight="normal", pad=4)

        if p == P - 1:
            ax_p.set_xlabel("Residual (ns)", fontsize=7, labelpad=2)
            ax_c.set_xlabel("Residual (ns)", fontsize=7, labelpad=2)

        print(f"{node_labels[p]:<12} | Pairwise LS | {mean_p:>9.2f} | {std_p:>8.2f} | {np.max(np.abs(data_p)):>8.2f}")
        print(f"{node_labels[p]:<12} | Central LS  | {mean_c:>9.2f} | {std_c:>8.2f} | {np.max(np.abs(data_c)):>8.2f}")

    fig.align_ylabels(axes[:, 0])

    legend_els = [
        Line2D([0], [0], color="#000000", lw=0.7, linestyle="-", label="GMM Model Fit"),
        Line2D([0], [0], color="#000000", lw=0.7, linestyle="--", label="Distribution Mean"),
        plt.Rectangle((0,0), 1, 1, fc=CLR_PAIRWISE, alpha=0.6, ec=CLR_PAIRWISE, label="Pairwise LS"),
        plt.Rectangle((0,0), 1, 1, fc=CLR_CENTRAL, alpha=0.6, ec=CLR_CENTRAL, label="Centralized LS")
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2, frameon=True, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.22, wspace=0.18, bottom=0.18 if P > 3 else 0.28)
    plt.savefig("benchmark_ls_histograms.pdf", format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()


def plot_summary_comparison(res_p_ns, res_c_ns, pairs, K, node_labels=None):
    P = len(pairs)
    if node_labels is None:
        node_labels = [f"N{i+1}–N{j+1}" for i, j in pairs]

    fig, axes = plt.subplots(4, 1, figsize=(1.75, 3.9))

    # [0] Boxplot Distribution Profiles
    ax = axes[0]
    box_data = []
    box_colors = []
    for p in range(P):
        box_data.extend([res_p_ns[p*K:(p+1)*K], res_c_ns[p*K:(p+1)*K]])
        box_colors.extend([CLR_PAIRWISE, CLR_CENTRAL])
        
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.5)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.4)
    for median in bp['medians']:
        median.set_color("#222222")
        median.set_linewidth(0.6)
        
    ax.set_xticks(np.arange(1.5, 2 * P + 0.5, 2))
    ax.set_xticklabels(node_labels, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Residual (ns)", fontsize=7)
    ax.grid(True, alpha=0.1, linestyle=":")
    ax.tick_params(labelsize=6, pad=1)
    
    box_legends = [
        plt.Rectangle((0,0), 1, 1, fc=CLR_PAIRWISE, alpha=0.5, ec="black", label="P"),
        plt.Rectangle((0,0), 1, 1, fc=CLR_CENTRAL, alpha=0.5, ec="black", label="C")
    ]
    ax.legend(handles=box_legends, fontsize=6, loc="upper right", framealpha=0.6, borderpad=0.2, handletextpad=0.2)

    # [1] Root-Mean-Square Error (RMSE) Profiles
    ax = axes[1]
    xpos = np.arange(P)
    wb = 0.35
    rmse_p = np.array([np.sqrt(np.mean(res_p_ns[p*K:(p+1)*K]**2)) for p in range(P)])
    rmse_c = np.array([np.sqrt(np.mean(res_c_ns[p*K:(p+1)*K]**2)) for p in range(P)])
    
    ax.bar(xpos - wb/2, rmse_p, wb, color=CLR_PAIRWISE, alpha=0.6, edgecolor="black", lw=0.4)
    ax.bar(xpos + wb/2, rmse_c, wb, color=CLR_CENTRAL, alpha=0.6, edgecolor="black", lw=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(node_labels, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("RMSE (ns)", fontsize=7)
    ax.grid(True, alpha=0.1, axis="y")
    ax.tick_params(labelsize=6, pad=1)

    # [2] Standard Deviation Variance Performance
    ax = axes[2]
    std_p = np.array([np.std(res_p_ns[p*K:(p+1)*K]) for p in range(P)])
    std_c = np.array([np.std(res_c_ns[p*K:(p+1)*K]) for p in range(P)])
    improvement_pct = 100 * (std_p - std_c) / (std_p + 1e-21)
    
    bars = ax.bar(xpos, improvement_pct, color="#2ca02c", alpha=0.5, edgecolor="black", lw=0.4, width=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (max(improvement_pct)*0.01), f"{h:+.1f}%", ha="center", fontsize=5, fontweight="bold")
    ax.set_xticks(xpos)
    ax.set_xticklabels(node_labels, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Reduction (%)", fontsize=7)
    ax.grid(True, alpha=0.1, axis="y")
    ax.tick_params(labelsize=6, pad=1)

    # [3] Cumulative CDF Error Profiles
    ax = axes[3]
    sorted_p = np.sort(np.abs(res_p_ns))
    sorted_c = np.sort(np.abs(res_c_ns))
    y_cdf = np.linspace(0, 1, len(res_p_ns))
    
    ax.plot(sorted_p, y_cdf, color=CLR_PAIRWISE, lw=1.0, label="P")
    ax.plot(sorted_c, y_cdf, color=CLR_CENTRAL, lw=1.0, label="C")
    ax.set_xlabel("Abs Error (ns)", fontsize=7, labelpad=2)
    ax.set_ylabel("CDF", fontsize=7)
    ax.grid(True, alpha=0.15, linestyle=":")
    ax.tick_params(labelsize=6, pad=1)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.7, handletextpad=0.2)

    labels = ["(a)", "(b)", "(c)", "(d)"]
    for idx, panel in enumerate(axes):
        panel.text(0.015, 0.94, labels[idx], transform=panel.transAxes, fontsize=8, fontweight="bold", va="top")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("benchmark_ls_summary.pdf", format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()


def print_estimated_clock_parameters(theta_vector, n_nodes, pairs_list):
    print("\n" + "=" * 55)
    print("      CLOCK SYNCHRONIZATION PARAMETERS       ")
    print("=" * 55)
    
    print(f"Node 1 (Global Reference Anchor):")
    print(f"  -> Clock Skew (alpha_1)   : 1.00000000")
    print(f"  -> Clock Offset (beta_1) : 0.00000000 s")
    
    idx = 0
    for i in range(1, n_nodes):
        print(f"Node {i+1}:")
        print(f"  -> Clock Skew (alpha_{i+1})   : {theta_vector[idx]:.8f}")
        idx += 1
        
    for i in range(1, n_nodes):
        offset_seconds = theta_vector[idx]
        print(f"Node {i+1}:")
        print(f"  -> Clock Offset (beta_{i+1}) : {offset_seconds:+.8e} s")
        idx += 1
        
    if len(theta_vector) > 2 * (n_nodes - 1):
        print("-" * 55)
        print("Link-level Aggregate Biased Propagation Delays (tau~):")
        for p, (i, j) in enumerate(pairs_list):
            tau_ns = theta_vector[idx] * 1e9
            print(f"  -> Link N{i+1}–N{j+1} (tau~_{i+1}{j+1}) : {tau_ns:.4f} ns")
            idx += 1
    print("=" * 55 + "\n")


def print_antenna_calibration_results(delta_vector):
    print("=" * 55)
    print("        CENTRALIZED INDIVIDUAL ANTENNA DELAYS (delta)        ")
    print("=" * 55)
    for idx, delay_sec in enumerate(delta_vector):
        print(f"Node {idx+1}:")
        print(f"  -> Antenna Calibration Error (delta_{idx+1}) : {delay_sec} s ({delay_sec*1e9:+.3f} ns)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    experiment_date, experiment_run, experiment_try = "2026-06-01", "01", "01"
    K_min, K_max = 300, 380
    K = K_max - K_min
    
    try:
        parameters = load_parameters(experiment_date, experiment_run, experiment_try)
        datas, _, _ = load_data(experiment_date, experiment_run, experiment_try, filter=True)
        
        N_nodes = len(datas)
        pairs = [(i, j) for i in range(N_nodes) for j in range(i + 1, N_nodes)]
        P = len(pairs)
        
        try:
            node_labels = [p["label"] for p in parameters["node_pairs"]]
        except (KeyError, TypeError):
            node_labels = [f"N{i+1}–N{j+1}" for i, j in pairs]
            
        print("=" * 75)
        print(f" Executing Global LS Architecture Benchmark: {N_nodes} Nodes -> {P} Links")
        print("=" * 75)
        
        mus_init = np.zeros((N_nodes, 2))
        
        res_p_ns = solve_pairwise_ls(datas, K_min, K_max, pairs, mus_init, ranging=True) * 1e9
        theta_ls, A_bar, t_1 = solve_centralized_ls(datas, K_min, K_max, pairs, mus_init, ranging=True)
        res_c_ns = (A_bar @ theta_ls - t_1) * 1e9
        
        true_distances = np.asarray([pair['distance'] for pair in parameters['node_pairs']])
        true_delays = [d / ELECTROMAGNETIC_SPEED for d in true_distances[:P]]

    except FileNotFoundError:
        print("\n[!] Data files absent!")
        sys.exit()

    print_estimated_clock_parameters(theta_ls, N_nodes, pairs)
    delta_vector = solve_centralized_antenna_delays(theta_ls, N_nodes, pairs, true_delays)
    print_antenna_calibration_results(delta_vector)

    # plot_summary_comparison(res_p_ns, res_c_ns, pairs, K, node_labels)
    plot_per_link_histograms(res_p_ns, res_c_ns, pairs, K, node_labels)
    print("\nProcessing complete. Baseline LS matrix suite resolved successfully.")