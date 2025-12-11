# msmp_typed_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Base directory: directory containing this script
BASE = "/Users/jetstibbe/Documents/caise_typed/src/"

# Input files:
f_msmp          = os.path.join(BASE, "msm_msmp_results.txt")
f_typed_only    = os.path.join(BASE, "msm_typed_only_results.txt")
f_typed_plusraw = os.path.join(BASE, "msm_typed_plus_raw_results.txt")

# Total number of possible pairs for the test set N*(N-1)/2
TOTAL_PAIRS = 187578

# ---------------------------------------------------------------------------
# Loading and basic utilities
# ---------------------------------------------------------------------------

def load_results(path):
    """
    Load CSV with LSH/MSM results and add 'frac_comp' = candidate_pairs / TOTAL_PAIRS.
    """
    df = pd.read_csv(path)
    df["frac_comp"] = df["cand_pairs_lsh"] / TOTAL_PAIRS
    df = df.sort_values("frac_comp")
    return df

df_msmp          = load_results(f_msmp)          # MSMP+
df_typed_only    = load_results(f_typed_only)    # MSMP+ typed only
df_typed_plusraw = load_results(f_typed_plusraw) # MSMP+ typed + raw


def compute_auc(x, y):
    """
    Numerical AUC via trapezoidal rule, after sorting by x.
    """
    x = x.to_numpy()
    y = y.to_numpy()
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric(metric, ylabel, title, outfile,
                x_from_minus=False):
    """
    Plot one metric for the three variants and write to PNG.
    Also prints AUC per curve.
    """

    plt.figure(figsize=(7, 4))

    # base x/y for AUC
    x_msmp          = df_msmp["frac_comp"].copy()
    y_msmp          = df_msmp[metric].copy()
    x_typed_only    = df_typed_only["frac_comp"].copy()
    y_typed_only    = df_typed_only[metric].copy()
    x_typed_plusraw = df_typed_plusraw["frac_comp"].copy()
    y_typed_plusraw = df_typed_plusraw[metric].copy()

    # For F1*: drop rows where PC_lsh == 0 (uninformative)
    if metric == "F1star_lsh":
        mask_msmp          = df_msmp["PC_lsh"] > 0
        mask_typed_only    = df_typed_only["PC_lsh"] > 0
        mask_typed_plusraw = df_typed_plusraw["PC_lsh"] > 0

        x_msmp          = df_msmp.loc[mask_msmp, "frac_comp"]
        y_msmp          = df_msmp.loc[mask_msmp, metric]
        x_typed_only    = df_typed_only.loc[mask_typed_only, "frac_comp"]
        y_typed_only    = df_typed_only.loc[mask_typed_only, metric]
        x_typed_plusraw = df_typed_plusraw.loc[mask_typed_plusraw, "frac_comp"]
        y_typed_plusraw = df_typed_plusraw.loc[mask_typed_plusraw, metric]

    # PQ
    if metric == "PQ_lsh":
        msmp_mask      = df_msmp["PQ_lsh"] > 0.0
        typed_mask     = df_typed_only["PQ_lsh"] > 0.0
        typed_raw_mask = df_typed_plusraw["PQ_lsh"] > 0.0

        x_msmp          = df_msmp.loc[msmp_mask, "frac_comp"]
        y_msmp          = df_msmp.loc[msmp_mask, "PQ_lsh"]
        x_typed_only    = df_typed_only.loc[typed_mask, "frac_comp"]
        y_typed_only    = df_typed_only.loc[typed_mask, "PQ_lsh"]
        x_typed_plusraw = df_typed_plusraw.loc[typed_raw_mask, "frac_comp"]
        y_typed_plusraw = df_typed_plusraw.loc[typed_raw_mask, "PQ_lsh"]


    # --- PLOT ---
    if metric == "PQ_lsh":
        plt.plot(x_msmp,          y_msmp,          label="MSMP+", linewidth=0.65)
        plt.plot(x_typed_only,    y_typed_only,    label="Typed-MSMP+", linewidth=0.65)
        plt.plot(x_typed_plusraw, y_typed_plusraw, label="Combined-MSMP+", linewidth=0.65)
    else:
        plt.plot(x_msmp,          y_msmp,          label="MSMP+", linewidth=0.65)
        plt.plot(x_typed_only,    y_typed_only,    label="Typed-MSMP+", linewidth=0.65)
        plt.plot(x_typed_plusraw, y_typed_plusraw, label="Combined-MSMP+", linewidth=0.65)

    plt.xlabel("fraction of comparisons")
    plt.ylabel(ylabel)
    plt.title(title)

    xmin = -0.05 if x_from_minus else 0.0
    xmax = max(
        df_msmp["frac_comp"].max(),
        df_typed_only["frac_comp"].max(),
        df_typed_plusraw["frac_comp"].max()
    )
    plt.xlim(xmin, xmax * 1.02)

    # Metric-specific zooms (same choices as before)
    if metric == "PQ_lsh":
        plt.xlim(-0.00005, 0.1)
    elif metric == "F1star_lsh":
        plt.xlim(-0.0005, 0.5)
        plt.ylim(0, 0.11)
    elif metric == "PC_lsh":
        plt.xlim(-0.0005, 1.0)
        plt.ylim(0, 1)
    elif metric == "F1_clu":
        plt.xlim(-0.0005, 0.8)
    else:
        ymax = max(
            df_msmp[metric].max(),
            df_typed_only[metric].max(),
            df_typed_plusraw[metric].max(),
        )
        plt.ylim(0, ymax + 0.02)

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, outfile), dpi=200)
    plt.close()

    # AUC diagnostics per metric/variant
    if len(x_msmp) > 1 and len(x_typed_only) > 1 and len(x_typed_plusraw) > 1:
        auc_msmp  = compute_auc(x_msmp,          y_msmp)
        auc_typed = compute_auc(x_typed_only,    y_typed_only)
        auc_comb  = compute_auc(x_typed_plusraw, y_typed_plusraw)

        rel_typed = (auc_typed - auc_msmp) / auc_msmp * 100.0
        rel_comb  = (auc_comb  - auc_msmp) / auc_msmp * 100.0

        print(f"{metric} AUC:")
        print(f"  MSMP+           : {auc_msmp:.6g}")
        print(f"  Typed-MSMP+     : {auc_typed:.6g}  (Δ = {rel_typed:+.2f} % vs MSMP+)")
        print(f"  Combined-MSMP+  : {auc_comb:.6g}  (Δ = {rel_comb:+.2f} % vs MSMP+)")
        print()


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

# Figure 1: LSH Pair Completeness
plot_metric(
    metric="PC_lsh",
    ylabel="Pair Completeness",
    title="LSH Pair Completeness",
    outfile="msm_pc_lsh_new.png",
    x_from_minus=False,
)

# Figure 2: LSH Pair Quality
plot_metric(
    metric="PQ_lsh",
    ylabel="Pair Quality",
    title="LSH Pair Quality",
    outfile="msm_pq_lsh.png",
    x_from_minus=False,
)

# Figure 3: LSH F1*
plot_metric(
    metric="F1star_lsh",
    ylabel="F1*",
    title="LSH F1*",
    outfile="msm_f1star_lsh_new.png",
    x_from_minus=False,
)

# Figure 4: MSM clustering F1
plot_metric(
    metric="F1_clu",
    ylabel="F1 clustering",
    title="Clustering F1",
    outfile="msm_f1_clu_new.png",
    x_from_minus=False,
)
