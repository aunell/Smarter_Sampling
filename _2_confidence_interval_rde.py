import os
import pandas as pd
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt

def fisher_icc_ci(icc_value, n_subjects, n_raters, confidence_level=0.95):
    if icc_value <= 0 or icc_value >= 1:
        return np.nan, np.nan
    
    alpha = 1 - confidence_level
    df1 = n_subjects - 1
    df2 = n_subjects * (n_raters - 1)
    
    f_lower = f.ppf(alpha/2, df1, df2)
    f_upper = f.ppf(1 - alpha/2, df1, df2)
    
    F_obs = (1 + (n_raters - 1) * icc_value) / (1 - icc_value)
    F_lower = F_obs / f_upper
    F_upper = F_obs / f_lower
    
    icc_lower = (F_lower - 1) / (F_lower + n_raters - 1)
    icc_upper = (F_upper - 1) / (F_upper + n_raters - 1)
    
    return max(0, icc_lower), min(1, icc_upper)

def analyze_icc_confidence_intervals(df, n_subjects, n_raters):
    results = []
    for _, row in df.iterrows():
        icc_val = row['subset_icc']
        n_subjects = row['n_expensive']  # override
        ci_lower, ci_upper = fisher_icc_ci(icc_val, n_subjects, n_raters)
        ci_width = ci_upper - ci_lower
        results.append({
            'strategy': row['strategy'],
            'n_expensive': row['n_expensive'],
            'icc': icc_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'dataset': row['dataset'],
            'rollout': row.get('rollout', None)  # keep rollout ID if present
        })
    return pd.DataFrame(results)


def summarize_rollouts(results_df):
    """Summarize rollout variability for each strategy and budget."""
    summary = (results_df
               .groupby(['strategy', 'n_expensive'])
               .agg(ci_width_mean=('ci_width', 'mean'),
                    ci_width_std=('ci_width', 'std'),
                    ci_width_median=('ci_width', 'median'),
                    ci_width_iqr=('ci_width', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
                    n_rollouts=('ci_width', 'count'))
               .reset_index())
    return summary

def plot_ci_width_by_strategy(summary_df, dataset_name, out_path=None):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_df['strategy'].unique())))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']
    
    for i, strategy in enumerate(summary_df['strategy'].unique()):
        strategy_data = summary_df[summary_df['strategy'] == strategy].sort_values('n_expensive')
        plt.plot(strategy_data['n_expensive'], strategy_data[('ci_width','mean')],
                 marker=markers[i % len(markers)],
                 color=colors[i],
                 label=strategy,
                 linewidth=2,
                 markersize=8)
    
    plt.xlabel('Number of Expensive Ratings (n_expensive)', fontsize=14)
    plt.ylabel('Mean Confidence Interval Width', fontsize=14)
    plt.title(f"Fisher's Confidence Interval Width' ({dataset_name} dataset)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()

def plot_ci_width_with_variability(summary_df, dataset_name, out_path=None):
    plt.figure(figsize=(12, 8))
    for strategy in summary_df['strategy'].unique():
        if strategy not in ["Random", "Cluster"]:
            continue
        sub = summary_df[summary_df['strategy'] == strategy].sort_values('n_expensive')
        plt.errorbar(sub['n_expensive'],
                     sub['ci_width_mean'],
                     yerr=sub['ci_width_std'],  # std as error bars
                     label=strategy,
                     marker='o', capsize=5, linewidth=2)
    
    plt.xlabel('Number of Expensive Ratings (n_expensive)', fontsize=14)
    plt.ylabel('Mean Confidence Interval Width', fontsize=14)
    dataset_names={"hanna": "HANNA",
                       "mslr": "MSLR",
                       "medval": "MedVal",
                       "summeval": "SummEval"}
    plt.title(f"Fisher's Confidence Interval Width ({dataset_names[dataset_name]} dataset)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()

def analyze_dir_for_hanna(base_dir, n_subjects=300, n_raters=2, dataset_name="hanna"):
    all_summaries = []
    for fname in os.listdir(base_dir):
        if dataset_name in fname.lower() and fname.endswith(".csv"):
            fpath = os.path.join(base_dir, fname)
            print(f"Processing {fpath}")
            df = pd.read_csv(fpath)
            summary_df = analyze_icc_confidence_intervals(df, n_subjects, n_raters)
            summary_df = summarize_rollouts(summary_df)
            summary_df['file'] = fname
            all_summaries.append(summary_df)
    
    if not all_summaries:
        print("No hanna CSVs found in", base_dir)
        return None
    
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    # average over all files
    averaged = (combined_summary
                .groupby(['strategy', 'n_expensive'])
                .mean(numeric_only=True)
                .reset_index())
    return averaged

# Example usage:
base_dir = "../results/images/rwe-icc-results"
base_dir = "../results/images/2025-09-23-1-rwe-icc-results-100"
n_subjects = 300
n_raters = 2
DATASET="mslr"
import pandas as pd

def compute_cluster_vs_random_improvement(df, value_col="ci_width_mean"):
    """
    Compute % improvement of Cluster over Random for a given metric.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns:
                           ["strategy", "n_expensive", "n_rollouts", value_col].
        value_col (str): Column name containing the metric to compare (e.g., 'ci_width_iqr').
    
    Returns:
        pd.DataFrame: Comparison of Cluster vs Random with % improvement.
    """
    
    # Subset for Cluster and Random
    cluster_df = df[df['strategy'] == 'Cluster']
    random_df  = df[df['strategy'] == 'Random']
    print(sorted(cluster_df['n_expensive'].unique()))
    print(sorted(random_df['n_expensive'].unique()))
    # Merge on experiment settings
    merged = pd.merge(
        cluster_df, random_df,
        on=['n_expensive'],
        suffixes=('_cluster', '_random')
    )
    
    # Compute % improvement (Random - Cluster) / Random
    merged['improvement_%'] = (
        (merged[f'{value_col}_random'] - merged[f'{value_col}_cluster'])
        # / merged[f'{value_col}_random'] * 100
    )
    
    # Return tidy dataframe
    return merged[['n_expensive', 
                   f'{value_col}_cluster', f'{value_col}_random',
                   'improvement_%']]


avg_summary = analyze_dir_for_hanna(base_dir, n_subjects, n_raters, dataset_name=DATASET)
if avg_summary is not None:
    out=compute_cluster_vs_random_improvement(avg_summary)
    breakpoint()
    plot_ci_width_with_variability(avg_summary, dataset_name=DATASET,
                              out_path=f"../confidence_plot_var_{DATASET}.png")
    print("Averaged summary stats:\n", avg_summary)
    print("Saved image to:" + f"../confidence_plot_var_{DATASET}.png" )
    
