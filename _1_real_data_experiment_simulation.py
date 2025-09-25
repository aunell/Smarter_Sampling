import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from generate_data import calculate_icc_cohen, calculate_icc_icc, calculate_icc_krip
from simulation import *

# ------------------------
# Centralized File Locations
# ------------------------
DATE = datetime.now().strftime("%Y-%m-%d")
BASE_DIR = "/share/pi/nigam/users/aunell/PAC_Judge/results"
JUDGE_SCORES_DIR = os.path.join(BASE_DIR, "judge_scores")
IMAGES_DIR = os.path.join(BASE_DIR, "images", f"{DATE}-rwe-icc-results-150")
# IMAGES_DIR = "/share/pi/nigam/users/aunell/PAC_Judge/results/images/2025-09-18-claude-rwe-icc-results"
IMAGES_DIR="/share/pi/nigam/users/aunell/PAC_Judge/results/images/2025-09-23-1-rwe-icc-results-100"
calculate_icc_fn = calculate_icc_icc

os.makedirs(IMAGES_DIR, exist_ok=True)


# ------------------------
# Data Loading
# ------------------------
def load_data(data_path, categorical=False, main_model="openai"):
    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        data = json.load(f)

    scores = {
        'expensive_score': [int(r['original_score']) if categorical else r['original_score']
                            for r in data['detailed_results']],
        'cheap_score': []
    }

    for result in data['detailed_results']:
        scores['cheap_score'].append(
            result.get('evaluation', {}).get('evaluation', {}).get('score',
            result.get('evaluation', {}).get('score', None))
        )

    size = 150
    cheap_ratings_2 = None

    # Attempt to load second set of cheap ratings
    parts = data_path.split('_')
    data_path_2 = '_'.join(parts[:-1]) + '_2_' + '_'.join(parts[-1:])
    if main_model=="claude":
        data_path_2= data_path_2.replace("_2_", "_")
    if os.path.exists(data_path_2):
        print(f"Loading second set of ratings from {data_path_2}")
        with open(data_path_2) as f2:
            data_2 = json.load(f2)

        scores_2 = [
            r.get('evaluation', {}).get('evaluation', {}).get('score',
            r.get('evaluation', {}).get('score', None))
            for r in data_2['detailed_results']
        ]
        cheap_ratings_2 = np.array(scores_2[:300])
        size = min(len(scores['cheap_score']), len(scores['expensive_score']), len(cheap_ratings_2))
        cheap_ratings_2 = cheap_ratings_2[:size]

    return (
        np.array(scores['cheap_score'][:size]),
        np.array(scores['expensive_score'][:size]).astype(int),
        cheap_ratings_2
    )


# ------------------------
# Evaluation
# ------------------------
def evaluate_selection(cheap_ratings, expensive_ratings, selected_indices, strategy_name, oracle_icc=None):
    try:
        if not oracle_icc:
            oracle_icc = calculate_icc_fn(cheap_ratings, expensive_ratings)
        subset_icc = calculate_icc_fn(
            [cheap_ratings[i] for i in selected_indices],
            [expensive_ratings[i] for i in selected_indices]
        )
        preservation_error = abs(subset_icc - oracle_icc)

        return {
            'strategy': strategy_name,
            'selected_indices': selected_indices,
            'subset_icc': subset_icc,
            'oracle_icc': oracle_icc,
            'preservation_error': preservation_error,
            'n_selected': len(selected_indices)
        }, oracle_icc
    except Exception as e:
        print(f"Error in {strategy_name}: {e}")


# ------------------------
# Simulation
# ------------------------
def run_simulation(cheap_ratings, expensive_ratings, n_expensive_range,
                   n_rollouts=100, cheap_ratings_2=None):
    """
    Run simulation for multiple selection strategies.
    `seeds`: list of integer seeds for reproducibility. If None, use np.random default.
    """
    print("\nRunning simulation...")
    all_results = []

    strategies = {
        "Stratified QBC": hybrid_selection,
        "QBC": QBC_selection,
        'Random': random_selection,
        'Stratified': stratified_selection,
        'Cluster': cluster_selection,
        'Maximum-Variation': maximum_variation_selection,
        'Density-Based': density_based_selection
    }

    if cheap_ratings_2 is None:
        for key in ["Stratified QBC", "QBC"]:
            strategies.pop(key, None)

    oracle_icc=None
    for n_expensive in tqdm(n_expensive_range, desc="Testing different n_expensive"):
        for seed in range(n_rollouts):
            for strategy_name, strategy_func in strategies.items():
                if strategy_name in {"Stratified QBC", "QBC"}:
                    selected = strategy_func(cheap_ratings, cheap_ratings_2, n_expensive, seed=seed)
                else:
                    selected = strategy_func(cheap_ratings, n_expensive, seed=seed)

                result, oracle_icc = evaluate_selection(cheap_ratings, expensive_ratings, selected, strategy_name, oracle_icc)
                if result:
                    # Record rollout, n_expensive, and seed
                    result.update({'rollout': seed, 'n_expensive': n_expensive, 'seed': seed})
                    all_results.append(result)

    return pd.DataFrame(all_results)

import pandas as pd
import os

def save_strategy_csvs(df, out_dir="csv_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    for dataset in df["dataset_name"].unique():
        subset = df[df["dataset_name"] == dataset]

        # pivot mean
        mean_pivot = subset.pivot(
            index="n_expensive",
            columns="strategy",
            values="mean"
        )

        # pivot std and rename columns
        std_pivot = subset.pivot(
            index="n_expensive",
            columns="strategy",
            values="std"
        )
        std_pivot.columns = [f"{c}_std" for c in std_pivot.columns]

        # combine mean + std into one table
        final_df = mean_pivot.join(std_pivot)

        # save CSV
        out_path = os.path.join(out_dir, f"{dataset}.csv")
        final_df.to_csv(out_path, float_format="%.6f")
        print(f"Saved: {out_path}")


# ------------------------
# Plotting
# ------------------------
def plot_progression_results(all_dataset_results, show_seed_points=False):
    print("\nPlotting progression results...")
    all_dataset_results['dataset_name'] = all_dataset_results['dataset'].str.extract(r'results_([^_]+)_')

    mean_results = (
        all_dataset_results
        .groupby(['dataset_name', 'strategy', 'n_expensive'])['preservation_error']
        .agg(['mean', 'std'])
        .reset_index()
    )
    # save_strategy_csvs(mean_results)

    for dataset in mean_results['dataset_name'].unique():
        if "oracle_icc" in all_dataset_results.columns:
            oracle_val = all_dataset_results.loc[
                all_dataset_results['dataset_name'] == dataset, "oracle_icc"
            ].iloc[0]
            print(f"Dataset: {dataset} | Oracle ICC: {oracle_val:.3f}")
        plt.figure(figsize=(10, 6))
        dataset_data = mean_results[mean_results['dataset_name'] == dataset]

        for strategy in dataset_data['strategy'].unique():
            strategy_data = dataset_data[dataset_data['strategy'] == strategy]
            plt.plot(strategy_data['n_expensive'], strategy_data['mean'], label=strategy, marker='o')
            print(dataset, strategy_data['mean'])
            if show_seed_points:
                seed_points = all_dataset_results[
                    (all_dataset_results['dataset_name'] == dataset) &
                    (all_dataset_results['strategy'] == strategy)
                ]
                plt.scatter(seed_points['n_expensive'], seed_points['preservation_error'],
                            alpha=0.3, s=20, color='gray')
            
        dataset_names={"hanna": "HANNA",
                       "mslr": "MSLR",
                       "medval": "MedVAL",
                       "summeval": "SummEval"}

        plt.title(
            f'Estimation Error vs Number of Expensive Ratings\nDataset: {dataset_names[dataset]}',
            fontsize=20
        )
        plt.xlabel('Number of Expensive Ratings', fontsize=18)
        plt.ylabel('Estimation Error', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True)

        # Increase tick label font sizes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        filename = f"progression_analysis_{dataset.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(os.path.join(IMAGES_DIR, filename))
        print("SAVED FIG TO:", os.path.join(IMAGES_DIR, filename))
        plt.close()

    mean_results.to_json(os.path.join(IMAGES_DIR, "progression_mean_results.json"),
                         orient="records", indent=2)
    return mean_results



# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print("Starting simulation experiment...")

    n_rollouts = 20

    all_dataset_results = []

    for root, dirs, files in os.walk(JUDGE_SCORES_DIR):
        for file in files:
            if file.endswith(".json") and "2" not in file:
                file_path = os.path.join(root, file)
                filename = file.split(".")[0]
                results_file = os.path.join(IMAGES_DIR, f"{filename}_simulation_results.csv")

                if not os.path.exists(results_file):
                    print(f"\nProcessing {file}")
                    cheap_ratings, expensive_ratings, cheap_ratings_2 = load_data(file_path, main_model="openai")

                    total_annotations = 100 #min(int(len(expensive_ratings) * 0.2), 100)
                    print("len(expensive_ratings)", len(expensive_ratings) )
                    step = total_annotations//20
                    n_expensive_range = range(step, total_annotations, step)

                    simulation_results = run_simulation(
                        cheap_ratings, expensive_ratings,
                        n_expensive_range, n_rollouts,
                        cheap_ratings_2=cheap_ratings_2)
                    simulation_results['dataset'] = filename
                    simulation_results.to_csv(results_file, index=False)
                    all_dataset_results.append(simulation_results)
                else:
                    print(f"\nSkipping {file} - results already exist")
                    simulation_results = pd.read_csv(results_file)
                    all_dataset_results.append(simulation_results)

    all_dataset_results = pd.concat(all_dataset_results, ignore_index=True)
    plot_progression_results(all_dataset_results)
    print("Simulation experiment complete!")
