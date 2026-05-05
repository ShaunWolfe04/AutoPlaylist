import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, ndcg_score
import scipy.stats as stats
import argparse
import sys

def calculate_95_ci(data):
    """Calculates the mean and 95% Confidence Interval for a list of values, ignoring NaNs."""
    clean_data = [x for x in data if not np.isnan(x)]
    if not clean_data:
        return np.nan, np.nan
        
    n = len(clean_data)
    mean = np.mean(clean_data)
    
    if n < 2:
        return mean, 0.0
        
    sem = stats.sem(clean_data)
    ci_margin = sem * stats.t.ppf((1 + 0.95) / 2., n - 1)
    return mean, ci_margin

def evaluate_predictions(csv_file):
    try:
        # Load the 7-column CSV (Run_ID + 3 True + 3 Pred)
        df = pd.read_csv(csv_file, header=None)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        sys.exit(1)
        
    if df.shape[1] != 7:
        print(f"Error: Expected exactly 7 columns, but found {df.shape[1]}.")
        sys.exit(1)

    # Assign headers internally for clean Pandas grouping
    df.columns = ['Run_ID', 'True_S', 'True_D', 'True_W', 'Pred_S', 'Pred_D', 'Pred_W']
    
    run_ids = df['Run_ID'].unique()
    print(f"\nFound {len(run_ids)} independent runs in {csv_file}.")
    
    mse_scores = []
    ndcg_macro_scores = []
    
    playlists = ['Study', 'Drive', 'Workout']
    ndcg_p_scores = {p: [] for p in playlists}
    
    for run in run_ids:
        run_data = df[df['Run_ID'] == run]
        
        y_true = run_data[['True_S', 'True_D', 'True_W']].values
        y_pred = run_data[['Pred_S', 'Pred_D', 'Pred_W']].values
        
        # Calculate unweighted MSE
        run_mse = mean_squared_error(y_true, y_pred)
        mse_scores.append(run_mse)
        
        # Calculate Macro NDCG
        run_ndcg = ndcg_score(y_true.T, y_pred.T)
        ndcg_macro_scores.append(run_ndcg)
        
        # Calculate NDCG per playlist
        for i, playlist in enumerate(playlists):
            true_p = y_true[:, i].reshape(1, -1)
            pred_p = y_pred[:, i].reshape(1, -1)
            
            if np.sum(true_p) == 0:
                ndcg_p_scores[playlist].append(np.nan)
            else:
                ndcg_p_scores[playlist].append(ndcg_score(true_p, pred_p))

    # Compute Statistics
    mean_mse, ci_mse = calculate_95_ci(mse_scores)
    mean_macro_ndcg, ci_macro_ndcg = calculate_95_ci(ndcg_macro_scores)
    
    print("\n--- Final 95% Confidence Intervals ---")
    print(f"MSE:        {mean_mse:.4f} ± {ci_mse:.4f}")
    print(f"Macro NDCG: {mean_macro_ndcg:.4f} ± {ci_macro_ndcg:.4f}")
    
    print("\n--- NDCG Breakdown by Playlist (95% CI) ---")
    for playlist in playlists:
        mean_p, ci_p = calculate_95_ci(ndcg_p_scores[playlist])
        if np.isnan(mean_p):
            print(f"  {playlist:7s}: N/A (No positive true labels)")
        else:
            print(f"  {playlist:7s}: {mean_p:.4f} ± {ci_p:.4f}")
    
    print("\n--- Per-Run Breakdown ---")
    for i in range(len(run_ids)):
        print(f"Run {i+1:02d} | MSE: {mse_scores[i]:.4f} | Macro NDCG: {ndcg_macro_scores[i]:.4f}")
        for playlist in playlists:
            val = ndcg_p_scores[playlist][i]
            val_str = f"{val:.4f}" if not np.isnan(val) else "N/A"
            print(f"          > {playlist:7s}: {val_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CI for MSE and per-playlist NDCG across multiple runs.")
    parser.add_argument("csv_file", type=str, help="Path to the multi-run predictions CSV file.")
    
    args = parser.parse_args()
    evaluate_predictions(args.csv_file)