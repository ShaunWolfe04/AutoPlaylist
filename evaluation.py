import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, ndcg_score
import argparse
import sys

def evaluate_predictions(csv_file):
    try:
        # Load the CSV without expecting headers
        df = pd.read_csv(csv_file, header=None)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        sys.exit(1)
        
    # Verify the file structure
    if df.shape[1] != 6:
        print(f"Error: Expected exactly 6 columns, but found {df.shape[1]}.")
        sys.exit(1)
        
    # Extract true labels (columns 0, 1, 2) and predicted probabilities (columns 3, 4, 5)
    y_true = df.iloc[:, 0:3].values
    y_pred = df.iloc[:, 3:6].values
    
    # 1. Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # 2. Normalized Discounted Cumulative Gain (NDCG)
    # Transposed (.T) to evaluate the ranking of songs for each specific playlist
    macro_ndcg = ndcg_score(y_true.T, y_pred.T)
    
    print(f"\n--- Results for {csv_file} ---")
    print(f"Samples Evaluated: {len(df)}")
    print(f"Overall MSE:       {mse:.4f}")
    print(f"Macro NDCG:        {macro_ndcg:.4f}\n")
    
    # Calculate NDCG per playlist
    playlists = ['Study', 'Drive', 'Workout']
    print("NDCG Breakdown by Playlist:")
    for i, playlist in enumerate(playlists):
        true_p = y_true[:, i].reshape(1, -1)
        pred_p = y_pred[:, i].reshape(1, -1)
        
        # Guard against undefined NDCG if a fold has no positive labels
        if np.sum(true_p) == 0:
            print(f"  {playlist:7s}: N/A (No positive true labels in test set)")
        else:
            p_ndcg = ndcg_score(true_p, pred_p)
            print(f"  {playlist:7s}: {p_ndcg:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MSE and NDCG from a headerless 6-column predictions CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the predictions CSV file (e.g., cnn_predictions.csv)")
    
    args = parser.parse_args()
    evaluate_predictions(args.csv_file)