import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, ndcg_score
import argparse
import sys

def evaluate_predictions(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        sys.exit(1)
        
    # Isolate the Test split for evaluation
    if 'Split' in df.columns:
        test_df = df[df['Split'] == 'Test']
        if test_df.empty:
            print("Warning: No 'Test' split found. Evaluating all rows.")
            test_df = df
    else:
        test_df = df
        
    # Extract the true labels and predicted probabilities
    y_true = test_df[['True_Study', 'True_Drive', 'True_Workout']].values
    y_pred = test_df[['Pred_Study', 'Pred_Drive', 'Pred_Workout']].values
    
    # 1. Mean Squared Error (MSE)
    # Calculates the unweighted average squared difference between predictions and truths
    mse = mean_squared_error(y_true, y_pred)
    
    # 2. Normalized Discounted Cumulative Gain (NDCG)
    # We transpose (.T) the matrices so the shape is (3_playlists, n_songs).
    # This evaluates how perfectly the songs are ranked for each playlist, 
    # rather than how the playlists are ranked for a single song.
    macro_ndcg = ndcg_score(y_true.T, y_pred.T)
    
    print(f"\n--- Results for {csv_file} ---")
    print(f"Samples Evaluated: {len(test_df)}")
    print(f"Overall MSE:       {mse:.4f}")
    print(f"Macro NDCG:        {macro_ndcg:.4f}\n")
    
    # Calculate NDCG per playlist for granular debugging
    playlists = ['Study', 'Drive', 'Workout']
    print("NDCG Breakdown by Playlist:")
    for i, playlist in enumerate(playlists):
        # ndcg_score requires 2D arrays, so we reshape from (n_songs,) to (1, n_songs)
        true_p = y_true[:, i].reshape(1, -1)
        pred_p = y_pred[:, i].reshape(1, -1)
        
        # If a fold happens to have all 0.0s for a playlist, NDCG is undefined.
        if np.sum(true_p) == 0:
            print(f"  {playlist:7s}: N/A (No positive true labels in test set)")
        else:
            p_ndcg = ndcg_score(true_p, pred_p)
            print(f"  {playlist:7s}: {p_ndcg:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MSE and NDCG from a predictions CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the predictions CSV file (e.g., cnn_predictions.csv)")
    
    args = parser.parse_args()
    evaluate_predictions(args.csv_file)