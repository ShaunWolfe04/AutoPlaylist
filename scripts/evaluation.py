import os
import glob
import numpy as np
import pandas as pd

# Config
PRED_DIR = "results"
TEST_LABELS = "test_labels.npy" 
K = 3  # NDCG@K (study/drive/workout)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def dcg(relevances):
    return np.sum([
        rel / np.log2(i + 2) for i, rel in enumerate(relevances)
    ])


def ndcg(y_true, y_pred):
    ndcg_scores = []

    for i in range(len(y_true)):
        true_rels = y_true[i]
        preds = y_pred[i]

        # ideal order
        ideal_order = np.argsort(-true_rels)
        ideal_dcg = dcg(true_rels[ideal_order][:K])

        # predicted order
        pred_order = np.argsort(-preds)
        pred_dcg = dcg(true_rels[pred_order][:K])

        ndcg_scores.append(pred_dcg / (ideal_dcg + 1e-8))
    return np.mean(ndcg_scores)


# Load test set labels
test_labels = np.load(TEST_LABELS)

# assume columns: study, drive, workout
y_true = test_labels[:, :]


# Grab all CSV Files in results
csv_files = glob.glob(os.path.join(PRED_DIR, "*.csv"))

results = []
for file in csv_files:
    pred = pd.read_csv(file)
    y_pred = pred[["study", "drive", "workout"]].values

    if y_pred.shape != y_true.shape:
        print(f"Skipping {file} (shape mismatch)")
        continue

    mse_score = mse(y_true, y_pred)
    ndcg_score = ndcg(y_true, y_pred)

    results.append({
        "file": os.path.basename(file),
        "mse": mse_score,
        "ndcg": ndcg_score
    })

# Print out results 
results_df = pd.DataFrame(results)

print("\nResults")
print(results_df)