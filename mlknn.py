import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

# Config
TRAIN_EMB_PATH = "train_embeddings_all_pools.npy"
TEST_EMB_PATH = "test_embeddings_all_pools.npy"
TRAIN_LABELS_PATH = "train_labels_all_pools.npy"
TEST_LABELS_PATH = "test_labels_all_pools.npy"
PLAYLIST_NAMES = ["study", "drive", "workout"]
K_VALUES = [3, 5, 7, 10, 13, 15, 17]
os.makedirs("results", exist_ok=True)
OUTPUT_TO = os.path.join("results", "knn_best_predictions.csv")

# Import data from np arrays
X_train = np.load(TRAIN_EMB_PATH)
X_test  = np.load(TEST_EMB_PATH)

Y_train = np.load(TRAIN_LABELS_PATH)
Y_test  = np.load(TEST_LABELS_PATH)

# Reshape train and test sets to work with knn
X_train = X_train.reshape(X_train.shape[0], -1)
X_test  = X_test.reshape(X_test.shape[0], -1)


#print(Y_train)
print(f"Train embeddings: {X_train.shape}")
print(f"Test embeddings:  {X_test.shape}")
print(f"Train labels:     {Y_train.shape}")
print(f"Test labels:      {Y_test.shape}")

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

def knn_predict(X_train, Y_train, X_test, k):
    preds = []

    for x in X_test:
        # L2 distance
        dists = np.linalg.norm(X_train - x, axis=1)

        # nearest neighbors
        nn_idx = np.argsort(dists)[:k]

        # average labels
        pred = np.mean(Y_train[nn_idx], axis=0)

        preds.append(pred)

    return np.array(preds)

# Keep track of best k for csv file saving
results = []
best = None
best_preds = None
best_k = None
best_score = float("inf")  # minimizing BCE

for K in K_VALUES:
    print(f"\nRunning kNN Regression with K = {K}")

    Y_pred = knn_predict(X_train, Y_train, X_test, K)

    mse = mean_squared_error(Y_test, Y_pred)
    bce = binary_cross_entropy(Y_test, Y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"BCE: {bce:.4f}")

    results.append((K, mse, bce))
    if bce < best_score:
        best_score = bce
        best_k = K
        best_preds = Y_pred


print("\nBEST K SELECTED")
print(f"K = {best_k}")
print(f"BCE = {best_score:.4f}")

rows = []
for i in range(len(best_preds)):
    rows.append({
        "study_label": float(Y_test[i][0]),
        "drive_label": float(Y_test[i][1]),
        "workout_label": float(Y_test[i][2]),
        "study_pred": float(best_preds[i][0]),
        "drive_pred": float(best_preds[i][1]),
        "workout_pred": float(best_preds[i][2]),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_TO, index=False, header=False)

print(f"\nSaved best k kNN predictions to {OUTPUT_TO}")

print("\nSample Predictions of best k")

# Give some sample outputs from the 
for i in range(min(5, len(best_preds))):
    print(f"\nSample {i}")

    for j, name in enumerate(PLAYLIST_NAMES):
        score = best_preds[i][j]

        if score >= 0.67:
            label = "YES"
        elif score >= 0.33:
            label = "MAYBE"
        else:
            label = "NO"

        print(f"{name:10s} score={score:.3f} → {label}")