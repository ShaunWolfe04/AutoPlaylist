import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

# Config
TRAIN_EMB_PATH = "train_embeddings.npy"
TEST_EMB_PATH = "test_embeddings.npy"
TRAIN_LABELS_PATH = "train_labels.npy"
TEST_LABELS_PATH = "test_labels.npy"
PLAYLIST_NAMES = ["workout", "drive", "study"]
K_VALUES = [3, 5, 7, 10, 13]

# Import data from np arrays
X_train = np.load(TRAIN_EMB_PATH)
X_test  = np.load(TEST_EMB_PATH)

Y_train = np.load(TRAIN_LABELS_PATH)
Y_test  = np.load(TEST_LABELS_PATH)

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

results = []

for K in K_VALUES:
    print(f"\nRunning kNN Regression with K = {K}")

    Y_pred = knn_predict(X_train, Y_train, X_test, K)

    mse = mean_squared_error(Y_test, Y_pred)
    bce = binary_cross_entropy(Y_test, Y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"BCE: {bce:.4f}")

    results.append((K, mse, bce))

    # Give some samples for testing
    print("\nSample Predictions")

    for i in range(min(5, len(Y_pred))):
        print(f"\nSample {i}")

        for j, name in enumerate(PLAYLIST_NAMES):
            score = Y_pred[i][j]

            if score >= 0.67:
                label = "YES"
            elif score >= 0.33:
                label = "MAYBE"
            else:
                label = "NO"

            print(f"{name:10s} score={score:.3f} → {label}")


# Find the best k
best = min(results, key=lambda x: x[2])  # minimize BCE

print("\nBEST K")
print(f"K = {best[0]}")
print(f"MSE = {best[1]:.4f}")
print(f"BCE = {best[2]:.4f}")