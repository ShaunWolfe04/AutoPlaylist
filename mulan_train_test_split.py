import csv
import numpy as np
import torch
import os

EMBEDDING_SIZE = 512
EMB_DIR = "mulan_embeddings"

np.random.seed(42)

with open("playlist_labels.csv") as f:
    g = csv.reader(f)
    labs = []
    for row in g:
        labs.append([row[0], row[3], row[4], row[5]])

r = np.array(labs)

embedds = []
for row in r:
    filename = row[0]
    path = os.path.join(EMB_DIR, f"{filename}.pt")

    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"failed to load embeddings for file {filename}")
        print(e)
        exit()

    print(filename)

    # handle different save formats
    if isinstance(data, dict):
        emb = data.get("embedding", data)
    else:
        emb = data

    emb = emb.detach().cpu().numpy()

    # ensure shape is (length, 500)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    # keep MuLan space: just mean-pool across time
    emb = np.mean(emb, axis=0) 

    # sanity check
    if emb.shape[0] != EMBEDDING_SIZE:
        print(f"Unexpected embedding size for {filename}: {emb.shape}")
        exit()

    embedds.append(emb)

embedds = np.array(embedds)
print("Final embedding shape:", embedds.shape)
# (num_songs, 500)

def simple_map(x):
    mapp = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
    return mapp[int(x)]

vf = np.vectorize(simple_map)
labels = vf(r[:, 1:])

# Same train test split as other models
indics = np.random.permutation(embedds.shape[0])

test_idx = indics[:31]
train_idx = indics[31:]

train_embedds = embedds[train_idx]
train_labels = labels[train_idx]

test_embedds = embedds[test_idx]
test_labels = labels[test_idx]

# Save files
np.save("train_embeddings_mulan.npy", train_embedds)
np.save("test_embeddings_mulan.npy", test_embedds)
np.save("train_labels_mulan.npy", train_labels)
np.save("test_labels_mulan.npy", test_labels)

print("\nSaved:")
print("train_embeddings_mulan.npy", train_embedds.shape)
print("test_embeddings_mulan.npy", test_embedds.shape)