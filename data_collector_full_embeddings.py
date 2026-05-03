import csv
import numpy as np


EMBEDDING_SIZE = 500

np.random.seed(42)

def extract_features(embeddings):
    #embeddings shape is (length, 500). We will make 1000 with mean and max (in that order)
    mean_features = list(np.mean(embeddings, axis=0))
    max_features = list(np.max(embeddings, axis=0))
    assert embeddings.shape[1] == EMBEDDING_SIZE
    assert len(mean_features) == EMBEDDING_SIZE
    assert len(max_features) == EMBEDDING_SIZE

    comb = mean_features + max_features
    assert(len(comb) == EMBEDDING_SIZE*2)
    return comb
    

#GOAL - generate all_embeddings and all_labels. all_embeddings is num_songs x embeddingdim, and all labels is num_songs x num_playlists

#First, process our labeled data
f = open("playlist_labels.csv")
g = csv.reader(f)
labs = []
for row in g:
    labs.append([row[0], row[3], row[4], row[5]])
r = np.array(labs)

#now, r contains [song file name, label1, label2, .... labeln]

#Using the file names in r, go find the correct embeddings
embedds = []
for row in r:
    filename = row[0]
    try:
        l = np.load(f"vectors/embeddings/{filename}.npz")
    except Exception as e:
        print(f"failed to load embeddings for file {filename}")
        print(e)
        exit()

    #now that we have the file open, pull the embeddings, process them, and add them to an array
    print(filename)
    feats = extract_features(l['embedding'])
    assert len(feats) == EMBEDDING_SIZE * 2
    embedds.append(feats)

embedds = np.array(embedds)
assert embedds.shape[1] == EMBEDDING_SIZE * 2

#map likert labels to decimal
def simple_map(x):
    mapp = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
    return mapp[int(x)]
vf = np.vectorize(simple_map)
labels = vf(r[:, 1:])

#now, embedds is [num items, embedding dim]

#now, output to files


assert labels.shape[0] == embedds.shape[0]

#now, leave out 31 random samples for test
indics = np.random.permutation(embedds.shape[0])

test_idx = indics[:31]
train_idx = indics[31:]

train_embedds = embedds[train_idx]
train_labels = labels[train_idx]

test_embedds = embedds[test_idx]
test_labels = labels[test_idx]



np.save("train_embeddings.npy", train_embedds)
np.save("test_embeddings.npy", test_embedds)
np.save("test_labels.npy", test_labels)
np.save("train_labels.npy", train_labels)
