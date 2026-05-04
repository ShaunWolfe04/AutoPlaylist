import csv
import numpy as np
import torch
import os

EMBEDDING_SIZE = 500
np.random.seed(42)

def simple_map(x):
    """Maps the 0-3 Likert scale to 0.0-1.0 probabilities."""
    mapp = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
    return mapp[int(x)]

def main():
    # 1. Read the labels
    try:
        with open("playlist_labels.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            raw_data = [row for row in reader if row]
    except FileNotFoundError:
        print("Error: playlist_labels.csv not found.")
        return

    dataset = []

    # 2. Process each song
    for row in raw_data:
        filename = row[0]
        # Map labels immediately
        labels = [simple_map(row[3]), simple_map(row[4]), simple_map(row[5])]
        
        try:
            # Load the raw .npz file
            file_path = f"vectors/embeddings/{filename}.npz"
            l = np.load(file_path)
            raw_embedding = l['embedding'] # Shape: (T, 500)
            
            # Verify the embedding size
            assert raw_embedding.shape[1] == EMBEDDING_SIZE, f"Expected 500 features, got {raw_embedding.shape[1]}"
            
            # Convert to PyTorch Tensors
            # CNNs in PyTorch expect the channel dimension before the sequence dimension: (Channels, Sequence Length)
            # So we transpose the (T, 500) matrix to (500, T)
            tensor_embedding = torch.tensor(raw_embedding, dtype=torch.float32).transpose(0, 1)
            tensor_labels = torch.tensor(labels, dtype=torch.float32)
            
            # Append as a tuple: (X, Y)
            dataset.append((tensor_embedding, tensor_labels))
            print(f"Processed {filename} | Shape: {tensor_embedding.shape}")
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not dataset:
        print("Dataset is empty. Check your file paths.")
        return

    # 3. Train / Test Split
    print(f"\nSuccessfully processed {len(dataset)} songs.")
    
    # Shuffle the dataset
    np.random.shuffle(dataset)
    
    # Leave out 31 random samples for test (as in your original script)
    test_dataset = dataset[:31]
    train_dataset = dataset[31:]
    
    # 4. Save using PyTorch
    torch.save(train_dataset, "train_dataset.pt")
    torch.save(test_dataset, "test_dataset.pt")
    
    print(f"Saved {len(train_dataset)} training samples to train_dataset.pt")
    print(f"Saved {len(test_dataset)} testing samples to test_dataset.pt")

if __name__ == "__main__":
    main()