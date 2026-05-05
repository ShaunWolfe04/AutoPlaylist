import os
from glob import glob
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from muq import MuQMuLan

# Config
TEST_EMB_PATH = "test_embeddings_mulan.npy"
TEST_LABELS_PATH = "test_labels_mulan.npy"
OUTPUT_CSV = os.path.join("results", "mulan_predictions.csv")
PLAYLIST_TEXTS = [
    "chill lo-fi study music",
    "steady rhythmic driving music",
    "intense high-energy workout music"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval().to(DEVICE)
    return model


def main():
    model = load_model()

    embedds = np.load(TEST_EMB_PATH)
    labels = np.load(TEST_LABELS_PATH)

    emb_tensor = torch.tensor(embedds, dtype=torch.float32).to(DEVICE)

    # Precompute text embeddings
    with torch.no_grad():
        text_emb = model(texts=PLAYLIST_TEXTS)

    text_emb = F.normalize(text_emb, dim=-1)
    # Normalize audio embeddings
    emb_tensor = F.normalize(emb_tensor, dim=-1)

    sim = emb_tensor @ text_emb.T   # (num_songs, num_playlists)
    raw_scores = sim.cpu().numpy()

    # Normalize globally (same as your original)
    global_min = raw_scores.min()
    global_max = raw_scores.max()
    scaled_scores = (raw_scores - global_min) / (global_max - global_min + 1e-8)

    # Save results
    results = []
    for i in range(scaled_scores.shape[0]):
        results.append({
            "study_label": float(labels[i][0]),
            "drive_label": float(labels[i][1]),
            "workout_label": float(labels[i][2]),
            "study_pred": float(scaled_scores[i][0]),
            "drive_pred": float(scaled_scores[i][1]),
            "workout_pred": float(scaled_scores[i][2]),
        })


    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, header=False)

    print(f"\nSaved results to {OUTPUT_CSV}")

    print("\nSAMPLE OUTPUT")
    for _, row in df.head(5).iterrows():
        print(f"  study   : {row['study_pred']:.3f}")
        print(f"  drive   : {row['drive_pred']:.3f}")
        print(f"  workout : {row['workout_pred']:.3f}")


if __name__ == "__main__":
    main()