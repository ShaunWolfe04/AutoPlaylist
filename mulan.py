import os
from glob import glob
import torch
import torch.nn.functional as F
import librosa
import pandas as pd
from muq import MuQMuLan

# Config
AUDIO_DIR = "songs"
OUTPUT_CSV = "mulan_predictions.csv"
BATCH_SIZE = 8
PLAYLIST_TEXTS = [ # TODO these probably need to be more descriptive for mulan to have better context
    "drive",
    "workout",
    "study"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Helper functions
def load_model():
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval().to(DEVICE)
    return model


def load_audio(path):
    wav, _ = librosa.load(path, sr=24000)
    return torch.tensor(wav, dtype=torch.float32)


def pad_batch(wavs):
    max_len = max(w.shape[0] for w in wavs)

    padded = []
    for w in wavs:
        if w.shape[0] < max_len:
            pad = torch.zeros(max_len - w.shape[0])
            w = torch.cat([w, pad])
        padded.append(w)

    return torch.stack(padded)


def rank_playlists(scores, playlist_texts):
    """
    Converts similarity scores into ranked outputs.
    No fake probability thresholds.
    """
    sorted_idx = torch.argsort(scores, descending=True)

    return [
        {
            "playlist": playlist_texts[i],
            "score": float(scores[i]),
            "rank": r + 1
        }
        for r, i in enumerate(sorted_idx)
    ]
    
def main():
    model = load_model()

    files = glob(os.path.join(AUDIO_DIR, "*.wav"))

    if len(files) == 0:
        print("No .wav files found.")
        return

    print(f"Found {len(files)} audio files.")

    # Precompute text embeddings
    with torch.no_grad():
        text_emb = model(texts=PLAYLIST_TEXTS)

    text_emb = F.normalize(text_emb, dim=-1)

    all_results = []

    # Batch processing
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]

        wavs = [load_audio(f) for f in batch_files]
        wav_batch = pad_batch(wavs).to(DEVICE)

        with torch.no_grad():
            audio_emb = model(wavs=wav_batch)

        audio_emb = F.normalize(audio_emb, dim=-1)

        # cosine similarity
        sim = audio_emb @ text_emb.T

        for j, file in enumerate(batch_files):
            scores = sim[j].cpu()

            ranked = rank_playlists(scores, PLAYLIST_TEXTS)

            for r in ranked:
                all_results.append({
                    "file": os.path.basename(file),
                    "playlist": r["playlist"],
                    "score": r["score"],
                    "rank": r["rank"]
                })

    # Save results to csv
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to {OUTPUT_CSV}")

    # print some samples for testing purposes
    print("\n=== SAMPLE OUTPUT ===")

    for file in df["file"].unique()[:5]:
        print(f"\n{file}")
        subset = df[df["file"] == file].sort_values("rank")

        for _, row in subset.iterrows():
            print(f"{row['playlist']:10s} score={row['score']:.3f} rank={row['rank']}")

if __name__ == "__main__":
    main()