import os
from glob import glob
import torch
import torch.nn.functional as F
import librosa
import pandas as pd
from muq import MuQMuLan

# Config
AUDIO_DIR = "songs"
OUTPUT_CSV =  os.path.join("results", "knn_best_predictions.csv")
BATCH_SIZE = 8
PLAYLIST_TEXTS = ["study", "drive", "workout"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]

        wavs = [load_audio(f) for f in batch_files]
        wav_batch = pad_batch(wavs).to(DEVICE)

        with torch.no_grad():
            audio_emb = model(wavs=wav_batch)

        audio_emb = F.normalize(audio_emb, dim=-1)

        sim = audio_emb @ text_emb.T  # (batch, 3)

        for j, file in enumerate(batch_files):
            scores = sim[j].cpu().numpy()

            row = {
                "study": float(scores[0]),
                "drive": float(scores[1]),
                "workout": float(scores[2]),
            }

            all_results.append(row)

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")
    # Give a few samples for testing purposes
    print("\nSAMPLE OUTPUT")

    for _, row in df.head(5).iterrows():
        print(f"\n{row['file']}")
        print(f"study   : {row['study']:.3f}")
        print(f"drive : {row['drive']:.3f}")
        print(f"workout   : {row['workout']:.3f}")

if __name__ == "__main__":
    main()