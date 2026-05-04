import os
from glob import glob
import torch
import torch.nn.functional as F
import librosa
import pandas as pd
from muq import MuQMuLan
import numpy as np

# Config
AUDIO_DIR = "../songs"
OUTPUT_CSV =  os.path.join("results", "mulan_predictions.csv")
BATCH_SIZE = 8
PLAYLIST_TEXTS = ["chill lo-fi study music", "steady rhythmic driving music", "intense high-energy workout music"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIR = "mulan_embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

def load_model():
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval().to(DEVICE)
    return model


def load_audio(path):
    wav, _ = librosa.load(path, sr=24000)
    return torch.tensor(wav, dtype=torch.float32)


def main():
    model = load_model()

    files = []
    for ext in ("*.mp3", "*.ogg"):
        files.extend(glob(os.path.join(AUDIO_DIR, ext)))

    if len(files) == 0:
        print("No .mp3 files found.")
        return

    print(f"Found {len(files)} audio files.")

    # Precompute text embeddings
    with torch.no_grad():
        text_emb = model(texts=PLAYLIST_TEXTS)

    text_emb = F.normalize(text_emb, dim=-1)

    file_list = []
    raw_scores = []
    for file in files:
        print(f"Processing {file}")

        name = os.path.basename(file)
        emb_path = os.path.join(EMB_DIR, f"{name}.pt")

        # load or compute embedding
        if os.path.exists(emb_path):
            emb = torch.load(emb_path).to(DEVICE)
        else:
            wav = load_audio(file)
            wav = wav.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model(wavs=wav)

            torch.save(emb.squeeze(0).cpu(), emb_path)

        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        sim = emb @ text_emb.T
        scores = sim[0].cpu().numpy()

        raw_scores.append(scores)
        file_list.append(file)
            
    raw_scores = np.array(raw_scores)
    global_min = raw_scores.min()
    global_max = raw_scores.max()
    scaled_scores = (raw_scores - global_min) / (global_max - global_min + 1e-8)

    all_results = []

    for i, file in enumerate(file_list):
        all_results.append({
            "study": float(scaled_scores[i][0]),
            "drive": float(scaled_scores[i][1]),
            "workout": float(scaled_scores[i][2]),
        })
    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")
    # Give a few samples for testing purposes
    print("\nSAMPLE OUTPUT")

    for _, row in df.head(5).iterrows():
        print(f"study   : {row['study']:.3f}")
        print(f"drive : {row['drive']:.3f}")
        print(f"workout   : {row['workout']:.3f}")

if __name__ == "__main__":
    main()