import os
from glob import glob
import torch
import librosa
from muq import MuQMuLan

# Config
AUDIO_DIR = "../songs"
EMB_DIR = "mulan_embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        print("No audio files found.")
        return

    print(f"Found {len(files)} audio files.")

    for file in files:
        name = os.path.basename(file)
        emb_path = os.path.join(EMB_DIR, f"{name}.pt")

        if os.path.exists(emb_path):
            print(f"Skipping (exists): {name}")
            continue

        print(f"Processing: {name}")

        wav = load_audio(file)
        wav = wav.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(wavs=wav)

        # save WITHOUT batch dimension
        torch.save(emb.squeeze(0).cpu(), emb_path)

    print("\nDone generating embeddings.")


if __name__ == "__main__":
    main()