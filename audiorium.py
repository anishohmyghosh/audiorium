# Music Recommendation using CLAP embeddings
# Run: pip install torch torchaudio transformers faiss-cpu spotipy librosa soundfile requests

import os
import glob
import pickle
import numpy as np
import torch
import librosa
import faiss
from transformers import AutoProcessor, ClapModel
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = os.path.join(SCRIPT_DIR, "mp3dataset")
INDEX_PATH   = os.path.join(SCRIPT_DIR, "clap_music_index.faiss")
IDS_PATH     = os.path.join(SCRIPT_DIR, "ids.pkl")

SPOTIFY_ID     = "9fb5619d84724d66a569470cc8d03ed2"
SPOTIFY_SECRET = "0b21b67af7a74961858724fa729ba902"

# ── Model setup ───────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "laion/larger_clap_music_and_speech"
processor  = AutoProcessor.from_pretrained(model_name)
model      = ClapModel.from_pretrained(model_name).to(device)
model.eval()

# ── Spotify (optional – only needed for download_preview) ────────────────────

sp = Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_ID, client_secret=SPOTIFY_SECRET
))

def download_preview(sp_track_id, out_path):
    import requests
    meta     = sp.track(sp_track_id)
    prev_url = meta["preview_url"]
    if prev_url:
        r = requests.get(prev_url)
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    return False

# ── Embedding helpers ─────────────────────────────────────────────────────────

def load_audio(path, sr=48000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return torch.tensor(wav).unsqueeze(0)  # [1, T]

def embed_audio(path):
    wav    = load_audio(path).squeeze(0).numpy()
    inputs = processor(audios=wav, return_tensors="pt", sampling_rate=48000).to(device)
    with torch.no_grad():
        feats = model.get_audio_features(**inputs)
    return feats.cpu().numpy().squeeze()

def embed_text(texts: list):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return feats.cpu().numpy()

# ── Build / load FAISS index ──────────────────────────────────────────────────

def build_index(audio_folder):
    paths = glob.glob(os.path.join(audio_folder, "**", "*.mp3"), recursive=True)
    paths += glob.glob(os.path.join(audio_folder, "**", "*.wav"), recursive=True)
    print(f"Found {len(paths)} audio files")

    embeddings, ids = [], []
    for idx, path in enumerate(paths):
        try:
            vec = embed_audio(path)
            embeddings.append(vec)
            ids.append(os.path.basename(path))
            print(f"  [{idx+1}/{len(paths)}] {os.path.basename(path)}")
        except Exception as e:
            print(f"  Skipped {os.path.basename(path)}: {e}")

    emb = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(emb)

    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    print(f"Index built with {idx.ntotal} tracks")

    faiss.write_index(idx, INDEX_PATH)
    with open(IDS_PATH, "wb") as f:
        pickle.dump(ids, f)
    print(f"Saved index → {INDEX_PATH}")
    print(f"Saved ids   → {IDS_PATH}")
    return idx, ids

def load_index():
    idx = faiss.read_index(INDEX_PATH)
    with open(IDS_PATH, "rb") as f:
        ids = pickle.load(f)
    print(f"Loaded index with {idx.ntotal} tracks")
    return idx, ids

# ── Recommendation functions ──────────────────────────────────────────────────

def recommend_by_text(prompt, k=5):
    q = embed_text([prompt])[0].astype("float32")
    faiss.normalize_L2(q.reshape(1, -1))
    distances, indices = index.search(q.reshape(1, -1), k)
    return [(ids[i], float(distances[0][j])) for j, i in enumerate(indices[0])]

def recommend_by_audio(path, k=5):
    q = embed_audio(path).astype("float32")
    faiss.normalize_L2(q.reshape(1, -1))
    distances, indices = index.search(q.reshape(1, -1), k)
    return [(ids[i], float(distances[0][j])) for j, i in enumerate(indices[0])]

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
        index, ids = load_index()
    else:
        index, ids = build_index(AUDIO_FOLDER)

    print("\nRecommendations for 'chill hip hop with fat 808s':")
    for track, score in recommend_by_text("chill hip hop song with fat 808s"):
        print(f"  {score:.4f}  {track}")

    print("\nRecommendations for 'female vocals':")
    for track, score in recommend_by_text("female vocals"):
        print(f"  {score:.4f}  {track}")
