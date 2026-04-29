# Audiorium — Music Recommendation Engine
# Implements the full Audiorium flow:
#   Search path (blue):  text prompt  → CLAP text-to-audio   → Top N → Rule-Based Ordering → recommendations
#   Search path (blue):  image upload → OpenCLIP image-to-audio → Top N → Rule-Based Ordering → top song
#   Queue  path (red):   song selected → CLAP audio-to-audio → Top N → Rule-Based Ordering → auto-queue (loops)
#
# pip install torch torchaudio transformers faiss-cpu spotipy librosa soundfile requests open_clip_torch Pillow

import os
import glob
import pickle
import numpy as np
import torch
import librosa
import faiss
from PIL import Image
from transformers import AutoProcessor, ClapModel
import open_clip
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# ── Config ────────────────────────────────────────────────────────────────────

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()  # fallback for interactive / Colab use

AUDIO_FOLDER = os.path.join(BASE_DIR, "mp3dataset")
INDEX_PATH   = os.path.join(BASE_DIR, "clap_music_index.faiss")
IDS_PATH     = os.path.join(BASE_DIR, "ids.pkl")

SPOTIFY_ID     = "################################" # enter spotipy API credentials here
SPOTIFY_SECRET = "################################" # enter spotipy API credentials here

# Music mood/genre labels used for OpenCLIP zero-shot image classification
MUSIC_PROMPTS = [
    "upbeat energetic music",
    "melancholic sad music",
    "aggressive heavy rock music",
    "calm relaxing ambient music",
    "romantic love music",
    "dance electronic music",
    "acoustic folk music",
    "smooth jazz music",
    "classical orchestral music",
    "hip hop urban music",
    "dramatic cinematic music",
    "dark moody atmospheric music",
    "happy cheerful pop music",
    "intense workout music",
    "dreamy ethereal music",
]

# ── Device ────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ── CLAP — text-to-audio & audio-to-audio ─────────────────────────────────────

clap_processor = AutoProcessor.from_pretrained("laion/larger_clap_music_and_speech")
clap_model     = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").to(device)
clap_model.eval()
print("CLAP loaded")

# ── OpenCLIP — image-to-audio ──────────────────────────────────────────────────

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model     = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
print("OpenCLIP loaded")

# ── Spotify (optional — only needed for download_preview) ─────────────────────

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

def embed_audio(path):
    """CLAP audio encoder — maps an audio file into the CLAP latent space."""
    wav, _ = librosa.load(path, sr=48000, mono=True)
    inputs = clap_processor(audios=wav, return_tensors="pt", sampling_rate=48000).to(device)
    with torch.no_grad():
        feats = clap_model.get_audio_features(**inputs)
    return feats.cpu().numpy().squeeze()

def embed_text(texts: list):
    """CLAP text encoder — maps a list of strings into the CLAP latent space."""
    inputs = clap_processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = clap_model.get_text_features(**inputs)
    return feats.cpu().numpy()

def embed_image(image_path):
    """
    OpenCLIP image-to-audio:
    1. Encode the image with OpenCLIP's vision encoder.
    2. Zero-shot classify against MUSIC_PROMPTS to find the best mood label.
    3. Encode that label with CLAP to enter the audio latent space.
    Returns: (clap_embedding, matched_label)
    """
    image       = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text_tokens = clip_tokenizer(MUSIC_PROMPTS).to(device)

    with torch.no_grad():
        img_feats  = clip_model.encode_image(image)
        txt_feats  = clip_model.encode_text(text_tokens)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        scores     = (img_feats @ txt_feats.T).softmax(dim=-1).squeeze()

    best_label = MUSIC_PROMPTS[scores.argmax().item()]
    confidence = scores.max().item()
    print(f"  Image → '{best_label}' ({confidence:.1%} confidence)")

    return embed_text([best_label])[0], best_label

# ── Rule-Based Ordering ───────────────────────────────────────────────────────

def rule_based_ordering(distances, indices, k=5, pool_factor=3):
    """
    Retrieve a wider candidate pool (top k*pool_factor hits), then draw k
    tracks via score-weighted random sampling. This adds slight randomization
    so results aren't always identical top-k hits, promoting equitable
    visibility of diverse and independent content.
    """
    pool      = min(len(ids), k * pool_factor)
    pool_dist = distances[:pool]
    pool_idx  = indices[:pool]

    weights   = np.exp(pool_dist - pool_dist.max())
    weights  /= weights.sum()
    chosen    = np.random.choice(pool, size=min(k, pool), replace=False, p=weights)
    chosen    = chosen[np.argsort(-pool_dist[chosen])]  # re-sort by score

    return [(ids[pool_idx[i]], float(pool_dist[i])) for i in chosen]

# ── FAISS index ───────────────────────────────────────────────────────────────

def build_index(audio_folder=AUDIO_FOLDER):
    paths = (
        glob.glob(os.path.join(audio_folder, "**", "*.mp3"), recursive=True) +
        glob.glob(os.path.join(audio_folder, "**", "*.wav"), recursive=True)
    )
    print(f"Found {len(paths)} audio files — embedding now...")

    embeddings, ids_list = [], []
    for i, path in enumerate(paths):
        try:
            vec = embed_audio(path)
            embeddings.append(vec)
            ids_list.append(os.path.basename(path))
            print(f"  [{i+1}/{len(paths)}] {os.path.basename(path)}")
        except Exception as e:
            print(f"  Skipped {os.path.basename(path)}: {e}")

    emb = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    faiss.write_index(idx, INDEX_PATH)
    with open(IDS_PATH, "wb") as f:
        pickle.dump(ids_list, f)
    print(f"Index saved: {idx.ntotal} tracks → {INDEX_PATH}")
    return idx, ids_list

def load_index():
    idx = faiss.read_index(INDEX_PATH)
    with open(IDS_PATH, "rb") as f:
        ids_list = pickle.load(f)
    print(f"Index loaded: {idx.ntotal} tracks")
    return idx, ids_list

# ── Search path (blue arrows) ─────────────────────────────────────────────────

def search_by_text(prompt, k=5):
    """
    User Search Prompt
      → CLAP text-to-audio
      → Top N Songs (Track ID + Accuracy Score)
      → Rule-Based Ordering
      → List of Song Recommendations
    """
    q = embed_text([prompt])[0].astype("float32")
    faiss.normalize_L2(q.reshape(1, -1))
    distances, indices = index.search(q.reshape(1, -1), min(len(ids), k * 3))
    return rule_based_ordering(distances[0], indices[0], k=k)

def search_by_image(image_path, k=5):
    """
    User Uploaded Image
      → OpenCLIP image-to-audio
      → Top N Songs (Track ID + Accuracy Score)
      → Rule-Based Ordering
      → Top Song (+ list for further queue building)
    """
    q, label = embed_image(image_path)
    q = q.astype("float32")
    faiss.normalize_L2(q.reshape(1, -1))
    distances, indices = index.search(q.reshape(1, -1), min(len(ids), k * 3))
    results = rule_based_ordering(distances[0], indices[0], k=k)
    return results, label

# ── Queue path (red arrows) ───────────────────────────────────────────────────

def build_queue(seed, queue_length=10):
    """
    User Selects a Song
      → CLAP audio-to-audio
      → Top N Songs (Track ID + Accuracy Score)
      → Rule-Based Ordering
      → Automatic Queue of Songs  (loops back to select next song and repeat)

    seed: track filename (e.g. 'Plain.mp3') or full path
    """
    if not os.path.isfile(seed):
        matches = glob.glob(os.path.join(AUDIO_FOLDER, "**", seed), recursive=True)
        if not matches:
            raise FileNotFoundError(f"Could not find: {seed}")
        seed = matches[0]

    print(f"Queue seed: {os.path.basename(seed)}")
    queue   = []
    visited = {os.path.basename(seed)}
    current = seed

    while len(queue) < queue_length:
        q = embed_audio(current).astype("float32")
        faiss.normalize_L2(q.reshape(1, -1))
        distances, indices = index.search(q.reshape(1, -1), min(len(ids), 18))
        candidates = rule_based_ordering(distances[0], indices[0], k=6)

        next_track = None
        for track, score in candidates:
            if track not in visited:
                next_track = (track, score)
                visited.add(track)
                break

        if next_track is None:
            break

        queue.append(next_track)

        # Resolve to full path for the next audio-to-audio iteration
        matches = glob.glob(os.path.join(AUDIO_FOLDER, "**", next_track[0]), recursive=True)
        if matches:
            current = matches[0]

    return queue

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
        index, ids = load_index()
    else:
        index, ids = build_index()

    # ── Search path: text
    print("\n── Text Search ─────────────────────────────────────")
    for track, score in search_by_text("chill hip hop song with fat 808s"):
        print(f"  {score:.4f}  {track}")

    # ── Search path: image  (point to any image file)
    # print("\n── Image Search ────────────────────────────────────")
    # results, label = search_by_image("assets/glowing_whale.HEIC")
    # print(f"  Mood: '{label}'")
    # for track, score in results:
    #     print(f"  {score:.4f}  {track}")

    # ── Queue path: select a song → auto-queue
    print("\n── Auto Queue ──────────────────────────────────────")
    seed = glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3"))[0]
    print(f"  Seed: {os.path.basename(seed)}")
    for track, score in build_queue(seed, queue_length=8):
        print(f"  {score:.4f}  {track}")

    # ── Full flow: image → top song → queue
    # print("\n── Image → Queue ───────────────────────────────────")
    # results, label = search_by_image("path/to/your/photo.jpg")
    # top_song = results[0][0]
    # print(f"  Image mood: '{label}' → seed: {top_song}")
    # for track, score in build_queue(top_song, queue_length=8):
    #     print(f"  {score:.4f}  {track}")
