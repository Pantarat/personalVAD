#!/usr/bin/env python3
"""Plot sample d-vectors from audio files.

Usage (from repo root):
  python src/AE_test/tests/test_dvector_plot.py
"""

import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from resemblyzer import VoiceEncoder

# =====================
# SETTINGS (edit these)
# =====================

LIBRISPEECH_ROOT = "data/LibriSpeech"
LIBRISPEECH_SUBSET = "dev-clean"  # e.g., dev-clean, test-clean, train-other-500
N_SAMPLES = 4
DURATION_SEC = 1.6  # None = use full file
WINDOW_SEC = 1.6
FORCE_CPU = False
RANDOM_SEED = 42

OUTPUT_PLOT_DIR = "src/AE_test/test_outputs/dvector_plots"


def _load_audio(path, sr=16000, duration_sec=None):
    """Load mono audio; optionally trim/pad to duration."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    if duration_sec is None:
        return audio
    target_len = int(sr * duration_sec)
    if len(audio) > target_len:
        return audio[:target_len]
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode="constant")
    return audio


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    rng = random.Random(RANDOM_SEED)

    device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading VoiceEncoder...")
    encoder = VoiceEncoder(device=device)

    root = Path(LIBRISPEECH_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"LibriSpeech root not found: {root}")

    candidates = list(root.rglob(f"{LIBRISPEECH_SUBSET}/**/*.flac"))
    if not candidates:
        candidates = list(root.rglob(f"{LIBRISPEECH_SUBSET}/**/*.wav"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No audio files found under {root}/{LIBRISPEECH_SUBSET}"
        )

    picks = rng.sample(candidates, k=min(N_SAMPLES, len(candidates)))

    dvectors = []
    labels = []
    for path in picks:
        audio = _load_audio(path, duration_sec=DURATION_SEC)
        dvec = encoder.embed_utterance(audio, rate=2.5)
        dvectors.append(dvec)
        labels.append(path.name)
        print(
            f"{path.name}: min={dvec.min():.4f}, max={dvec.max():.4f}, "
            f"mean={dvec.mean():.4f}, l2={np.linalg.norm(dvec):.4f}"
        )

    dvectors = np.asarray(dvectors, dtype=np.float32)
    flat_vals = dvectors.ravel()
    print(
        "\nOverall d-vector value range: "
        f"min={flat_vals.min():.4f}, max={flat_vals.max():.4f}, "
        f"mean={flat_vals.mean():.4f}, std={flat_vals.std():.4f}"
    )

    out_dir = Path(OUTPUT_PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot components for each sampled d-vector
    fig = plt.figure(figsize=(14, 6))
    dims = np.arange(dvectors.shape[1])
    for idx, dvec in enumerate(dvectors):
        plt.plot(dims, dvec, linewidth=1.0, alpha=0.8, label=labels[idx])
    plt.title("Sample d-vector components")
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, frameon=False)
    comp_path = out_dir / "dvector_components.png"
    plt.tight_layout()
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {comp_path}")

    # Plot histogram of all d-vector values
    fig = plt.figure(figsize=(8, 5))
    plt.hist(flat_vals, bins=80, color="steelblue", alpha=0.8)
    plt.title("D-vector value distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    hist_path = out_dir / "dvector_value_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hist_path}")

    # Plot cosine similarity matrix
    n = len(dvectors)
    cos_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            cos_mat[i, j] = _cosine_similarity(dvectors[i], dvectors[j])

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cos_mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(np.arange(n), [f"{i}" for i in range(n)], rotation=0)
    plt.yticks(np.arange(n), [f"{i}" for i in range(n)])
    plt.title("Pairwise cosine similarity")
    heat_path = out_dir / "dvector_cosine_heatmap.png"
    plt.tight_layout()
    plt.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {heat_path}")


if __name__ == "__main__":
    main()
