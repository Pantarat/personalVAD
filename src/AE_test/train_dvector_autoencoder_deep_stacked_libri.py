#!/usr/bin/env python3
"""
Train a Deep Stacked d-vector autoencoder using LibriSpeech data.

Pipeline:
1) Sample clean LibriSpeech utterances.
2) Add MUSAN speech babble at random SNR to create noisy inputs.
3) Extract paired (noisy_dvector, clean_dvector) with Resemblyzer.
4) Train DeepStackedDvectorAutoencoder to reconstruct clean d-vectors.
5) Save checkpoints, plots, and config in loader-compatible format.
"""

import json
import pickle
import random
import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.optim as optim
from resemblyzer import VoiceEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchsummary import summary

from autoencoder_utils import DeepStackedDvectorAutoencoder, load_autoencoder
from readable_dvector_cache import (
    checkpoint_dir_for_config as _readable_checkpoint_dir_for_config,
    extract_with_cache as _readable_extract_with_cache,
    normalize_for_hash as _normalize_for_hash,
)

# Suppress FutureWarning from resemblyzer library (librosa positional args deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="resemblyzer")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Optional base model initialization (set to None to train deep stacked from scratch)
# BASE_IDENTITY_MODEL_DIR = "test_outputs/models/dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_v2_12-4-26"
BASE_IDENTITY_MODEL_DIR = None
INIT_FIRST_DAE_FROM_BASE = False

# Optional greedy-layerwise initialization for first block
GREEDY_INIT_MODEL_DIR = 'test_outputs/models/dvector_ae_greedy_layerwise_8_2-5-26'
INIT_FIRST_BLOCK_FROM_GREEDY = True

# LibriSpeech source
LIBRISPEECH_ROOT = "../../data/LibriSpeech"
LIBRISPEECH_SUBSETS = [
    # "train-clean-100",
    # "train-clean-360",
    "train-other-500",
]
N_LIBRI_UTTERANCES = 40000  # -1 uses all discovered files
MIN_UTTERANCE_SEC = 1.6
MAX_UTTERANCE_SEC = None  # Set None to disable max duration filter
CHUNK_DURATION_SEC = 1.6  # Duration of chunks to extract from LibriSpeech (matches encoder window)
MAX_CHUNKS_PER_UTTERANCE = 5  # Set >1 to extract multiple chunks from longer utterances (with hop)

# MUSAN speech root for babble augmentation
MUSAN_SPEECH_NOISE_ROOT = "../../kaldi/egs/pvad/musan/musan_speech_train"
MUSAN_BABBLE_SNR_RANGE_DB = (20.0, 17.0, 15.0, 13.0)
INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS = True

# Preview augmented audio exports for listening/debugging
SAVE_AUGMENTED_AUDIO_PREVIEW = True
N_AUGMENTED_AUDIO_PREVIEW = 10
AUGMENTED_AUDIO_PREVIEW_DIR = "test_outputs/debug/deep_stacked_libri_augmented_preview"

# Output model directory
MODEL_SAVE_DIR = "test_outputs/models/dvector_ae_deep_stacked_greedy_babble_11_4-5-26"

# Deep stacked architecture
DEEP_STACKED_NUM_DAES = 2
  # [] = 256->256; [128] = 256->128->256; [256,128,256] = 256->256->128->256, etc.
DEEP_STACKED_FIRST_BLOCK_HIDDEN_DIMS = [512]
DEEP_STACKED_REFINEMENT_HIDDEN_DIMS = [(512)]
# DEEP_STACKED_REFINEMENT_HIDDEN_DIMS = [(512,384,256), (512,384,256)]

# Fallback architecture values (used when BASE_IDENTITY_MODEL_DIR is None)
FALLBACK_INPUT_DIM = 256
FALLBACK_HIDDEN_DIMS = []
# FALLBACK_HIDDEN_DIMS = [192,128,192]
FALLBACK_DROPOUT_RATE = 0.1
FALLBACK_NORM_TYPE = "batchnorm"
FALLBACK_ACTIVATION_TYPE = "tanh"
FALLBACK_USE_RESIDUAL = False
FALLBACK_RESIDUAL_SCALE_INIT = 0.5

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 200
VALIDATION_SPLIT = 0.05
TEST_SPLIT = 0.0
EARLY_STOPPING_PATIENCE = 10

# Mixed reconstruction loss: total = MSE_WEIGHT * MSE + COSINE_WEIGHT * (1 - cosine)
LOSS_MSE_WEIGHT = 0.3
LOSS_COSINE_WEIGHT = 0.7
LOSS_COSINE_EPS = 1e-8

# Audio settings
SAMPLE_RATE = 16000

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed
RANDOM_SEED = 42

# D-vector extraction cache
DVECTOR_CACHE_ENABLED = True
DVECTOR_CACHE_DIR = "test_outputs/dvector_cache"
DVECTOR_CACHE_VERSION = 1
EXTRACTION_CACHE_NAME = "deep_stacked_libri_babble_pairs"
CHECKPOINT_INTERVAL = 500  # Save partial progress every N samples during extraction


def _checkpoint_dir_for_config(cache_dir, cache_name, cache_config):
    return _readable_checkpoint_dir_for_config(
        cache_dir=cache_dir,
        cache_name=cache_name,
        cache_config=cache_config,
        cache_version=DVECTOR_CACHE_VERSION,
    )


def _extract_with_cache(cache_dir, cache_name, cache_config, extractor_fn):
    if not DVECTOR_CACHE_ENABLED:
        return extractor_fn()
    return _readable_extract_with_cache(
        cache_dir=cache_dir,
        cache_name=cache_name,
        cache_config=cache_config,
        extractor_fn=extractor_fn,
        cache_version=DVECTOR_CACHE_VERSION,
    )

class NoisyCleanDvectorDataset(Dataset):
    """Dataset for mapping noisy d-vectors to clean d-vectors."""

    def __init__(self, noisy_dvectors, clean_dvectors, indices):
        self.noisy_dvectors = noisy_dvectors
        self.clean_dvectors = clean_dvectors
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        noisy = torch.from_numpy(self.noisy_dvectors[i]).float()
        clean = torch.from_numpy(self.clean_dvectors[i]).float()
        return noisy, clean


def _load_extraction_checkpoint(checkpoint_dir):
    """Load partial extraction progress from checkpoint."""
    checkpoint_file = checkpoint_dir / "checkpoint.pkl"
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"[checkpoint] Loaded from {checkpoint_file}")
        print(f"  Samples so far: {checkpoint['sample_count']}")
        print(f"  Files processed: {checkpoint['last_file_index']} / {checkpoint['total_files']}")
        return checkpoint
    except Exception as e:
        print(f"[checkpoint] Failed to load: {e}. Starting fresh.")
        return None


def _save_extraction_checkpoint(checkpoint_dir, keys, noisy_dvectors, clean_dvectors, metadata_rows,
                               last_file_index, total_files, preview_items):
    """Save partial extraction progress to checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "checkpoint.pkl"
    
    checkpoint = {
        "keys": keys,
        "noisy_dvectors": np.asarray(noisy_dvectors, dtype=np.float32),
        "clean_dvectors": np.asarray(clean_dvectors, dtype=np.float32),
        "metadata_rows": metadata_rows,
        "last_file_index": last_file_index,
        "total_files": total_files,
        "sample_count": len(keys),
        "preview_items": preview_items,
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"[checkpoint] Saved at sample count {len(keys)}, file index {last_file_index}/{total_files}")


def discover_musan_speech_noise_files(speech_noise_root):
    """Discover MUSAN speech-train files from wav/flac and optional wav.scp."""
    speech_noise_root = Path(speech_noise_root)
    if not speech_noise_root.exists():
        return []

    speech_noise_files = set()

    for pattern in ("**/*.wav", "**/*.flac"):
        for file_path in speech_noise_root.glob(pattern):
            if file_path.is_file():
                speech_noise_files.add(file_path.resolve())

    for wav_scp_path in speech_noise_root.glob("**/wav.scp"):
        try:
            with open(wav_scp_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or "|" in line:
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    token = parts[1]
                    candidate_bases = [
                        speech_noise_root,
                        speech_noise_root.parent,
                        speech_noise_root.parent.parent,
                        wav_scp_path.parent,
                        wav_scp_path.parent.parent,
                    ]

                    for ancestor in (speech_noise_root, *speech_noise_root.parents):
                        if ancestor.name == "pvad":
                            candidate_bases.append(ancestor)
                            break

                    candidate_paths = [Path(token)]
                    candidate_paths.extend(base / token for base in candidate_bases)

                    for candidate in candidate_paths:
                        if candidate.exists() and candidate.is_file():
                            speech_noise_files.add(candidate.resolve())
                            break
        except OSError:
            continue

    return sorted(str(p) for p in speech_noise_files)


def discover_librispeech_audio_files(
    librispeech_root,
    subsets,
    n_utterances=-1,
    min_utt_sec=1.6,
    max_utt_sec=None,
    random_seed=42,
):
    """Discover LibriSpeech audio files with optional duration filtering and sampling."""
    root = Path(librispeech_root)
    if not root.exists():
        return []

    candidate_files = []
    for subset in subsets:
        subset_path = root / subset
        if not subset_path.exists():
            print(f"[libri] Subset not found, skipping: {subset_path}")
            continue

        subset_files = []
        for pattern in ("**/*.flac", "**/*.wav"):
            subset_files.extend([p for p in subset_path.glob(pattern) if p.is_file()])
        print(f"[libri] Discovered {len(subset_files)} files in subset {subset}")
        candidate_files.extend(subset_files)

    if len(candidate_files) == 0:
        return []

    filtered = []
    progress = tqdm(candidate_files, desc="filter libri durations", unit="file")
    for path in progress:
        try:
            info = sf.info(str(path))
            dur = float(info.frames) / float(info.samplerate)
        except Exception:
            continue

        if min_utt_sec is not None and dur < float(min_utt_sec):
            continue
        if max_utt_sec is not None and dur > float(max_utt_sec):
            continue
        filtered.append(path)

    rng = random.Random(random_seed)
    if n_utterances > 0 and len(filtered) > n_utterances:
        filtered = rng.sample(filtered, int(n_utterances))

    filtered = sorted(filtered)
    print(f"[libri] Final selected files: {len(filtered)}")
    return filtered


def load_noise_segment(noise_path, target_samples, target_sr=16000, rng=None):
    """Load one noise segment with exact target length."""
    noise, _ = librosa.load(noise_path, sr=target_sr)
    if len(noise) == 0:
        return None

    max_abs = np.max(np.abs(noise))
    if max_abs > 0:
        noise = noise / (max_abs + 1e-8)

    if len(noise) >= target_samples:
        if rng is None:
            start_idx = random.randint(0, len(noise) - target_samples)
        else:
            start_idx = rng.randint(0, len(noise) - target_samples)
        segment = noise[start_idx:start_idx + target_samples]
    else:
        n_repeats = int(np.ceil(target_samples / len(noise)))
        segment = np.tile(noise, n_repeats)[:target_samples]

    return segment


def mix_audio_with_snr(audio_signal, noise_signal, snr_db):
    """Mix signal and noise at requested SNR (signal-to-noise, dB)."""
    power_signal = np.mean(audio_signal ** 2)
    power_noise = np.mean(noise_signal ** 2)

    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(power_signal / (power_noise * snr_linear + 1e-8))

    scaled_noise = scale * noise_signal
    mixed = audio_signal + scaled_noise

    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        limit_scale = 0.95 / max_val
        mixed = mixed * limit_scale
        scaled_noise = scaled_noise * limit_scale

    return mixed, scaled_noise


def extract_noisy_clean_dvector_pairs_from_libri(
    librispeech_root,
    librispeech_subsets,
    musan_speech_noise_root,
    n_utterances=-1,
    min_utt_sec=1.6,
    max_utt_sec=20.0,
    sample_rate=16000,
    device="cuda",
    snr_range_db=(20.0, 17.0, 15.0, 13.0),
    include_noise_only_zero_target_pairs=True,  # Note: now uses silence d-vector, not zero
    random_seed=42,
    preview_limit=10,
    checkpoint_dir=None,
):
    """Build paired d-vectors from LibriSpeech clean utterances and MUSAN babble.
    
    Supports resumable extraction via checkpoint_dir:
    - If checkpoint exists, resumes from last saved position
    - Periodically saves progress to checkpoint (configurable interval)
    - On interruption, partial progress is preserved for resume
    """
    print("\n[extract] Building noisy->clean d-vector pairs from LibriSpeech")
    print(f"  Libri root: {librispeech_root}")
    print(f"  Libri subsets: {', '.join(librispeech_subsets)}")
    print(f"  Target utterances: {n_utterances}")
    print(f"  Duration filter: min={min_utt_sec}, max={max_utt_sec}")
    print(f"  MUSAN speech root: {musan_speech_noise_root}")
    print(f"  SNR values (dB): {snr_range_db}")
    print(f"  Extra pairs enabled (noise->silence): {include_noise_only_zero_target_pairs}")
    print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL} samples")

    libri_files = discover_librispeech_audio_files(
        librispeech_root=librispeech_root,
        subsets=librispeech_subsets,
        n_utterances=n_utterances,
        min_utt_sec=min_utt_sec,
        max_utt_sec=max_utt_sec,
        random_seed=random_seed,
    )
    if len(libri_files) == 0:
        raise RuntimeError("No LibriSpeech files selected for extraction")

    noise_files = discover_musan_speech_noise_files(musan_speech_noise_root)
    if len(noise_files) == 0:
        raise FileNotFoundError(f"No MUSAN speech files found under: {musan_speech_noise_root}")

    if torch.cuda.is_available() and device == "cuda":
        device_obj = torch.device("cuda")
        print("  Encoder device: CUDA")
    else:
        device_obj = torch.device("cpu")
        print("  Encoder device: CPU")

    encoder = VoiceEncoder(device=device_obj)
    rng = random.Random(random_seed)
    snr_values_db = [float(v) for v in snr_range_db]

    keys = []
    noisy_dvectors = []
    clean_dvectors = []
    metadata_rows = []
    preview_items = []
    start_file_index = 0

    # Try to load checkpoint if available
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint = _load_extraction_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            keys = list(checkpoint["keys"])
            noisy_dvectors = [v for v in checkpoint["noisy_dvectors"]]
            clean_dvectors = [v for v in checkpoint["clean_dvectors"]]
            metadata_rows = list(checkpoint["metadata_rows"])
            preview_items = list(checkpoint["preview_items"])
            start_file_index = checkpoint["last_file_index"]
            print(f"[extract] Resuming from checkpoint: {start_file_index} files processed, {len(keys)} samples extracted")

    progress = tqdm(libri_files, desc="extract libri pairs", unit="utt", initial=start_file_index, total=len(libri_files))
    for file_idx, audio_path in enumerate(libri_files):
        # Skip already processed files
        if file_idx < start_file_index:
            progress.update(1)
            continue
        try:
            clean_audio, _ = librosa.load(str(audio_path), sr=sample_rate) 
            if len(clean_audio) == 0: 
                continue
            chunk_size = int(sample_rate * CHUNK_DURATION_SEC)
            hop_size = chunk_size  # no overlap (you can change later)

            chunk_count = 0
            for start in range(0, len(clean_audio) - chunk_size + 1, hop_size):
                if chunk_count >= MAX_CHUNKS_PER_UTTERANCE:
                    break
                clean_chunk = clean_audio[start:start + chunk_size]

                noise_path = rng.choice(noise_files)
                noise_segment = load_noise_segment(
                    noise_path,
                    target_samples=chunk_size,
                    target_sr=sample_rate,
                    rng=rng,
                )
                if noise_segment is None:
                    continue

                snr_db = float(rng.choice(snr_values_db))
                noisy_chunk, scaled_noise = mix_audio_with_snr(clean_chunk, noise_segment, snr_db)

                with torch.no_grad():
                    clean_dvec = encoder.embed_utterance(clean_chunk).astype(np.float32)
                    noisy_dvec = encoder.embed_utterance(noisy_chunk).astype(np.float32)
                    noise_only_dvec = encoder.embed_utterance(scaled_noise).astype(np.float32)
                    
                    # Extract silence d-vector for noise-only target
                    silence_segment = np.zeros(chunk_size, dtype=np.float32)
                    silence_dvec = encoder.embed_utterance(silence_segment).astype(np.float32)

                key = f"{audio_path.stem}_{start}"
                noise_only_key = f"{audio_path.stem}_{start}_noise_only"

                keys.append(key)
                noisy_dvectors.append(noisy_dvec)
                clean_dvectors.append(clean_dvec)

                dur_sec = float(len(clean_audio)) / float(sample_rate)
                metadata_rows.append(
                    {
                        "key": key,
                        "audio_path": str(audio_path),
                        "noise_file": str(noise_path),
                        "snr_db": float(snr_db),
                        "pair_type": "clean_plus_noise",
                        "duration_sec": dur_sec,
                    }
                )

                if include_noise_only_zero_target_pairs:
                    keys.append(noise_only_key)
                    noisy_dvectors.append(noise_only_dvec)
                    clean_dvectors.append(silence_dvec)
                    metadata_rows.append(
                        {
                            "key": noise_only_key,
                            "audio_path": str(audio_path),
                            "noise_file": str(noise_path),
                            "snr_db": float(snr_db),
                            "pair_type": "noise_only_silence_target",
                            "duration_sec": dur_sec,
                        }
                    )

                if len(preview_items) < preview_limit:
                    preview_items.append(
                        {
                            "key": key,
                            "audio": noisy_chunk.astype(np.float32),
                            "sample_rate": sample_rate,
                            "snr_db": float(snr_db),
                            "noise_file": str(noise_path),
                            "source_audio": str(audio_path),
                        }
                    )

                chunk_count += 1
        except Exception:
            progress.update(1)
            continue
        
        progress.update(1)
        
        # Save checkpoint periodically
        if checkpoint_dir is not None and len(keys) > 0 and len(keys) % CHECKPOINT_INTERVAL == 0:
            _save_extraction_checkpoint(
                checkpoint_dir=checkpoint_dir,
                keys=keys,
                noisy_dvectors=noisy_dvectors,
                clean_dvectors=clean_dvectors,
                metadata_rows=metadata_rows,
                last_file_index=file_idx + 1,
                total_files=len(libri_files),
                preview_items=preview_items,
            )

    if len(keys) == 0:
        raise RuntimeError("No noisy->clean d-vector pairs were created")
    
    # Clean up checkpoint after successful completion
    if checkpoint_dir is not None:
        checkpoint_file = Path(checkpoint_dir) / "checkpoint.pkl"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"[checkpoint] Cleaned up after successful extraction")
            except Exception as e:
                print(f"[checkpoint] Warning: could not remove checkpoint: {e}")

    print("\n[extract] Summary")
    print(f"  Total pairs: {len(keys)}")
    return {
        "keys": keys,
        "noisy_dvectors": np.asarray(noisy_dvectors, dtype=np.float32),
        "clean_dvectors": np.asarray(clean_dvectors, dtype=np.float32),
        "metadata": metadata_rows,
        "preview_items": preview_items,
    }


def save_augmented_preview_audio(preview_items, preview_dir):
    """Save a small set of augmented audio files for manual examination."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, item in enumerate(preview_items):
        safe_key = item["key"].replace("/", "_").replace("\\", "_")
        filename = f"augmented_{i:02d}_{safe_key}.wav"
        out_path = preview_dir / filename

        sf.write(out_path, item["audio"], item["sample_rate"])

        manifest.append(
            {
                "file": filename,
                "key": item["key"],
                "snr_db": float(item["snr_db"]),
                "noise_file": item["noise_file"],
                "source_audio": item["source_audio"],
            }
        )

    manifest_path = preview_dir / "preview_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[preview] Saved {len(preview_items)} augmented files to: {preview_dir}")
    print(f"[preview] Manifest: {manifest_path}")


def split_indices(n_items, validation_split, test_split, random_seed):
    """Split index space into train/val/test."""
    if validation_split < 0 or test_split < 0:
        raise ValueError("validation_split and test_split must be >= 0")
    if validation_split + test_split >= 1.0:
        raise ValueError("validation_split + test_split must be < 1.0")

    all_idx = np.arange(n_items)

    if test_split > 0:
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=test_split,
            random_state=random_seed,
            shuffle=True,
        )
    else:
        train_val_idx = all_idx
        test_idx = np.array([], dtype=np.int64)

    if validation_split > 0:
        val_fraction_of_train_val = validation_split / max(1.0 - test_split, 1e-8)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction_of_train_val,
            random_state=random_seed,
            shuffle=True,
        )
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=np.int64)

    return train_idx, val_idx, test_idx


def build_model(device, dvector_dim):
    """Build deep stacked model and optionally initialize first DAE from base model."""
    base_model = None
    base_model_config = {}
    greedy_model = None
    greedy_model_config = {}

    if BASE_IDENTITY_MODEL_DIR is not None and str(BASE_IDENTITY_MODEL_DIR).strip() != "":
        base_model, base_model_config = load_autoencoder(BASE_IDENTITY_MODEL_DIR, device)

    if GREEDY_INIT_MODEL_DIR is not None and str(GREEDY_INIT_MODEL_DIR).strip() != "":
        greedy_model, greedy_model_config = load_autoencoder(GREEDY_INIT_MODEL_DIR, device)

    input_dim = int(base_model_config.get("input_dim", dvector_dim if dvector_dim > 0 else FALLBACK_INPUT_DIM))
    hidden_dims = list(base_model_config.get("hidden_dims", FALLBACK_HIDDEN_DIMS))
    dropout_rate = float(base_model_config.get("dropout_rate", FALLBACK_DROPOUT_RATE))
    norm_type = str(base_model_config.get("norm_type", FALLBACK_NORM_TYPE))
    activation_type = str(base_model_config.get("activation_type", FALLBACK_ACTIVATION_TYPE))
    use_residual = bool(base_model_config.get("use_residual", FALLBACK_USE_RESIDUAL))
    residual_scale_init = float(base_model_config.get("residual_scale_init", FALLBACK_RESIDUAL_SCALE_INIT))

    model = DeepStackedDvectorAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        norm_type=norm_type,
        activation_type=activation_type,
        use_residual=use_residual,
        residual_scale_init=residual_scale_init,
        n_stacked_daes=int(DEEP_STACKED_NUM_DAES),
        stack_refinement_hidden_dims=DEEP_STACKED_REFINEMENT_HIDDEN_DIMS,
        first_block_hidden_dims=DEEP_STACKED_FIRST_BLOCK_HIDDEN_DIMS,
    )

    if INIT_FIRST_BLOCK_FROM_GREEDY and greedy_model is not None:
        greedy_hidden_dims = greedy_model_config.get("greedy_hidden_dims", [])
        expected_hidden_dims = list(DEEP_STACKED_FIRST_BLOCK_HIDDEN_DIMS)
        if list(greedy_hidden_dims) != list(expected_hidden_dims):
            print(
                "[init] Greedy init skipped: hidden dims mismatch "
                f"(greedy={greedy_hidden_dims}, first_block={expected_hidden_dims})"
            )
        else:
            _init_first_block_from_greedy(model, greedy_model)
    
    # Print model summary
    if device == "cuda":
        summary(model.cuda(), (dvector_dim,))
    else:
        summary(model, (dvector_dim,))

    resolved_config = {
        "model_type": "deep_stacked_dae",
        "input_dim": int(input_dim),
        "hidden_dims": list(hidden_dims),
        "dropout_rate": float(dropout_rate),
        "norm_type": str(norm_type),
        "activation_type": str(activation_type),
        "use_residual": bool(use_residual),
        "residual_scale_init": float(residual_scale_init),
        "n_stacked_daes": int(DEEP_STACKED_NUM_DAES),
        "first_block_hidden_dims": list(DEEP_STACKED_FIRST_BLOCK_HIDDEN_DIMS),
        "stack_refinement_hidden_dims": [
            [int(v) for v in block_dims] if isinstance(block_dims, (list, tuple)) else int(block_dims)
            for block_dims in DEEP_STACKED_REFINEMENT_HIDDEN_DIMS
        ],
        "greedy_init_model_dir": str(GREEDY_INIT_MODEL_DIR) if GREEDY_INIT_MODEL_DIR else None,
        "init_first_block_from_greedy": bool(INIT_FIRST_BLOCK_FROM_GREEDY),
        "greedy_init_hidden_dims": list(greedy_model_config.get("greedy_hidden_dims", [])),
    }

    return model, base_model_config, resolved_config


def _linear_layers(module):
    return [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]


def _init_first_block_from_greedy(model, greedy_model):
    """Initialize deep-stacked first block from greedy-layerwise stacked model."""
    greedy_encoders = getattr(greedy_model, "encoders", None)
    greedy_decoders = getattr(greedy_model, "decoders", None)
    if greedy_encoders is None or greedy_decoders is None:
        print("[init] Greedy init skipped: model has no encoders/decoders")
        return

    greedy_encoder_linears = []
    for encoder in greedy_encoders:
        greedy_encoder_linears.extend(_linear_layers(encoder))

    greedy_decoder_linears = []
    for decoder in greedy_decoders:
        greedy_decoder_linears.extend(_linear_layers(decoder))

    greedy_linears = list(greedy_encoder_linears) + list(reversed(greedy_decoder_linears))
    first_block_linears = _linear_layers(model.first_block.net)

    if len(greedy_linears) != len(first_block_linears):
        print(
            "[init] Greedy init skipped: linear count mismatch "
            f"(greedy={len(greedy_linears)}, first_block={len(first_block_linears)})"
        )
        return

    for src, dst in zip(greedy_linears, first_block_linears):
        if src.weight.shape != dst.weight.shape or src.bias.shape != dst.bias.shape:
            print("[init] Greedy init skipped: layer shape mismatch")
            return

    with torch.no_grad():
        for src, dst in zip(greedy_linears, first_block_linears):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)

    print("[init] First block initialized from greedy-layerwise model")


def _normalize_output(recon):
    recon = torch.sigmoid(recon)
    return F.normalize(recon, p=2, dim=1, eps=LOSS_COSINE_EPS)


def evaluate_model(model, data_loader, device):
    """Evaluate mixed loss and cosine improvements on a loader."""
    if data_loader is None or len(data_loader) == 0:
        return None

    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_mse = 0.0
    total_cos_loss = 0.0
    input_target_cos = []
    recon_target_cos = []

    with torch.no_grad():
        for noisy_batch, clean_batch in data_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            clean_batch_norm = F.normalize(clean_batch, p=2, dim=1, eps=LOSS_COSINE_EPS)
            noisy_batch = F.normalize(noisy_batch, p=2, dim=1, eps=LOSS_COSINE_EPS)

            recon = model(noisy_batch)
            recon = _normalize_output(recon)

            mse_loss = F.mse_loss(recon, clean_batch_norm)
            cos_loss = 1.0 - F.cosine_similarity(
                recon, clean_batch_norm, dim=1, eps=LOSS_COSINE_EPS
            ).mean()
            mixed_loss = (LOSS_MSE_WEIGHT * mse_loss) + (LOSS_COSINE_WEIGHT * cos_loss)

            total_loss += mixed_loss.item()
            total_mse += mse_loss.item()
            total_cos_loss += cos_loss.item()

            input_cos = F.cosine_similarity(
                noisy_batch, clean_batch_norm, dim=1, eps=LOSS_COSINE_EPS
            ).detach().cpu().numpy()
            recon_cos = F.cosine_similarity(
                recon, clean_batch_norm, dim=1, eps=LOSS_COSINE_EPS
            ).detach().cpu().numpy()

            input_target_cos.append(input_cos)
            recon_target_cos.append(recon_cos)

    n_batches = len(data_loader)
    input_target_cos = np.concatenate(input_target_cos)
    recon_target_cos = np.concatenate(recon_target_cos)

    return {
        "mixed_loss": total_loss / n_batches,
        "mse_loss": total_mse / n_batches,
        "cosine_loss": total_cos_loss / n_batches,
        "input_target_cosine_mean": float(np.mean(input_target_cos)),
        "input_target_cosine_std": float(np.std(input_target_cos)),
        "recon_target_cosine_mean": float(np.mean(recon_target_cos)),
        "recon_target_cosine_std": float(np.std(recon_target_cos)),
        "cosine_improvement_mean": float(np.mean(recon_target_cos - input_target_cos)),
        "input_target_cosines": input_target_cos,
        "recon_target_cosines": recon_target_cos,
    }


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir):
    """Train deep stacked model on noisy->clean mapping."""
    print("\n[train] Training deep stacked autoencoder")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    history = {
        "train_loss": [],
        "train_mse_loss": [],
        "train_cosine_loss": [],
        "val_loss": [],
        "val_mse_loss": [],
        "val_cosine_loss": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_cos = 0.0

        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            clean_batch_norm = F.normalize(clean_batch, p=2, dim=1, eps=LOSS_COSINE_EPS)
            noisy_batch = F.normalize(noisy_batch, p=2, dim=1, eps=LOSS_COSINE_EPS)

            optimizer.zero_grad()
            recon = model(noisy_batch)
            recon = _normalize_output(recon)
            # print(f"recon shape: {recon.shape}, clean_batch shape: {clean_batch_norm.shape}")

            mse_loss = F.mse_loss(recon, clean_batch_norm)
            cos_loss = 1.0 - F.cosine_similarity(
                recon, clean_batch_norm, dim=1, eps=LOSS_COSINE_EPS
            ).mean()
            mixed_loss = (LOSS_MSE_WEIGHT * mse_loss) + (LOSS_COSINE_WEIGHT * cos_loss)

            mixed_loss.backward()
            optimizer.step()

            train_loss += mixed_loss.item()
            train_mse += mse_loss.item()
            train_cos += cos_loss.item()

        n_train_batches = len(train_loader)
        train_loss /= n_train_batches
        train_mse /= n_train_batches
        train_cos /= n_train_batches

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_cos = 0.0

        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                clean_batch_norm = F.normalize(clean_batch, p=2, dim=1, eps=LOSS_COSINE_EPS)
                noisy_batch = F.normalize(noisy_batch, p=2, dim=1, eps=LOSS_COSINE_EPS)
                recon = model(noisy_batch)
                mse_loss = F.mse_loss(recon, clean_batch_norm)
                cos_loss = 1.0 - F.cosine_similarity(
                    recon, clean_batch_norm, dim=1, eps=LOSS_COSINE_EPS
                ).mean()
                mixed_loss = (LOSS_MSE_WEIGHT * mse_loss) + (LOSS_COSINE_WEIGHT * cos_loss)

                val_loss += mixed_loss.item()
                val_mse += mse_loss.item()
                val_cos += cos_loss.item()

        n_val_batches = len(val_loader)
        val_loss /= n_val_batches
        val_mse /= n_val_batches
        val_cos /= n_val_batches

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_mse_loss"].append(train_mse)
        history["train_cosine_loss"].append(train_cos)
        history["val_loss"].append(val_loss)
        history["val_mse_loss"].append(val_mse)
        history["val_cosine_loss"].append(val_cos)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train {train_loss:.6f} (mse {train_mse:.6f}, cos {train_cos:.6f}) | "
            f"Val {val_loss:.6f} (mse {val_mse:.6f}, cos {val_cos:.6f}) | "
            f"LR {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = Path(save_dir) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mse_loss": val_mse,
                    "val_cosine_loss": val_cos,
                },
                best_model_path,
            )
            print(f"  Saved best model (val_loss={val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n[train] Early stopping at epoch {epoch + 1}")
            break

    print(f"\n[train] Done. Best val loss: {best_val_loss:.6f}")
    return history, best_val_loss


def plot_training_history(history, save_path):
    """Plot train/val losses and LR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mixed loss")
    axes[0].set_title("Training History")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["learning_rate"], color="green", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved training history: {save_path}")


def plot_cosine_comparison(eval_results, save_path):
    """Plot cosine distribution before and after reconstruction."""
    if eval_results is None:
        return

    input_cos = eval_results["input_target_cosines"]
    recon_cos = eval_results["recon_target_cosines"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(input_cos, bins=50, alpha=0.7, label="Input vs target", edgecolor="black")
    axes[0].hist(recon_cos, bins=50, alpha=0.7, label="Recon vs target", edgecolor="black")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Cosine Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(input_cos, recon_cos, alpha=0.35, s=10)
    min_v = min(float(np.min(input_cos)), float(np.min(recon_cos)))
    max_v = max(float(np.max(input_cos)), float(np.max(recon_cos)))
    axes[1].plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="No change")
    axes[1].set_xlabel("Input vs target cosine")
    axes[1].set_ylabel("Recon vs target cosine")
    axes[1].set_title("Per-sample Cosine Improvement")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved cosine comparison: {save_path}")


def save_model_and_config(model, config, save_dir):
    """Save final model and run configuration."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "final_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[save] Model: {model_path}")

    config_path = save_dir / "config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"[save] Config: {config_path}")

    summary_path = save_dir / "config_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DEEP STACKED LIBRI FINE-TUNING CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Base model dir: {config['base_identity_model_dir']}\n")
        f.write(f"Model type: {config.get('model_type', 'deep_stacked_dae')}\n")
        f.write(f"Stacked DAEs: {config.get('n_stacked_daes', 2)}\n")
        f.write(f"Refinement hidden dims: {config.get('stack_refinement_hidden_dims', [1024, 1024])}\n")
        f.write(f"Libri root: {config['librispeech_root']}\n")
        f.write(f"Libri subsets: {config['librispeech_subsets']}\n")
        f.write(f"Target utterances: {config['n_librispeech_utterances']}\n")
        f.write(f"MUSAN speech root: {config['musan_speech_noise_root']}\n")
        f.write(f"SNR values (dB): {config['musan_babble_snr_range_db']}\n")
        f.write(f"Include noise-only silence-target pairs: {config['include_noise_only_silence_target_pairs']}\n")
        f.write(f"Pairs used: {config['n_pairs']}\n")
        f.write(f"Train/Val/Test: {config['n_train']}/{config['n_val']}/{config['n_test']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Learning rate: {config['learning_rate']}\n")
        f.write(f"Epochs: {config['num_epochs']}\n")
        f.write(f"Loss weights (mse/cos): {config['loss_mse_weight']}/{config['loss_cosine_weight']}\n")
        f.write(f"Device: {config['device']}\n")

        eval_results = config.get("eval_results")
        if eval_results:
            f.write("\nEVALUATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mixed loss: {eval_results['mixed_loss']:.6f}\n")
            f.write(f"MSE loss: {eval_results['mse_loss']:.6f}\n")
            f.write(f"Cosine loss: {eval_results['cosine_loss']:.6f}\n")
            f.write(
                f"Input->target cosine: {eval_results['input_target_cosine_mean']:.4f} +/- "
                f"{eval_results['input_target_cosine_std']:.4f}\n"
            )
            f.write(
                f"Recon->target cosine: {eval_results['recon_target_cosine_mean']:.4f} +/- "
                f"{eval_results['recon_target_cosine_std']:.4f}\n"
            )
            f.write(f"Mean cosine improvement: {eval_results['cosine_improvement_mean']:.4f}\n")

    print(f"[save] Summary: {summary_path}")


def main():
    # np.random.seed(RANDOM_SEED)
    # random.seed(RANDOM_SEED)
    # torch.manual_seed(RANDOM_SEED)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(RANDOM_SEED)

    script_dir = Path(__file__).parent

    base_model_dir = None
    if BASE_IDENTITY_MODEL_DIR is not None and str(BASE_IDENTITY_MODEL_DIR).strip() != "":
        base_model_dir = Path(BASE_IDENTITY_MODEL_DIR)
        if not base_model_dir.is_absolute():
            base_model_dir = script_dir / base_model_dir

    librispeech_root = Path(LIBRISPEECH_ROOT)
    if not librispeech_root.is_absolute():
        librispeech_root = script_dir / librispeech_root

    musan_root = Path(MUSAN_SPEECH_NOISE_ROOT)
    if not musan_root.is_absolute():
        musan_root = script_dir / musan_root

    save_dir = Path(MODEL_SAVE_DIR)
    if not save_dir.is_absolute():
        save_dir = script_dir / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = Path(AUGMENTED_AUDIO_PREVIEW_DIR)
    if not preview_dir.is_absolute():
        preview_dir = script_dir / preview_dir

    cache_dir = Path(DVECTOR_CACHE_DIR)
    if not cache_dir.is_absolute():
        cache_dir = script_dir / cache_dir
    if DVECTOR_CACHE_ENABLED:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("DEEP STACKED D-VECTOR AE TRAINING (LIBRISPEECH CLEAN + MUSAN BABBLED INPUT)")
    print("=" * 90)
    print("\n[config]")
    print(f"  Base model dir: {base_model_dir}")
    print(f"  Init first DAE from base: {INIT_FIRST_DAE_FROM_BASE}")
    print(f"  Greedy init model dir: {GREEDY_INIT_MODEL_DIR}")
    print(f"  Init first block from greedy: {INIT_FIRST_BLOCK_FROM_GREEDY}")
    print(f"  Libri root: {librispeech_root}")
    print(f"  Libri subsets: {LIBRISPEECH_SUBSETS}")
    print(f"  Libri utterances target: {N_LIBRI_UTTERANCES}")
    print(f"  Duration filter (sec): min={MIN_UTTERANCE_SEC}, max={MAX_UTTERANCE_SEC}")
    print(f"  MUSAN speech root: {musan_root}")
    print(f"  SNR values (dB): {MUSAN_BABBLE_SNR_RANGE_DB}")
    print(f"  Deep stacked DAEs: {DEEP_STACKED_NUM_DAES}")
    print(f"  Deep stacked refinement hidden dims: {DEEP_STACKED_REFINEMENT_HIDDEN_DIMS}")
    print(f"  Save augmented previews: {SAVE_AUGMENTED_AUDIO_PREVIEW}")
    if SAVE_AUGMENTED_AUDIO_PREVIEW:
        print(f"  Preview count: {N_AUGMENTED_AUDIO_PREVIEW}")
        print(f"  Preview dir: {preview_dir}")
    print(f"  Model output dir: {save_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  Cache: {'enabled' if DVECTOR_CACHE_ENABLED else 'disabled'}")
    if DVECTOR_CACHE_ENABLED:
        print(f"  Cache dir: {cache_dir}")

    if base_model_dir is not None and (not base_model_dir.exists()):
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not librispeech_root.exists():
        raise FileNotFoundError(f"LibriSpeech root not found: {librispeech_root}")
    if not musan_root.exists():
        raise FileNotFoundError(f"MUSAN speech root not found: {musan_root}")

    extraction_cache_config = {
        "source": "deep_stacked_libri_noisy_clean_pairs",
        "librispeech_root": str(librispeech_root.resolve()),
        "librispeech_subsets": list(LIBRISPEECH_SUBSETS),
        "n_librispeech_utterances": int(N_LIBRI_UTTERANCES),
        "min_utterance_sec": float(MIN_UTTERANCE_SEC) if MIN_UTTERANCE_SEC is not None else None,
        "max_utterance_sec": float(MAX_UTTERANCE_SEC) if MAX_UTTERANCE_SEC is not None else None,
        "musan_speech_noise_root": str(musan_root.resolve()),
        "sample_rate": int(SAMPLE_RATE),
        "snr_range_db": list(MUSAN_BABBLE_SNR_RANGE_DB),
        "random_seed": int(RANDOM_SEED),
    }

    if INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS:
        extraction_cache_config["include_noise_only_silence_target_pairs"] = True

    # Prepare checkpoint directory for partial extraction
    checkpoint_dir = _checkpoint_dir_for_config(cache_dir, EXTRACTION_CACHE_NAME, extraction_cache_config)
    
    pair_data = _extract_with_cache(
        cache_dir=cache_dir,
        cache_name=EXTRACTION_CACHE_NAME,
        cache_config=extraction_cache_config,
        extractor_fn=lambda: extract_noisy_clean_dvector_pairs_from_libri(
            librispeech_root=librispeech_root,
            librispeech_subsets=LIBRISPEECH_SUBSETS,
            musan_speech_noise_root=musan_root,
            n_utterances=N_LIBRI_UTTERANCES,
            min_utt_sec=MIN_UTTERANCE_SEC,
            max_utt_sec=MAX_UTTERANCE_SEC,
            sample_rate=SAMPLE_RATE,
            device=DEVICE,
            snr_range_db=MUSAN_BABBLE_SNR_RANGE_DB,
            include_noise_only_zero_target_pairs=INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS,
            random_seed=RANDOM_SEED,
            preview_limit=N_AUGMENTED_AUDIO_PREVIEW,
            checkpoint_dir=checkpoint_dir,
        ),
    )

    if SAVE_AUGMENTED_AUDIO_PREVIEW:
        save_augmented_preview_audio(pair_data.get("preview_items", []), preview_dir)

    n_pairs = len(pair_data["keys"])
    noisy_dvectors = pair_data["noisy_dvectors"]
    clean_dvectors = pair_data["clean_dvectors"]

    print("\n[data]")
    print(f"  Paired samples: {n_pairs}")
    print(f"  D-vector shape: {noisy_dvectors.shape}")

    train_idx, val_idx, test_idx = split_indices(
        n_items=n_pairs,
        validation_split=VALIDATION_SPLIT,
        test_split=TEST_SPLIT,
        random_seed=RANDOM_SEED,
    )

    print("\n[split]")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")

    if len(train_idx) == 0:
        raise RuntimeError("Training split is empty. Adjust splits or increase data.")
    if len(train_idx) < 2:
        raise RuntimeError("Training split has fewer than 2 samples. Increase data or reduce splits.")
    if len(val_idx) == 0:
        raise RuntimeError("Validation split is empty. Increase VALIDATION_SPLIT.")

    train_dataset = NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, train_idx)
    val_dataset = NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, val_idx)
    test_dataset = (
        NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, test_idx)
        if len(test_idx) > 0
        else None
    )

    # Only drop last batch when it would be a single-sample remainder (BatchNorm safety).
    train_drop_last = (len(train_dataset) % BATCH_SIZE) == 1
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=train_drop_last)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None

    print("\n[model] Building deep stacked model")
    model, base_model_config, resolved_model_config = build_model(DEVICE, noisy_dvectors.shape[1])
    print(f"[model] Model class: {model.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    history, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_dir=save_dir,
    )

    best_model_path = save_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_model_path}")

    best_checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"\n[model] Loaded best checkpoint from epoch {best_checkpoint['epoch'] + 1}")

    eval_loader = test_loader if test_loader is not None else val_loader
    eval_split_name = "test" if test_loader is not None else "val"

    eval_results = evaluate_model(model, eval_loader, DEVICE)
    if eval_results:
        print(f"\n[eval] On {eval_split_name} split")
        print(f"  Mixed loss: {eval_results['mixed_loss']:.6f}")
        print(f"  MSE loss: {eval_results['mse_loss']:.6f}")
        print(f"  Cosine loss: {eval_results['cosine_loss']:.6f}")
        print(
            f"  Input->target cosine: {eval_results['input_target_cosine_mean']:.4f} +/- "
            f"{eval_results['input_target_cosine_std']:.4f}"
        )
        print(
            f"  Recon->target cosine: {eval_results['recon_target_cosine_mean']:.4f} +/- "
            f"{eval_results['recon_target_cosine_std']:.4f}"
        )
        print(f"  Mean cosine improvement: {eval_results['cosine_improvement_mean']:.4f}")

    eval_summary = None
    if eval_results is not None:
        eval_summary = {
            "mixed_loss": float(eval_results["mixed_loss"]),
            "mse_loss": float(eval_results["mse_loss"]),
            "cosine_loss": float(eval_results["cosine_loss"]),
            "input_target_cosine_mean": float(eval_results["input_target_cosine_mean"]),
            "input_target_cosine_std": float(eval_results["input_target_cosine_std"]),
            "recon_target_cosine_mean": float(eval_results["recon_target_cosine_mean"]),
            "recon_target_cosine_std": float(eval_results["recon_target_cosine_std"]),
            "cosine_improvement_mean": float(eval_results["cosine_improvement_mean"]),
        }

    plot_training_history(history, save_dir / "training_history.png")
    if eval_results is not None:
        plot_cosine_comparison(eval_results, save_dir / "cosine_comparison.png")

    config = {
        # Model architecture (keys expected by loader/eval scripts)
        "model_type": resolved_model_config["model_type"],
        "input_dim": resolved_model_config["input_dim"],
        "hidden_dims": resolved_model_config["hidden_dims"],
        "dropout_rate": resolved_model_config["dropout_rate"],
        "norm_type": resolved_model_config["norm_type"],
        "activation_type": resolved_model_config["activation_type"],
        "use_residual": resolved_model_config["use_residual"],
        "residual_scale_init": resolved_model_config["residual_scale_init"],
        "n_stacked_daes": resolved_model_config["n_stacked_daes"],
        "first_block_hidden_dims": resolved_model_config["first_block_hidden_dims"],
        "stack_refinement_hidden_dims": resolved_model_config["stack_refinement_hidden_dims"],

        "base_identity_model_dir": str(base_model_dir) if base_model_dir is not None else None,
        "base_model_config": base_model_config,
        "deep_stacked_enabled": True,
        "deep_stacked_init_first_dae_from_base": bool(INIT_FIRST_DAE_FROM_BASE),
        "greedy_init_model_dir": str(GREEDY_INIT_MODEL_DIR) if GREEDY_INIT_MODEL_DIR else None,
        "init_first_block_from_greedy": bool(INIT_FIRST_BLOCK_FROM_GREEDY),

        "librispeech_root": str(librispeech_root),
        "librispeech_subsets": list(LIBRISPEECH_SUBSETS),
        "n_librispeech_utterances": int(N_LIBRI_UTTERANCES),
        "min_utterance_sec": float(MIN_UTTERANCE_SEC) if MIN_UTTERANCE_SEC is not None else None,
        "max_utterance_sec": float(MAX_UTTERANCE_SEC) if MAX_UTTERANCE_SEC is not None else None,
        "musan_speech_noise_root": str(musan_root),
        "musan_babble_snr_range_db": MUSAN_BABBLE_SNR_RANGE_DB,
        "include_noise_only_silence_target_pairs": bool(INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS),

        "sample_rate": SAMPLE_RATE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "loss_mse_weight": LOSS_MSE_WEIGHT,
        "loss_cosine_weight": LOSS_COSINE_WEIGHT,
        "device": DEVICE,
        "random_seed": RANDOM_SEED,

        "dvector_cache_enabled": DVECTOR_CACHE_ENABLED,
        "dvector_cache_dir": str(cache_dir),
        "dvector_cache_version": DVECTOR_CACHE_VERSION,
        "extraction_cache_name": EXTRACTION_CACHE_NAME,

        "save_augmented_audio_preview": SAVE_AUGMENTED_AUDIO_PREVIEW,
        "n_augmented_audio_preview": N_AUGMENTED_AUDIO_PREVIEW,
        "augmented_audio_preview_dir": str(preview_dir),

        "n_pairs": n_pairs,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),

        "best_val_loss": float(best_val_loss),
        "eval_split": eval_split_name,
        "eval_results": eval_summary,
        "test_mixed_loss": eval_summary["mixed_loss"] if eval_summary is not None else None,
        "test_mse_loss": eval_summary["mse_loss"] if eval_summary is not None else None,
        "test_cosine_loss": eval_summary["cosine_loss"] if eval_summary is not None else None,
    }

    save_model_and_config(model, config, save_dir)

    print("\nDone.")
    print(f"  Best model checkpoint: {best_model_path}")
    print(f"  Final model: {save_dir / 'final_model.pth'}")
    print(f"  Config: {save_dir / 'config.pkl'}")


if __name__ == "__main__":
    main()
