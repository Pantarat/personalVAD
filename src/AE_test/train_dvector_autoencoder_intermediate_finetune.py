#!/usr/bin/env python3
"""
Intermediate fine-tuning for d-vector autoencoder.

Goal:
- Start from a pre-trained identity autoencoder.
- Use clean train-other-mix audio as target.
- Add MUSAN speech babble to that audio as input.
- Fine-tune AE to reconstruct clean d-vectors from noisy d-vectors.

The script also:
- Saves a small preview set of augmented audio files for listening checks.
- Caches processed (noisy, clean) d-vector pairs with a deterministic hash key.
"""

import hashlib
import inspect
import json
import pickle
import random
import time
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

from autoencoder_utils import DeepStackedDvectorAutoencoder, load_autoencoder

# Suppress FutureWarning from resemblyzer library (librosa positional args deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="resemblyzer")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base model to fine-tune (directory containing config.pkl + final_model.pth / best_model.pth)
BASE_IDENTITY_MODEL_DIR = "test_outputs/models/dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_v2_12-4-26"

# Input overlap/mix directories (clean train-other-mix preferred)
TRAIN_OTHER_MIX_DIRS = [
    "test_outputs/data/train_other_mix_1,6s_100000_balanced_intermediate_18-4",
    # "test_outputs/data/train_other_mix_1,6s_100000_balanced_4-1",
]

# MUSAN speech root for babble augmentation
MUSAN_SPEECH_NOISE_ROOT = "../../kaldi/egs/pvad/musan/musan_speech_train"
# Match Kaldi pvad babble augmentation SNR choices.
MUSAN_BABBLE_SNR_RANGE_DB = (20.0, 17.0, 15.0, 13.0)

# If True, skip source samples that already have MUSAN noise according to metadata
SKIP_SAMPLES_WITH_EXISTING_MUSAN_NOISE = True

# Number of overlap samples to use per input directory (-1 = all)
N_OVERLAP_SAMPLES_PER_DIR = -1

# Preview augmented audio exports for listening/debugging
SAVE_AUGMENTED_AUDIO_PREVIEW = True
N_AUGMENTED_AUDIO_PREVIEW = 10
AUGMENTED_AUDIO_PREVIEW_DIR = "test_outputs/debug/intermediate_finetune_augmented_preview"

# Output model directory
MODEL_SAVE_DIR = "test_outputs/models/dvector_ae_intermediate_babble_finetune_23-4-26"

# Deep Stacked DAE settings (paper-inspired)
# When enabled, model = first DAE + refinement DAEs with concat(prev, noisy-prev).
USE_DEEP_STACKED_DAE = False
DEEP_STACKED_NUM_DAES = 2
DEEP_STACKED_REFINEMENT_HIDDEN_DIMS = [1024, 1024]
DEEP_STACKED_INIT_FIRST_DAE_FROM_BASE = True

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0.0
EARLY_STOPPING_PATIENCE = 40

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
PARTIAL_CACHE_SAVE_EVERY = 500


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


def _normalize_for_hash(value):
    """Normalize nested config payloads into deterministic JSON-serializable structures."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        normalized = {}
        for key in sorted(value.keys(), key=lambda x: str(x)):
            normalized[str(key)] = _normalize_for_hash(value[key])
        return normalized

    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]

    if isinstance(value, set):
        return [_normalize_for_hash(v) for v in sorted(value, key=lambda x: str(x))]

    return value


def _cache_file_for_config(cache_dir, cache_name, cache_config):
    """Build deterministic cache file path from cache name + config."""
    payload = {
        "cache_name": cache_name,
        "cache_version": DVECTOR_CACHE_VERSION,
        "config": _normalize_for_hash(cache_config),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{cache_name}_{digest}.pkl"


def _normalize_cached_extraction_data(data):
    """Normalize cached extraction payload into the canonical in-memory format."""
    if not isinstance(data, dict):
        return data

    normalized = dict(data)

    if "keys" not in normalized or normalized["keys"] is None:
        normalized["keys"] = []

    if "metadata" not in normalized or normalized["metadata"] is None:
        normalized["metadata"] = []

    if "preview_items" not in normalized or normalized["preview_items"] is None:
        normalized["preview_items"] = []

    for vector_key in ("noisy_dvectors", "clean_dvectors"):
        if vector_key in normalized and normalized[vector_key] is not None:
            normalized[vector_key] = np.asarray(normalized[vector_key], dtype=np.float32)

    return normalized


def _extract_with_cache(cache_dir, cache_name, cache_config, extractor_fn):
    """Load extraction cache and resume extraction to fill any missing samples."""
    t_cache_start = time.perf_counter()
    normalized_config = _normalize_for_hash(cache_config)
    cache_file = _cache_file_for_config(cache_dir, cache_name, normalized_config)
    partial_cache_file = cache_file.with_suffix(cache_file.suffix + ".partial")

    def _save_partial_checkpoint(partial_data, n_processed):
        if not DVECTOR_CACHE_ENABLED:
            return

        partial_payload = {
            "cache_version": DVECTOR_CACHE_VERSION,
            "config": normalized_config,
            "is_partial": True,
            "n_processed": int(n_processed),
            "data": partial_data,
        }
        try:
            with open(partial_cache_file, "wb") as f:
                pickle.dump(partial_payload, f)
            print(
                f"[cache] Partial save for {cache_name}: {n_processed} pairs -> {partial_cache_file}"
            )
        except Exception as e:
            print(f"[cache] Could not write partial cache for {cache_name}: {e}")

    resume_data = None

    if DVECTOR_CACHE_ENABLED and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_payload = pickle.load(f)

            if isinstance(cached_payload, dict) and "config" in cached_payload and "data" in cached_payload:
                cached_config = cached_payload.get("config")
                cached_version = cached_payload.get("cache_version")
                if cached_version == DVECTOR_CACHE_VERSION and cached_config == normalized_config:
                    resume_data = _normalize_cached_extraction_data(cached_payload["data"])
                    n_cached = len(resume_data.get("keys", [])) if isinstance(resume_data, dict) else 0
                    print(f"\n[cache] Hit for {cache_name}: {cache_file} (cached pairs: {n_cached})")
                else:
                    print(f"\n[cache] Config/version mismatch for {cache_name}; regenerating")
            else:
                resume_data = _normalize_cached_extraction_data(cached_payload)
                n_cached = len(resume_data.get("keys", [])) if isinstance(resume_data, dict) else 0
                print(f"\n[cache] Legacy hit for {cache_name}: {cache_file} (cached pairs: {n_cached})")
        except Exception as e:
            print(f"\n[cache] Failed reading {cache_name}: {e}")
            print("[cache] Trying partial cache fallback")

    if resume_data is None and DVECTOR_CACHE_ENABLED and partial_cache_file.exists():
        try:
            with open(partial_cache_file, "rb") as f:
                partial_payload = pickle.load(f)

            if isinstance(partial_payload, dict) and "config" in partial_payload and "data" in partial_payload:
                partial_config = partial_payload.get("config")
                partial_version = partial_payload.get("cache_version")
                if partial_version == DVECTOR_CACHE_VERSION and partial_config == normalized_config:
                    resume_data = _normalize_cached_extraction_data(partial_payload["data"])
                    n_recovered = len(resume_data.get("keys", [])) if isinstance(resume_data, dict) else 0
                    print(
                        f"\n[cache] Recovered from partial cache for {cache_name}: "
                        f"{n_recovered} pairs from {partial_cache_file}"
                    )
                else:
                    print(f"\n[cache] Partial cache mismatch for {cache_name}; regenerating")
            else:
                # Legacy partial payloads may store raw extraction data directly.
                resume_data = _normalize_cached_extraction_data(partial_payload)
                n_recovered = len(resume_data.get("keys", [])) if isinstance(resume_data, dict) else 0
                print(
                    f"\n[cache] Recovered legacy partial cache for {cache_name}: "
                    f"{n_recovered} pairs from {partial_cache_file}"
                )
        except Exception as e:
            print(f"\n[cache] Failed reading partial cache for {cache_name}: {e}")
            print("[cache] Re-extracting d-vectors")

    if resume_data is None:
        print(f"\n[cache] Miss for {cache_name}; extracting from scratch")
    else:
        n_resume = len(resume_data.get("keys", [])) if isinstance(resume_data, dict) else 0
        print(f"\n[cache] Using cached resume seed: {n_resume} pairs")

    t_cache_loaded = time.perf_counter()
    print(f"[perf] Cache load phase: {t_cache_loaded - t_cache_start:.2f}s")

    supports_partial_save = False
    supports_resume_data = False
    try:
        signature = inspect.signature(extractor_fn)
        supports_partial_save = "partial_save_fn" in signature.parameters
        supports_resume_data = "resume_data" in signature.parameters
    except (TypeError, ValueError):
        supports_partial_save = False
        supports_resume_data = False

    if resume_data is not None and not supports_resume_data:
        print("[cache] Extractor does not support resume_data; returning cached snapshot")
        return resume_data

    extractor_kwargs = {}
    if supports_partial_save:
        extractor_kwargs["partial_save_fn"] = _save_partial_checkpoint
    if supports_resume_data:
        extractor_kwargs["resume_data"] = resume_data

    if extractor_kwargs:
        data = extractor_fn(**extractor_kwargs)
    else:
        data = extractor_fn()

    t_extract_done = time.perf_counter()
    print(f"[perf] Extraction/merge phase: {t_extract_done - t_cache_loaded:.2f}s")

    data = _normalize_cached_extraction_data(data)

    if DVECTOR_CACHE_ENABLED:
        cache_payload = {
            "cache_version": DVECTOR_CACHE_VERSION,
            "config": normalized_config,
            "data": data,
        }
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_payload, f)
            print(f"[cache] Saved {cache_name} to: {cache_file}")

            if partial_cache_file.exists():
                try:
                    partial_cache_file.unlink()
                    print(f"[cache] Cleared partial cache: {partial_cache_file}")
                except Exception as e:
                    print(f"[cache] Could not remove partial cache for {cache_name}: {e}")
        except Exception as e:
            print(f"[cache] Could not save cache for {cache_name}: {e}")

    t_cache_end = time.perf_counter()
    print(f"[perf] Cache save phase: {t_cache_end - t_extract_done:.2f}s")
    print(f"[perf] Total cache wrapper time: {t_cache_end - t_cache_start:.2f}s")

    return data


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

    mixed = audio_signal + (scale * noise_signal)

    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)

    return mixed


def resolve_overlap_audio_path(metadata, overlap_dir):
    """Resolve audio file path for both legacy and current metadata formats."""
    overlap_dir = Path(overlap_dir)
    audio_dir = overlap_dir / "audio"

    audio_file_token = metadata.get("audio_file")
    if audio_file_token:
        token_path = Path(audio_file_token)
        candidates = []

        if token_path.is_absolute():
            candidates.append(token_path)
        else:
            candidates.append(overlap_dir / token_path)
            candidates.append(audio_dir / token_path)
            candidates.append(audio_dir / token_path.name)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

    overlap_file = metadata.get("overlap_file")
    if overlap_file:
        candidate = audio_dir / overlap_file
        if candidate.exists() and candidate.is_file():
            return candidate

    output_name = metadata.get("output_name", "")
    if output_name:
        if not output_name.endswith(".wav"):
            output_name = f"{output_name}.wav"
        candidate = audio_dir / output_name
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def _build_sample_key(overlap_dir, metadata, audio_path):
    """Build deterministic cache key for one overlap sample."""
    sample_id = metadata.get("sample_id", "")
    output_name = metadata.get("output_name", "")
    stem = output_name if output_name else Path(audio_path).stem
    key_core = sample_id if sample_id else stem
    return f"{Path(overlap_dir).name}_{key_core}"


def _build_sample_key_fast(overlap_dir, metadata):
    """Build key without filesystem access when metadata has enough information."""
    sample_id = metadata.get("sample_id", "")
    if sample_id:
        return f"{Path(overlap_dir).name}_{sample_id}"

    output_name = metadata.get("output_name", "")
    if output_name:
        return f"{Path(overlap_dir).name}_{output_name}"

    overlap_file = metadata.get("overlap_file", "")
    if overlap_file:
        return f"{Path(overlap_dir).name}_{Path(overlap_file).stem}"

    audio_file = metadata.get("audio_file", "")
    if audio_file:
        return f"{Path(overlap_dir).name}_{Path(audio_file).stem}"

    return None


def _collect_metadata_signatures(overlap_dirs):
    """Collect lightweight metadata.json signatures for fast unchanged-data checks."""
    signatures = {}
    for overlap_dir in overlap_dirs:
        dir_path = Path(overlap_dir).resolve()
        metadata_path = dir_path / "metadata.json"
        key = str(dir_path)

        if not metadata_path.exists() or (not metadata_path.is_file()):
            signatures[key] = None
            continue

        stat = metadata_path.stat()
        signatures[key] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    return signatures


def _pack_extraction_data(
    keys,
    noisy_dvectors,
    clean_dvectors,
    metadata_rows,
    preview_items,
    as_numpy=False,
    extra_fields=None,
):
    """Pack extraction data for partial/final cache writes."""
    payload = {
        "keys": list(keys),
        "metadata": list(metadata_rows),
        "preview_items": list(preview_items),
    }

    if as_numpy:
        payload["noisy_dvectors"] = np.asarray(noisy_dvectors, dtype=np.float32)
        payload["clean_dvectors"] = np.asarray(clean_dvectors, dtype=np.float32)
    else:
        payload["noisy_dvectors"] = list(noisy_dvectors)
        payload["clean_dvectors"] = list(clean_dvectors)

    if isinstance(extra_fields, dict):
        payload.update(extra_fields)

    return payload


def _collect_extraction_candidates(
    overlap_dirs,
    n_samples_per_dir,
    skip_existing_musan_noise,
    random_seed,
    cached_keys=None,
):
    """Scan data folders and return extractable sample records."""
    cached_keys = set() if cached_keys is None else set(cached_keys)
    np_rng = np.random.RandomState(random_seed)
    seen_keys = set()
    candidates = []
    stats = {
        "skipped_already_noisy": 0,
        "skipped_missing_audio": 0,
        "skipped_duplicate_key": 0,
        "cached_key_shortcuts": 0,
    }

    dir_progress = tqdm(
        overlap_dirs,
        desc="scan directories",
        unit="dir",
        leave=True,
        position=0,
    )

    for dir_idx, overlap_dir in enumerate(dir_progress, 1):
        overlap_dir = Path(overlap_dir)
        metadata_path = overlap_dir / "metadata.json"
        dir_progress.set_postfix(current=overlap_dir.name)

        print(f"\n  [scan {dir_idx}/{len(overlap_dirs)}] {overlap_dir}")
        if not metadata_path.exists():
            print("    metadata.json missing, skipping directory")
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        if n_samples_per_dir > 0 and len(metadata_list) > n_samples_per_dir:
            sampled_idx = np_rng.choice(len(metadata_list), size=n_samples_per_dir, replace=False)
            metadata_list = [metadata_list[int(i)] for i in sampled_idx]
            print(f"    Using sampled subset: {len(metadata_list)}")
        else:
            print(f"    Using all metadata samples: {len(metadata_list)}")

        dir_candidates = 0
        for item in metadata_list:
            already_noisy = bool(item.get("musan_speech_noise_added", False))
            if skip_existing_musan_noise and already_noisy:
                stats["skipped_already_noisy"] += 1
                continue

            fast_key = _build_sample_key_fast(overlap_dir, item)
            if fast_key is not None and fast_key in cached_keys:
                if fast_key in seen_keys:
                    stats["skipped_duplicate_key"] += 1
                    continue

                seen_keys.add(fast_key)
                candidates.append(
                    {
                        "key": fast_key,
                        "item": item,
                        "overlap_dir": overlap_dir,
                        "audio_path": None,
                        "source_had_musan_noise": already_noisy,
                    }
                )
                stats["cached_key_shortcuts"] += 1
                dir_candidates += 1
                continue

            if fast_key is not None:
                if fast_key in seen_keys:
                    stats["skipped_duplicate_key"] += 1
                    continue

                seen_keys.add(fast_key)
                candidates.append(
                    {
                        "key": fast_key,
                        "item": item,
                        "overlap_dir": overlap_dir,
                        # Resolve path lazily only if this candidate is not cached.
                        "audio_path": None,
                        "source_had_musan_noise": already_noisy,
                    }
                )
                dir_candidates += 1
                continue

            audio_path = resolve_overlap_audio_path(item, overlap_dir)
            if audio_path is None or (not audio_path.exists()):
                stats["skipped_missing_audio"] += 1
                continue

            key = _build_sample_key(overlap_dir, item, audio_path)
            if key in seen_keys:
                stats["skipped_duplicate_key"] += 1
                continue

            seen_keys.add(key)
            candidates.append(
                {
                    "key": key,
                    "item": item,
                    "overlap_dir": overlap_dir,
                    "audio_path": audio_path,
                    "source_had_musan_noise": already_noisy,
                }
            )
            dir_candidates += 1

        print(f"    Candidate samples in directory: {dir_candidates}")

    return candidates, stats


def extract_noisy_clean_dvector_pairs(
    overlap_dirs,
    musan_speech_noise_root,
    n_samples_per_dir=-1,
    sample_rate=16000,
    device="cuda",
    snr_range_db=(20.0, 17.0, 15.0, 13.0),
    skip_existing_musan_noise=True,
    random_seed=42,
    preview_limit=10,
    partial_save_fn=None,
    partial_save_every=500,
    resume_data=None,
):
    """
    Build paired d-vectors:
    - input: d-vector(clean overlap + added MUSAN babble)
    - target: d-vector(clean overlap)
    """
    print("\n[extract] Building noisy->clean d-vector pairs")
    print(f"  Input directories: {len(overlap_dirs)}")
    print(f"  Samples per directory: {n_samples_per_dir}")
    print(f"  SNR values (dB): {snr_range_db}")
    print(f"  Skip already-noisy metadata samples: {skip_existing_musan_noise}")

    t_scan_start = time.perf_counter()

    resume_data = _normalize_cached_extraction_data(resume_data)
    keys = []
    noisy_dvectors = []
    clean_dvectors = []
    metadata_rows = []
    preview_items = []

    cached_index_by_key = {}
    resume_noisy = np.asarray([], dtype=np.float32)
    resume_clean = np.asarray([], dtype=np.float32)
    resume_meta = []

    if isinstance(resume_data, dict):
        resume_keys = list(resume_data.get("keys", []))
        resume_noisy = np.asarray(resume_data.get("noisy_dvectors", []), dtype=np.float32)
        resume_clean = np.asarray(resume_data.get("clean_dvectors", []), dtype=np.float32)
        resume_meta = list(resume_data.get("metadata", []))
        preview_items = list(resume_data.get("preview_items", []))[:preview_limit]

        core_len = min(len(resume_keys), len(resume_noisy), len(resume_clean))
        if core_len < len(resume_keys) or core_len < len(resume_noisy) or core_len < len(resume_clean):
            print(
                "[extract] Resume cache arrays have mismatched lengths; "
                f"trimming to {core_len} entries"
            )
            resume_keys = resume_keys[:core_len]
            resume_noisy = resume_noisy[:core_len]
            resume_clean = resume_clean[:core_len]

        if len(resume_meta) < core_len:
            resume_meta.extend({} for _ in range(core_len - len(resume_meta)))
        else:
            resume_meta = resume_meta[:core_len]

        for i, key in enumerate(resume_keys):
            cached_index_by_key[key] = i

        if len(cached_index_by_key) < len(resume_keys):
            print(
                "[extract] Resume cache contains duplicate keys; "
                f"using latest entries ({len(cached_index_by_key)} unique)"
            )

    metadata_signatures = _collect_metadata_signatures(overlap_dirs)
    resume_metadata_signatures = None
    if isinstance(resume_data, dict):
        resume_metadata_signatures = resume_data.get("metadata_signatures")

    can_skip_scan = (
        isinstance(resume_metadata_signatures, dict)
        and len(cached_index_by_key) > 0
        and resume_metadata_signatures == metadata_signatures
        and resume_data.get("scan_n_samples_per_dir") == int(n_samples_per_dir)
        and resume_data.get("scan_skip_existing_musan_noise") == bool(skip_existing_musan_noise)
        and resume_data.get("scan_random_seed") == int(random_seed)
    )

    if can_skip_scan:
        print("[extract] Metadata signatures unchanged; skipping candidate rescan")
        candidates = []
        for key, idx in cached_index_by_key.items():
            cached_meta = resume_meta[idx] if idx < len(resume_meta) else {}
            if not isinstance(cached_meta, dict):
                cached_meta = {}

            source_dir = cached_meta.get("source_dir", "")
            overlap_dir = Path(source_dir) if source_dir else Path(overlap_dirs[0])
            candidates.append(
                {
                    "key": key,
                    "item": cached_meta,
                    "overlap_dir": overlap_dir,
                    "audio_path": cached_meta.get("audio_path", None),
                    "source_had_musan_noise": bool(cached_meta.get("source_had_musan_noise", False)),
                }
            )

        scan_stats = {
            "skipped_already_noisy": 0,
            "skipped_missing_audio": 0,
            "skipped_duplicate_key": 0,
            "cached_key_shortcuts": len(candidates),
        }
    else:
        candidates, scan_stats = _collect_extraction_candidates(
            overlap_dirs=overlap_dirs,
            n_samples_per_dir=n_samples_per_dir,
            skip_existing_musan_noise=skip_existing_musan_noise,
            random_seed=random_seed,
            cached_keys=set(cached_index_by_key.keys()),
        )

    skipped_already_noisy = int(scan_stats["skipped_already_noisy"])
    skipped_missing_audio = int(scan_stats["skipped_missing_audio"])
    skipped_duplicate_key = int(scan_stats["skipped_duplicate_key"])
    cached_key_shortcuts = int(scan_stats.get("cached_key_shortcuts", 0))

    t_scan_end = time.perf_counter()
    print(f"[perf] Candidate scan phase: {t_scan_end - t_scan_start:.2f}s")

    if len(candidates) == 0:
        hint = ""
        if skipped_already_noisy > 0 and skip_existing_musan_noise:
            hint = (
                " All candidate samples were tagged as already MUSAN-noisy. "
                "Point TRAIN_OTHER_MIX_DIRS to a clean train-other-mix directory "
                "or disable SKIP_SAMPLES_WITH_EXISTING_MUSAN_NOISE."
            )
        raise RuntimeError("No noisy->clean d-vector pairs were created." + hint)

    remaining_candidates = []
    for record in candidates:
        key = record["key"]
        cached_idx = cached_index_by_key.get(key)
        if cached_idx is None:
            remaining_candidates.append(record)
            continue

        keys.append(key)
        noisy_dvectors.append(np.asarray(resume_noisy[cached_idx], dtype=np.float32))
        clean_dvectors.append(np.asarray(resume_clean[cached_idx], dtype=np.float32))

        cached_metadata = resume_meta[cached_idx] if cached_idx < len(resume_meta) else None
        if isinstance(cached_metadata, dict) and len(cached_metadata) > 0:
            metadata_rows.append(cached_metadata)
        else:
            item = record["item"]
            audio_path_value = ""
            if record.get("audio_path") is not None:
                audio_path_value = str(record["audio_path"])
            metadata_rows.append(
                {
                    "key": key,
                    "sample_id": item.get("sample_id", ""),
                    "output_name": item.get("output_name", ""),
                    "source_dir": str(record["overlap_dir"]),
                    "audio_path": audio_path_value,
                    "source_had_musan_noise": record["source_had_musan_noise"],
                }
            )

    n_candidates = len(candidates)
    n_cached = len(keys)
    n_remaining = len(remaining_candidates)
    stale_cached = max(0, len(cached_index_by_key) - n_cached)

    print("\n[extract] Candidate coverage")
    print(f"  Candidate samples from data folders: {n_candidates}")
    print(f"  Candidate scan shortcuts from cached keys: {cached_key_shortcuts}")
    print(f"  Already cached and reusable: {n_cached}")
    print(f"  Remaining to process now: {n_remaining}")
    if stale_cached > 0:
        print(f"  Ignoring stale cached samples not in current folders/config: {stale_cached}")

    skipped_extraction_error = 0
    cache_extra_fields = {
        "metadata_signatures": metadata_signatures,
        "scan_n_samples_per_dir": int(n_samples_per_dir),
        "scan_skip_existing_musan_noise": bool(skip_existing_musan_noise),
        "scan_random_seed": int(random_seed),
    }

    if n_remaining > 0:
        if torch.cuda.is_available() and device == "cuda":
            device_obj = torch.device("cuda")
            print("  Encoder device: CUDA")
        else:
            device_obj = torch.device("cpu")
            print("  Encoder device: CPU")

        encoder = VoiceEncoder(device=device_obj)
        print("  Voice encoder loaded")

        noise_files = discover_musan_speech_noise_files(musan_speech_noise_root)
        if len(noise_files) == 0:
            raise FileNotFoundError(
                f"No MUSAN speech files found under: {musan_speech_noise_root}"
            )
        print(f"  Discovered MUSAN speech noise files: {len(noise_files)}")

        rng = random.Random(random_seed)
        snr_values_db = [float(v) for v in snr_range_db]
        if len(snr_values_db) == 0:
            raise ValueError("snr_range_db must contain at least one SNR value")
        process_progress = tqdm(
            remaining_candidates,
            desc="extract remaining",
            unit="sample",
            leave=True,
            position=0,
        )

        for record in process_progress:
            try:
                item = record["item"]
                audio_path = record["audio_path"]
                overlap_dir = record["overlap_dir"]
                key = record["key"]
                already_noisy = record["source_had_musan_noise"]

                if audio_path is None:
                    audio_path = resolve_overlap_audio_path(item, overlap_dir)
                    if audio_path is None or (not audio_path.exists()):
                        skipped_missing_audio += 1
                        continue

                clean_audio, _ = librosa.load(str(audio_path), sr=sample_rate)
                if len(clean_audio) == 0:
                    skipped_extraction_error += 1
                    continue

                noise_path = rng.choice(noise_files)
                noise_segment = load_noise_segment(
                    noise_path,
                    target_samples=len(clean_audio),
                    target_sr=sample_rate,
                    rng=rng,
                )
                if noise_segment is None:
                    skipped_extraction_error += 1
                    continue

                snr_db = float(rng.choice(snr_values_db))
                noisy_audio = mix_audio_with_snr(clean_audio, noise_segment, snr_db)

                with torch.no_grad():
                    clean_dvec = encoder.embed_utterance(clean_audio).astype(np.float32)
                    noisy_dvec = encoder.embed_utterance(noisy_audio).astype(np.float32)

                keys.append(key)
                noisy_dvectors.append(noisy_dvec)
                clean_dvectors.append(clean_dvec)

                metadata_rows.append(
                    {
                        "key": key,
                        "sample_id": item.get("sample_id", ""),
                        "output_name": item.get("output_name", ""),
                        "source_dir": str(overlap_dir),
                        "audio_path": str(audio_path),
                        "noise_file": str(noise_path),
                        "snr_db": float(snr_db),
                        "source_had_musan_noise": already_noisy,
                    }
                )

                if len(preview_items) < preview_limit:
                    preview_items.append(
                        {
                            "key": key,
                            "audio": noisy_audio.astype(np.float32),
                            "sample_rate": sample_rate,
                            "snr_db": float(snr_db),
                            "noise_file": str(noise_path),
                            "source_audio": str(audio_path),
                        }
                    )

                if (
                    partial_save_fn is not None
                    and partial_save_every > 0
                    and (len(keys) % partial_save_every == 0)
                ):
                    partial_save_fn(
                        _pack_extraction_data(
                            keys,
                            noisy_dvectors,
                            clean_dvectors,
                            metadata_rows,
                            preview_items,
                            as_numpy=False,
                            extra_fields=cache_extra_fields,
                        ),
                        len(keys),
                    )

            except Exception:
                skipped_extraction_error += 1
                continue

            if len(keys) > 0 and (len(keys) % 50 == 0):
                process_progress.set_postfix(
                    total=len(keys),
                    remaining=max(0, n_candidates - len(keys)),
                )

    if len(keys) == 0:
        hint = ""
        if skipped_already_noisy > 0 and skip_existing_musan_noise:
            hint = (
                " All candidate samples were tagged as already MUSAN-noisy. "
                "Point TRAIN_OTHER_MIX_DIRS to a clean train-other-mix directory "
                "or disable SKIP_SAMPLES_WITH_EXISTING_MUSAN_NOISE."
            )
        raise RuntimeError("No noisy->clean d-vector pairs were created." + hint)

    print("\n[extract] Summary")
    print(f"  Total pairs: {len(keys)}")
    print(f"  Reused from cache: {n_cached}")
    print(f"  Added in this run: {max(0, len(keys) - n_cached)}")
    print(f"  Skipped already noisy: {skipped_already_noisy}")
    print(f"  Skipped missing audio: {skipped_missing_audio}")
    print(f"  Cached key shortcuts used during scan: {cached_key_shortcuts}")
    print(f"  Skipped duplicate keys: {skipped_duplicate_key}")
    print(f"  Skipped extraction errors: {skipped_extraction_error}")

    if partial_save_fn is not None and len(keys) > 0:
        partial_save_fn(
            _pack_extraction_data(
                keys,
                noisy_dvectors,
                clean_dvectors,
                metadata_rows,
                preview_items,
                as_numpy=False,
                extra_fields=cache_extra_fields,
            ),
            len(keys),
        )

    return _pack_extraction_data(
        keys,
        noisy_dvectors,
        clean_dvectors,
        metadata_rows,
        preview_items,
        as_numpy=True,
        extra_fields=cache_extra_fields,
    )


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

    print(f"\n[preview] Saved {len(preview_items)} augmented audio files to: {preview_dir}")
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


def build_finetune_model(base_model_dir, device):
    """Build the model used for intermediate fine-tuning."""
    base_model, base_model_config = load_autoencoder(base_model_dir, device)

    if not USE_DEEP_STACKED_DAE:
        return base_model, base_model_config

    input_dim = int(base_model_config.get("input_dim", 256))
    hidden_dims = list(base_model_config.get("hidden_dims", [128, 64, 128]))
    dropout_rate = float(base_model_config.get("dropout_rate", 0.2))
    norm_type = str(base_model_config.get("norm_type", "batchnorm"))
    use_residual = bool(base_model_config.get("use_residual", False))
    residual_scale_init = float(base_model_config.get("residual_scale_init", 0.5))

    n_stacked = max(1, int(DEEP_STACKED_NUM_DAES))
    if n_stacked < 2:
        print(
            "[model] DEEP_STACKED_NUM_DAES < 2, "
            "falling back to a single standard DAE"
        )

    model = DeepStackedDvectorAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        norm_type=norm_type,
        use_residual=use_residual,
        residual_scale_init=residual_scale_init,
        n_stacked_daes=n_stacked,
        stack_refinement_hidden_dims=DEEP_STACKED_REFINEMENT_HIDDEN_DIMS,
    )

    if DEEP_STACKED_INIT_FIRST_DAE_FROM_BASE:
        if isinstance(base_model, DeepStackedDvectorAutoencoder):
            seed_state = base_model.first_dae.state_dict()
        else:
            seed_state = base_model.state_dict()
        model.first_dae.load_state_dict(seed_state, strict=True)
        print("[model] Initialized first DAE block from base identity checkpoint")

    return model, base_model_config


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

            recon = model(noisy_batch)

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
    """Fine-tune model on noisy->clean mapping."""
    print("\n[train] Fine-tuning autoencoder")
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

            optimizer.zero_grad()
            recon = model(noisy_batch)

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
        f.write("INTERMEDIATE FINE-TUNING CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Base model dir: {config['base_identity_model_dir']}\n")
        f.write(f"Model type: {config.get('model_type', 'dvector_autoencoder')}\n")
        if config.get("model_type") == "deep_stacked_dae":
            f.write(f"Stacked DAEs: {config.get('n_stacked_daes', 2)}\n")
            f.write(
                f"Refinement hidden dims: {config.get('stack_refinement_hidden_dims', [1024, 1024])}\n"
            )
            f.write(
                f"Init first DAE from base: {config.get('deep_stacked_init_first_dae_from_base', False)}\n"
            )
        f.write(f"Input dirs: {config['train_other_mix_dirs']}\n")
        f.write(f"MUSAN speech root: {config['musan_speech_noise_root']}\n")
        f.write(f"Skip already-noisy source: {config['skip_samples_with_existing_musan_noise']}\n")
        f.write(f"SNR values (dB): {config['musan_babble_snr_range_db']}\n")
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
            f.write(
                f"Mean cosine improvement: {eval_results['cosine_improvement_mean']:.4f}\n"
            )

    print(f"[save] Summary: {summary_path}")


def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    script_dir = Path(__file__).parent

    base_model_dir = Path(BASE_IDENTITY_MODEL_DIR)
    if not base_model_dir.is_absolute():
        base_model_dir = script_dir / base_model_dir

    input_dirs = []
    for d in TRAIN_OTHER_MIX_DIRS:
        p = Path(d)
        if not p.is_absolute():
            p = script_dir / p
        input_dirs.append(p)

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
    print("INTERMEDIATE D-VECTOR AE FINE-TUNING (NOISY BABBLED INPUT -> CLEAN TARGET)")
    print("=" * 90)
    print("\n[config]")
    print(f"  Base model dir: {base_model_dir}")
    print(f"  Input dirs: {', '.join(str(p) for p in input_dirs)}")
    print(f"  MUSAN speech root: {musan_root}")
    print(f"  SNR values (dB): {MUSAN_BABBLE_SNR_RANGE_DB}")
    print(f"  Skip already-noisy source samples: {SKIP_SAMPLES_WITH_EXISTING_MUSAN_NOISE}")
    print(f"  Samples per dir: {N_OVERLAP_SAMPLES_PER_DIR}")
    print(f"  Save augmented previews: {SAVE_AUGMENTED_AUDIO_PREVIEW}")
    if SAVE_AUGMENTED_AUDIO_PREVIEW:
        print(f"  Preview count: {N_AUGMENTED_AUDIO_PREVIEW}")
        print(f"  Preview dir: {preview_dir}")
    print(f"  Use deep stacked DAE: {USE_DEEP_STACKED_DAE}")
    if USE_DEEP_STACKED_DAE:
        print(f"  Deep stacked DAEs: {DEEP_STACKED_NUM_DAES}")
        print(f"  Deep stacked refinement hidden dims: {DEEP_STACKED_REFINEMENT_HIDDEN_DIMS}")
        print(f"  Init first DAE from base: {DEEP_STACKED_INIT_FIRST_DAE_FROM_BASE}")
    print(f"  Model output dir: {save_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  Cache: {'enabled' if DVECTOR_CACHE_ENABLED else 'disabled'}")
    if DVECTOR_CACHE_ENABLED:
        print(f"  Cache dir: {cache_dir}")
        print(f"  Partial cache save every: {PARTIAL_CACHE_SAVE_EVERY} pairs")

    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

    for p in input_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Input overlap directory not found: {p}")

    if not musan_root.exists():
        raise FileNotFoundError(f"MUSAN speech root not found: {musan_root}")

    extraction_cache_config = {
        "source": "intermediate_noisy_clean_overlap_pairs",
        "overlap_dirs": [str(p.resolve()) for p in input_dirs],
        "musan_speech_noise_root": str(musan_root.resolve()),
        "n_overlap_samples_per_dir": N_OVERLAP_SAMPLES_PER_DIR,
        "sample_rate": SAMPLE_RATE,
        "snr_range_db": list(MUSAN_BABBLE_SNR_RANGE_DB),
        "skip_existing_musan_noise": SKIP_SAMPLES_WITH_EXISTING_MUSAN_NOISE,
        "random_seed": RANDOM_SEED,
    }

    pair_data = _extract_with_cache(
        cache_dir=cache_dir,
        cache_name="intermediate_babble_pairs",
        cache_config=extraction_cache_config,
        extractor_fn=lambda partial_save_fn=None, resume_data=None: extract_noisy_clean_dvector_pairs(
            overlap_dirs=input_dirs,
            musan_speech_noise_root=musan_root,
            n_samples_per_dir=N_OVERLAP_SAMPLES_PER_DIR,
            sample_rate=SAMPLE_RATE,
            device=DEVICE,
            snr_range_db=MUSAN_BABBLE_SNR_RANGE_DB,
            skip_existing_musan_noise=SKIP_SAMPLES_WITH_EXISTING_MUSAN_NOISE,
            random_seed=RANDOM_SEED,
            preview_limit=N_AUGMENTED_AUDIO_PREVIEW,
            partial_save_fn=partial_save_fn,
            partial_save_every=PARTIAL_CACHE_SAVE_EVERY,
            resume_data=resume_data,
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

    test_dataset = None
    if len(test_idx) > 0:
        test_dataset = NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, test_idx)

    # Only drop last batch when it would be a single-sample remainder (BatchNorm safety).
    train_drop_last = (len(train_dataset) % BATCH_SIZE) == 1
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=train_drop_last)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) if test_dataset else None

    print("\n[model] Loading base identity model")
    model, base_model_config = build_finetune_model(base_model_dir, DEVICE)
    print(f"[model] Fine-tune model class: {model.__class__.__name__}")

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

    # Keep config schema compatible with existing AE model directories.
    resolved_input_dim = int(base_model_config.get("input_dim", noisy_dvectors.shape[1]))
    resolved_hidden_dims = list(base_model_config.get("hidden_dims", [128, 64, 128]))
    resolved_dropout_rate = float(base_model_config.get("dropout_rate", 0.2))
    resolved_norm_type = str(base_model_config.get("norm_type", "batchnorm"))
    resolved_use_residual = bool(base_model_config.get("use_residual", False))
    resolved_residual_scale_init = float(base_model_config.get("residual_scale_init", 0.5))
    resolved_model_type = (
        "deep_stacked_dae" if USE_DEEP_STACKED_DAE else str(base_model_config.get("model_type", "dvector_autoencoder"))
    )
    resolved_n_stacked_daes = int(DEEP_STACKED_NUM_DAES) if USE_DEEP_STACKED_DAE else 1
    resolved_stack_refinement_hidden_dims = (
        [int(v) for v in DEEP_STACKED_REFINEMENT_HIDDEN_DIMS]
        if USE_DEEP_STACKED_DAE
        else []
    )

    plot_training_history(history, save_dir / "training_history.png")
    if eval_results is not None:
        plot_cosine_comparison(eval_results, save_dir / "cosine_comparison.png")

    config = {
        # Model architecture (standard keys expected by loaders/eval scripts)
        "model_type": resolved_model_type,
        "input_dim": resolved_input_dim,
        "hidden_dims": resolved_hidden_dims,
        "dropout_rate": resolved_dropout_rate,
        "norm_type": resolved_norm_type,
        "use_residual": resolved_use_residual,
        "residual_scale_init": resolved_residual_scale_init,
        "n_stacked_daes": resolved_n_stacked_daes,
        "stack_refinement_hidden_dims": resolved_stack_refinement_hidden_dims,

        "base_identity_model_dir": str(base_model_dir),
        "base_model_config": base_model_config,
        "deep_stacked_enabled": bool(USE_DEEP_STACKED_DAE),
        "deep_stacked_init_first_dae_from_base": bool(DEEP_STACKED_INIT_FIRST_DAE_FROM_BASE),
        "overlap_samples_dirs": [str(p) for p in input_dirs],
        "train_other_mix_dirs": [str(p) for p in input_dirs],
        "musan_speech_noise_root": str(musan_root),
        "musan_babble_snr_range_db": MUSAN_BABBLE_SNR_RANGE_DB,
        "skip_samples_with_existing_musan_noise": SKIP_SAMPLES_WITH_EXISTING_MUSAN_NOISE,
        "n_overlap_samples_per_dir": N_OVERLAP_SAMPLES_PER_DIR,
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
        "save_augmented_audio_preview": SAVE_AUGMENTED_AUDIO_PREVIEW,
        "n_augmented_audio_preview": N_AUGMENTED_AUDIO_PREVIEW,
        "augmented_audio_preview_dir": str(preview_dir),
        "n_pairs": n_pairs,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
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
