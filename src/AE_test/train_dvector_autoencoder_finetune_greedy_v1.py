#!/usr/bin/env python3
"""
Finetune a greedy-layerwise pretrained d-vector autoencoder on mixed pair types.

Inputs (each gated by a flag):
- Babble noise -> silence target
- Clean target + babble noise -> clean target
- Target overlap (target + other) -> clean target
- Non-target overlap (other + other) -> same overlap (identity)
- Clean target -> clean target (identity)
- Clean non-target -> clean non-target (identity)
"""

import builtins
import hashlib
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from resemblyzer import VoiceEncoder
from tqdm.auto import tqdm

from autoencoder_utils import (
    GreedyLayerwiseStackedDvectorAutoencoder,
    load_autoencoder,
)
from train_dvector_autoencoder import (
    create_training_pairs,
    discover_all_available_speakers,
    detect_speakers_from_overlap_samples,
    extract_noise_dvectors,
    extract_overlap_utterance_dvectors,
    extract_silence_dvectors,
    extract_single_speaker_dataset_dvectors,
    extract_single_utterance_dvectors,
)
from train_dvector_autoencoder_deep_stacked_libri import (
    discover_musan_speech_noise_files,
    load_noise_segment,
    mix_audio_with_snr,
)
from readable_dvector_cache import extract_with_cache as _readable_extract_with_cache


def print_ts(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args:
        first = args[0]
        if isinstance(first, str) and first.startswith("\n"):
            args = (f"\n[{timestamp}] {first[1:]}",) + args[1:]
        else:
            args = (f"[{timestamp}] {first}",) + args[1:]
    else:
        args = (f"[{timestamp}]",)
    return builtins.print(*args, **kwargs)


print = print_ts

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model init (greedy layerwise pretrain)
PRETRAINED_MODEL_PATH = "test_outputs/models/greedy/dvector_ae_greedy_layerwise_weight_sweep_15_22-5-26/mse0p0_cos0p6_neg0p4"
# PRETRAINED_MODEL_PATH = "test_outputs/models/stacked/dvector_ae_stacked_1_17-7-26"
# PRETRAINED_MODEL_PATH = ""
RANDOM_INIT_GREEDY_HIDDEN_DIMS = [192, 192]
RANDOM_INIT_DROPOUT_RATE = 0.1
RANDOM_INIT_NORM_TYPE = "layernorm"
RANDOM_INIT_ACTIVATION_TYPE = "tanh"

MODEL_SAVE_DIR = "test_outputs/models/greedy_finetune/121/v7test/11_26-7-26"

# Data sources
LIBRISPEECH_PATH = "../../data/LibriSpeech"
OVERLAP_TARGET_DIRS = [
    "test_outputs/data/121/121_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp_2000"
]
OVERLAP_NON_TARGET_DIRS = [
    "test_outputs/300Dev_5s_trainOther500_100pctAmp_40000"
]

# Single speaker dataset dirs for target speaker (optional override)
SINGLE_SPEAKER_DATASETS = [
    "../../data/speaker_121/train1",
    "../../data/speaker_121/train2",
    "../../data/speaker_121/train3",
    "../../data/speaker_121/train4",
    "../../data/speaker_121/train5",
    "../../data/speaker_121/train6",
    "../../data/speaker_121/train7",
    "../../data/speaker_121/train8",
    "../../data/speaker_121/train9",
    "../../data/speaker_121/train10",
    "../../data/speaker_121/train11",
    "../../data/speaker_121/train12",
    "../../data/speaker_121/train13",
    "../../data/speaker_121/train14",
    "../../data/speaker_121/train15",
    "../../data/speaker_121/train16",
    "../../data/speaker_121/train17",
    "../../data/speaker_121/train18",
    "../../data/speaker_121/train19",
    "../../data/speaker_121/train20",
    "../../data/speaker_121/train21",
    "../../data/speaker_121/train22",
    "../../data/speaker_121/train23",
    "../../data/speaker_121/train24",
    "../../data/speaker_121/train25",
]

# LibriSpeech subset(s) for clean non-target identity pairs
ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS = [
    # "train-clean-100",
    # "train-clean-360",
    "train-other-500",
]

# Speaker config (auto-detect when None)
MAIN_SPEAKERS = None
OTHER_SPEAKERS = None
AUTO_DETECT_OTHER_SPEAKERS = True
N_SINGLE_UTTERANCES_PER_SPEAKER = 50

# Noise data (babble -> silence)
MUSAN_ROOT = "../../kaldi/egs/pvad/musan"
MUSAN_SPEECH_NOISE_ROOT = "../../kaldi/egs/pvad/musan/musan_speech_train"
RIRS_ROOT = "../../kaldi/egs/pvad/RIRS_NOISES"
N_NOISE_SAMPLES_PER_TYPE = 100
N_RIR_SAMPLES = 0
MUSAN_BABBLE_SNR_RANGE_DB = (20.0, 17.0, 15.0, 13.0)
# MUSAN_BABBLE_SNR_RANGE_DB = [i for i in range(5, 21, 1)]

# Pair chunk duration
PAIR_CHUNK_DURATION_SEC = 1.6

# Debug sample saving
SAVE_DEBUG_PAIR_SAMPLES = True
DEBUG_PAIR_SAMPLES_PER_TYPE = 4
DEBUG_PAIR_DIR = "test_outputs/debug/greedy_finetune_pair_samples"
DEBUG_PAIR_SKIP_IF_EXISTS = False

# Pair caps / balance
MAX_PAIRS_PER_TYPE = 2000  # e.g., 5000 to cap each type
BALANCE_STRATEGY = "None"  # "min" or "none"

# Exact pair targets per enabled type.
# If available pairs are fewer than requested, samples are repeated randomly
# (with replacement) until the target count is reached.
PAIR_TARGET_COUNTS = {
    "clean_target_plus_noise": 2000,
    "overlap_target": 2000,
    "clean_target_identity": 2000,
    "overlap_non_target_self": 2000,
    "clean_non_target_identity": 0,
    "noise_to_silence": 100,
}


def _pair_target_count(pair_name):
    if pair_name not in PAIR_TARGET_COUNTS:
        return 0

    count = PAIR_TARGET_COUNTS.get(pair_name)
    if count is None:
        return None
    return int(count)


def _pair_enabled(pair_name):
    count = _pair_target_count(pair_name)
    if count is None:
        return True
    return count > 0

# Training
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
NUM_EPOCHS = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.05
EARLY_STOPPING_PATIENCE = 40

# Learning rate decay (ReduceLROnPlateau)
USE_LR_SCHEDULER = True
LR_DECAY_FACTOR = 0.5
LR_DECAY_PATIENCE = 4
LR_MIN_LR = 1e-7

# Smooth validation negative loss to reduce decision volatility (EMA alpha for previous)
VAL_NEG_EMA_ALPHA = 1

# Loss
LOSS_MSE_WEIGHT = 0
LOSS_COSINE_WEIGHT = 0.7
LOSS_COSINE_EPS = 1e-8
NEGATIVE_CONTRASTIVE_WEIGHT = 0.3
NEGATIVE_CONTRASTIVE_MARGIN = 0.2
USE_IN_BATCH_SHUFFLED_NEGATIVES = True
USE_NOISE_NEGATIVE_POOL = True
EXCLUDE_TARGET_PAIRS_FROM_NEGATIVE = True

# Misc
SAMPLE_RATE = 16000
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# D-vector extraction cache
DVECTOR_CACHE_ENABLED = True
DVECTOR_CACHE_DIR = "test_outputs/dvector_cache2"
DVECTOR_CACHE_VERSION = 1


class PairDataset(Dataset):
    def __init__(self, inputs, targets, is_target_pair):
        self.inputs = inputs
        self.targets = targets
        self.is_target_pair = is_target_pair

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx]).float(),
            torch.from_numpy(self.targets[idx]).float(),
            torch.tensor(self.is_target_pair[idx], dtype=torch.bool),
        )


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_list(values):
    if not values:
        return None

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


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


def _flatten_clean_dict(clean_by_speaker):
    flat = {}
    for speaker_id, dvecs in clean_by_speaker.items():
        flat.update(dvecs)
    return flat


def _discover_clean_audio_files(dataset_dirs):
    audio_files = []
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            continue
        for pattern in ("**/*.wav", "**/*.flac"):
            audio_files.extend([p for p in dataset_path.glob(pattern) if p.is_file()])
    return sorted(str(p) for p in audio_files)


def _load_librispeech_speaker_subset_map(librispeech_root):
    speakers_txt_path = Path(librispeech_root) / "SPEAKERS.TXT"
    if not speakers_txt_path.exists():
        return {}

    speaker_to_subset = {}
    try:
        with open(speakers_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 3:
                    continue
                speaker_id = parts[0]
                subset = parts[2]
                if speaker_id.isdigit() and subset:
                    speaker_to_subset[speaker_id] = subset
    except Exception:
        return {}

    return speaker_to_subset


def _discover_librispeech_speaker_files(librispeech_root, speaker_ids, subsets=None, max_files=None):
    audio_files = []
    for path_str in _iter_librispeech_speaker_files(
        librispeech_root,
        speaker_ids,
        subsets=subsets,
    ):
        audio_files.append(path_str)
        if max_files is not None and len(audio_files) >= int(max_files):
            break
    return sorted(audio_files)


def _iter_librispeech_speaker_files(librispeech_root, speaker_ids, subsets=None):
    root = Path(librispeech_root)
    if not root.exists():
        return iter(())

    subset_list = list(subsets) if subsets else None
    speaker_subset_map = {}
    if subset_list is None:
        speaker_subset_map = _load_librispeech_speaker_subset_map(root)

    default_subsets = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]

    def _iter_speaker_dirs(speaker_id):
        speaker_str = str(speaker_id)
        if subset_list:
            candidates = [root / subset / speaker_str for subset in subset_list]
        elif speaker_subset_map:
            subset = speaker_subset_map.get(speaker_str)
            if subset:
                candidates = [root / subset / speaker_str]
            else:
                candidates = [root / subset / speaker_str for subset in default_subsets]
        else:
            candidates = [root / subset / speaker_str for subset in default_subsets]

        for speaker_dir in candidates:
            if speaker_dir.exists():
                yield speaker_dir

    def _iter_utt_ids(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id = line.split()[0]
                    if utt_id:
                        yield utt_id
        except Exception:
            return

    def _iter_chapter_audio_files(chapter_dir):
        trans_files = list(chapter_dir.glob("*.trans.txt"))
        align_files = list(chapter_dir.glob("*.alignment.txt"))
        utt_iter = None
        if trans_files:
            utt_iter = _iter_utt_ids(trans_files[0])
        elif align_files:
            utt_iter = _iter_utt_ids(align_files[0])

        if utt_iter is not None:
            for utt_id in utt_iter:
                flac_path = chapter_dir / f"{utt_id}.flac"
                if flac_path.is_file():
                    yield str(flac_path)
                    continue
                wav_path = chapter_dir / f"{utt_id}.wav"
                if wav_path.is_file():
                    yield str(wav_path)
            return

        for ext in (".flac", ".wav"):
            for path in chapter_dir.glob(f"*{ext}"):
                if path.is_file():
                    yield str(path)

    def _generator():
        for speaker_id in speaker_ids:
            for speaker_dir in _iter_speaker_dirs(speaker_id):
                for chapter_dir in speaker_dir.iterdir():
                    if not chapter_dir.is_dir():
                        continue
                    for path_str in _iter_chapter_audio_files(chapter_dir):
                        yield path_str

    return _generator()


def _extract_clean_target_plus_noise_pairs(
    clean_audio_files,
    noise_files,
    sample_rate,
    chunk_duration_sec,
    snr_values_db,
    max_pairs,
    random_seed,
    device,
    debug_writer=None,
):
    if not clean_audio_files or not noise_files:
        return []

    if device == "cuda" and torch.cuda.is_available():
        encoder_device = torch.device("cuda")
    else:
        encoder_device = torch.device("cpu")

    encoder = VoiceEncoder(device=encoder_device)
    rng = random.Random(random_seed)
    chunk_size = int(sample_rate * chunk_duration_sec)
    snr_values_db = [float(v) for v in snr_values_db]

    target_pairs = int(max_pairs) if max_pairs is not None and int(max_pairs) > 0 else len(noise_files)
    pairs = []
    progress = tqdm(total=target_pairs, desc="clean+noise pairs", unit="pair")

    while len(pairs) < target_pairs:
        noise_path = rng.choice(noise_files)
        clean_path = rng.choice(clean_audio_files)
        clean_audio, _ = librosa.load(clean_path, sr=sample_rate)
        if len(clean_audio) < chunk_size:
            continue

        max_start = max(0, len(clean_audio) - chunk_size)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        clean_chunk = clean_audio[start:start + chunk_size]

        noise_segment = load_noise_segment(
            noise_path,
            target_samples=chunk_size,
            target_sr=sample_rate,
            rng=rng,
        )
        if noise_segment is None:
            continue

        snr_db = float(rng.choice(snr_values_db))
        noisy_chunk, _ = mix_audio_with_snr(clean_chunk, noise_segment, snr_db)

        if debug_writer is not None:
            debug_writer.try_write(
                "clean_target_plus_noise",
                noisy_chunk,
                clean_chunk,
                metadata={
                    "clean_source": clean_path,
                    "noise_source": noise_path,
                    "snr_db": snr_db,
                },
            )

        with torch.no_grad():
            clean_dvec = encoder.embed_utterance(clean_chunk).astype(np.float32)
            noisy_dvec = encoder.embed_utterance(noisy_chunk).astype(np.float32)

        pairs.append((noisy_dvec, clean_dvec))
        progress.update(1)

    progress.close()

    return pairs


def _sample_pairs(pairs, n_samples):
    if n_samples is None or n_samples < 0 or n_samples >= len(pairs):
        return pairs
    idx = np.random.choice(len(pairs), size=n_samples, replace=False)
    return [pairs[i] for i in idx]


def _resample_pairs_to_target(pairs, target_count):
    if target_count is None or target_count < 0:
        return pairs
    if len(pairs) == 0:
        return pairs

    target_count = int(target_count)
    if target_count == len(pairs):
        return pairs
    if target_count < len(pairs):
        return _sample_pairs(pairs, target_count)

    idx = np.random.choice(len(pairs), size=target_count, replace=True)
    return [pairs[i] for i in idx]


def _balance_pair_groups(groups, strategy, max_pairs_per_type=None, pair_target_counts=None):
    balanced = {}
    counts = {}
    pair_target_counts = pair_target_counts or {}

    for name, pairs in groups.items():
        target_count = pair_target_counts.get(name)
        if target_count is not None:
            pairs = _resample_pairs_to_target(pairs, target_count)
        if max_pairs_per_type is not None:
            pairs = _sample_pairs(pairs, max_pairs_per_type)
        balanced[name] = pairs
        counts[name] = len(pairs)

    if str(strategy).lower() != "min":
        return balanced, counts

    if not balanced:
        return balanced, counts

    min_count = min(len(pairs) for pairs in balanced.values())
    for name, pairs in balanced.items():
        balanced[name] = _sample_pairs(pairs, min_count)

    counts = {name: len(pairs) for name, pairs in balanced.items()}
    return balanced, counts


def _build_inputs_targets(pair_groups, target_pair_types=None):
    inputs = []
    targets = []
    is_target_pair = []
    target_pair_types = set(target_pair_types or [])
    for name, pairs in pair_groups.items():
        is_target = name in target_pair_types
        for inp, tgt in pairs:
            inputs.append(inp)
            targets.append(tgt)
            is_target_pair.append(is_target)
    return (
        np.stack(inputs).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.array(is_target_pair, dtype=bool),
    )


def _build_random_init_model_config(input_dim):
    return {
        "model_type": "greedy_layerwise_stacked",
        "input_dim": int(input_dim),
        "greedy_hidden_dims": list(RANDOM_INIT_GREEDY_HIDDEN_DIMS),
        "dropout_rate": float(RANDOM_INIT_DROPOUT_RATE),
        "norm_type": str(RANDOM_INIT_NORM_TYPE),
        "activation_type": str(RANDOM_INIT_ACTIVATION_TYPE),
    }


def _build_audio_index(audio_files):
    index = {}
    for path_str in audio_files or []:
        stem = Path(path_str).stem
        if stem not in index:
            index[stem] = path_str
    return index


def _chunk_audio(audio, sample_rate, chunk_duration_sec):
    chunk_size = int(sample_rate * chunk_duration_sec)
    if chunk_size <= 0 or audio is None:
        return []
    if len(audio) < chunk_size:
        return []

    chunks = []
    for start in range(0, len(audio) - chunk_size + 1, chunk_size):
        chunks.append(audio[start:start + chunk_size])
    return chunks


def _embed_chunks(encoder, chunks):
    if not chunks:
        return []

    dvectors = []
    with torch.no_grad():
        for chunk in chunks:
            dvectors.append(encoder.embed_utterance(chunk).astype(np.float32))
    return dvectors


def _save_wav_pair(output_dir, pair_type, idx, input_audio, target_audio, sample_rate, metadata=None):
    if input_audio is None or target_audio is None:
        return

    output_dir = Path(output_dir) / pair_type
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / f"{pair_type}_{idx:03d}_input.wav"
    target_path = output_dir / f"{pair_type}_{idx:03d}_target.wav"
    sf.write(input_path, input_audio, sample_rate)
    sf.write(target_path, target_audio, sample_rate)

    if metadata is not None:
        meta_path = output_dir / f"{pair_type}_{idx:03d}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


class _DebugPairWriter:
    def __init__(self, output_dir, samples_per_type, sample_rate):
        self.output_dir = Path(output_dir)
        self.samples_per_type = int(samples_per_type)
        self.sample_rate = sample_rate
        self.counts = {}

    def try_write(self, pair_type, input_audio, target_audio, metadata=None):
        if self.samples_per_type <= 0:
            return

        idx = self.counts.get(pair_type, 0)
        if idx >= self.samples_per_type:
            return

        _save_wav_pair(
            self.output_dir,
            pair_type,
            idx,
            input_audio,
            target_audio,
            self.sample_rate,
            metadata=metadata,
        )
        self.counts[pair_type] = idx + 1

def _extract_speech_noise_dvectors(
    encoder,
    musan_speech_root,
    n_noise_samples,
    sample_rate,
    noise_sample_duration,
    random_seed=42,
    debug_writer=None,
):
    noise_files = discover_musan_speech_noise_files(musan_speech_root)
    if not noise_files:
        return {}

    rng = random.Random(random_seed)
    rng.shuffle(noise_files)
    target_samples = int(sample_rate * noise_sample_duration)

    dvectors = {}
    for idx, noise_path in enumerate(noise_files[:n_noise_samples]):
        segment = load_noise_segment(
            noise_path,
            target_samples=target_samples,
            target_sr=sample_rate,
            rng=rng,
        )
        if segment is None:
            continue

        if debug_writer is not None:
            debug_writer.try_write(
                "noise_to_silence",
                segment,
                np.zeros_like(segment),
                metadata={"source": noise_path},
            )

        dvector = encoder.embed_utterance(segment).astype(np.float32)
        dvectors[f"speech_{idx:06d}"] = {
            "dvector": dvector,
            "type": "noise_speech",
            "source": noise_path,
        }

    return dvectors


def _build_identity_pairs_from_audio(
    encoder,
    audio_files,
    sample_rate,
    chunk_duration_sec,
    target_count=None,
    random_seed=42,
    pair_type=None,
    debug_writer=None,
):
    if not audio_files:
        return []

    rng = random.Random(random_seed)
    if isinstance(audio_files, (list, tuple)):
        files = list(audio_files)
        rng.shuffle(files)
    else:
        files = audio_files
    pairs = []

    for path_str in files:
        audio, _ = librosa.load(path_str, sr=sample_rate)
        chunks = _chunk_audio(audio, sample_rate, chunk_duration_sec)
        dvectors = _embed_chunks(encoder, chunks)
        for chunk, dvec in zip(chunks, dvectors):
            if debug_writer is not None and pair_type:
                debug_writer.try_write(
                    pair_type,
                    chunk,
                    chunk,
                    metadata={"source": path_str},
                )
            pairs.append((dvec, dvec))
            if target_count is not None and len(pairs) >= int(target_count):
                return pairs

    return pairs


def _build_overlap_target_pairs_from_audio(
    encoder,
    overlap_dvectors,
    overlap_pairs,
    clean_audio_index,
    sample_rate,
    chunk_duration_sec,
    target_count=None,
    debug_writer=None,
):
    pairs = []
    for overlap_key, clean_key in overlap_pairs:
        overlap_item = overlap_dvectors.get(overlap_key, {})
        source_dir = overlap_item.get("source_dir", "")
        output_name = overlap_item.get("output_name", "")
        if not source_dir or not output_name:
            continue

        overlap_path = Path(source_dir) / "audio" / f"{output_name}.wav"
        clean_path = clean_audio_index.get(clean_key)
        if not clean_path or not overlap_path.exists():
            continue

        overlap_audio, _ = librosa.load(str(overlap_path), sr=sample_rate)
        clean_audio, _ = librosa.load(clean_path, sr=sample_rate)
        overlap_chunks = _chunk_audio(overlap_audio, sample_rate, chunk_duration_sec)
        clean_chunks = _chunk_audio(clean_audio, sample_rate, chunk_duration_sec)
        n_chunks = min(len(overlap_chunks), len(clean_chunks))
        if n_chunks == 0:
            continue

        overlap_dvecs = _embed_chunks(encoder, overlap_chunks[:n_chunks])
        clean_dvecs = _embed_chunks(encoder, clean_chunks[:n_chunks])
        for overlap_chunk, clean_chunk, in_vec, out_vec in zip(
            overlap_chunks[:n_chunks],
            clean_chunks[:n_chunks],
            overlap_dvecs,
            clean_dvecs,
        ):
            if debug_writer is not None:
                debug_writer.try_write(
                    "overlap_target",
                    overlap_chunk,
                    clean_chunk,
                    metadata={
                        "overlap_key": overlap_key,
                        "overlap_audio": str(overlap_path),
                        "clean_key": clean_key,
                        "clean_audio": clean_path,
                    },
                )
            pairs.append((in_vec, out_vec))
            if target_count is not None and len(pairs) >= int(target_count):
                return pairs

    return pairs


def _build_non_target_overlap_pairs_from_audio(
    encoder,
    overlap_non_target_dirs,
    sample_rate,
    chunk_duration_sec,
    target_count=None,
    random_seed=42,
    debug_writer=None,
):
    if not overlap_non_target_dirs:
        return []

    rng = random.Random(random_seed)
    pairs = []

    for overlap_dir in overlap_non_target_dirs:
        metadata_path = Path(overlap_dir) / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        rng.shuffle(metadata_list)
        for metadata in metadata_list:
            output_name = metadata.get("output_name", "")
            if output_name and not output_name.endswith(".wav"):
                output_name = f"{output_name}.wav"
            audio_path = Path(overlap_dir) / "audio" / output_name
            if not audio_path.exists():
                continue

            audio, _ = librosa.load(str(audio_path), sr=sample_rate)
            chunks = _chunk_audio(audio, sample_rate, chunk_duration_sec)
            dvectors = _embed_chunks(encoder, chunks)
            for chunk, dvec in zip(chunks, dvectors):
                if debug_writer is not None:
                    debug_writer.try_write(
                        "overlap_non_target_self",
                        chunk,
                        chunk,
                        metadata={"source": str(audio_path)},
                    )
                pairs.append((dvec, dvec))
                if target_count is not None and len(pairs) >= int(target_count):
                    return pairs

    return pairs


def _train_epoch(model, loader, optimizer, noise_negative_pool=None):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    total_neg = 0.0

    logged_mask = False
    for batch_idx, (noisy, clean, is_target_pair) in enumerate(loader):
        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)
        is_target_pair = is_target_pair.to(DEVICE)
        clean_norm = F.normalize(clean, p=2, dim=1, eps=LOSS_COSINE_EPS)

        optimizer.zero_grad()
        recon = model(noisy)
        mse = F.mse_loss(recon, clean_norm)
        cos = 1.0 - F.cosine_similarity(recon, clean_norm, dim=1, eps=LOSS_COSINE_EPS).mean()

        # if noisy.size(0) > 1:
        #     shuffle = torch.randperm(noisy.size(0), device=noisy.device)
        #     neg_target = clean_norm[shuffle]
        #     neg_cos = F.cosine_similarity(recon, neg_target, dim=1, eps=LOSS_COSINE_EPS)
        #     neg = F.relu(neg_cos - NEGATIVE_CONTRASTIVE_MARGIN).mean()
        # else:
        #     neg = torch.tensor(0.0, device=noisy.device)
        neg = torch.tensor(0.0, device=noisy.device)

        if USE_IN_BATCH_SHUFFLED_NEGATIVES and noisy.size(0) > 1:
            shuffle = torch.randperm(noisy.size(0), device=noisy.device)
            neg_target = clean_norm[shuffle]

            neg_cos = F.cosine_similarity(
                recon,
                neg_target,
                dim=1,
                eps=LOSS_COSINE_EPS,
            )

            neg_vals = F.relu(neg_cos - NEGATIVE_CONTRASTIVE_MARGIN)
            if EXCLUDE_TARGET_PAIRS_FROM_NEGATIVE:
                neg_mask = (~is_target_pair).float()
                valid = neg_mask.sum()
                if valid > 0:
                    neg = neg + (neg_vals * neg_mask).sum() / valid
                if not logged_mask and batch_idx == 0:
                    masked = int((is_target_pair).sum().item())
                    print(f"[neg] In-batch masked {masked}/{noisy.size(0)} target-pair samples")
                    logged_mask = True
            else:
                neg = neg + neg_vals.mean()

        if noise_negative_pool is not None and noise_negative_pool.numel() > 0:
            noise_idx = torch.randint(0, noise_negative_pool.size(0), (noisy.size(0),), device=noisy.device)
            noise_target = noise_negative_pool[noise_idx]
            noise_target = F.normalize(noise_target, p=2, dim=1, eps=LOSS_COSINE_EPS)
            noise_cos = F.cosine_similarity(recon, noise_target, dim=1, eps=LOSS_COSINE_EPS)
            noise_vals = F.relu(noise_cos - NEGATIVE_CONTRASTIVE_MARGIN)
            if EXCLUDE_TARGET_PAIRS_FROM_NEGATIVE:
                noise_mask = (~is_target_pair).float()
                # noise_mask = (is_target_pair).float()
                valid = noise_mask.sum()
                if valid > 0:
                    neg = neg + (noise_vals * noise_mask).sum() / valid
                if not logged_mask and batch_idx == 0:
                    masked = int((is_target_pair).sum().item())
                    print(f"[neg] Noise-pool masked {masked}/{noisy.size(0)} target-pair samples")
                    logged_mask = True
            else:
                neg = neg + noise_vals.mean()

        loss = (LOSS_MSE_WEIGHT * mse) + (LOSS_COSINE_WEIGHT * cos) + (NEGATIVE_CONTRASTIVE_WEIGHT * neg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_cos += cos.item()
        total_neg += neg.item()

    n_batches = max(1, len(loader))
    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches, total_neg / n_batches


def _eval_epoch(model, loader, noise_negative_pool=None):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    total_neg = 0.0

    logged_mask = False
    with torch.no_grad():
        for batch_idx, (noisy, clean, is_target_pair) in enumerate(loader):
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)
            is_target_pair = is_target_pair.to(DEVICE)
            clean_norm = F.normalize(clean, p=2, dim=1, eps=LOSS_COSINE_EPS)

            recon = model(noisy)
            mse = F.mse_loss(recon, clean_norm)
            cos = 1.0 - F.cosine_similarity(recon, clean_norm, dim=1, eps=LOSS_COSINE_EPS).mean()

            # if noisy.size(0) > 1:
            #     shuffle = torch.randperm(noisy.size(0), device=noisy.device)
            #     neg_target = clean_norm[shuffle]
            #     neg_cos = F.cosine_similarity(recon, neg_target, dim=1, eps=LOSS_COSINE_EPS)
            #     neg = F.relu(neg_cos - NEGATIVE_CONTRASTIVE_MARGIN).mean()
            # else:
            #     neg = torch.tensor(0.0, device=noisy.device)
            neg = torch.tensor(0.0, device=noisy.device)

            if USE_IN_BATCH_SHUFFLED_NEGATIVES and noisy.size(0) > 1:
                shuffle = torch.randperm(noisy.size(0), device=noisy.device)
                neg_target = clean_norm[shuffle]

                neg_cos = F.cosine_similarity(
                    recon,
                    neg_target,
                    dim=1,
                    eps=LOSS_COSINE_EPS,
                )

                neg_vals = F.relu(neg_cos - NEGATIVE_CONTRASTIVE_MARGIN)
                if EXCLUDE_TARGET_PAIRS_FROM_NEGATIVE:
                    neg_mask = (~is_target_pair).float()
                    valid = neg_mask.sum()
                    if valid > 0:
                        neg = neg + (neg_vals * neg_mask).sum() / valid
                    if not logged_mask and batch_idx == 0:
                        masked = int((is_target_pair).sum().item())
                        print(f"[neg] In-batch masked {masked}/{noisy.size(0)} target-pair samples (val)")
                        logged_mask = True
                else:
                    neg = neg + neg_vals.mean()

            if noise_negative_pool is not None and noise_negative_pool.numel() > 0:
                noise_idx = torch.randint(0, noise_negative_pool.size(0), (noisy.size(0),), device=noisy.device)
                noise_target = noise_negative_pool[noise_idx]
                noise_target = F.normalize(noise_target, p=2, dim=1, eps=LOSS_COSINE_EPS)
                noise_cos = F.cosine_similarity(recon, noise_target, dim=1, eps=LOSS_COSINE_EPS)
                noise_vals = F.relu(noise_cos - NEGATIVE_CONTRASTIVE_MARGIN)
                if EXCLUDE_TARGET_PAIRS_FROM_NEGATIVE:
                    noise_mask = (~is_target_pair).float()
                    # noise_mask = (is_target_pair).float()
                    valid = noise_mask.sum()
                    if valid > 0:
                        neg = neg + (noise_vals * noise_mask).sum() / valid
                    if not logged_mask and batch_idx == 0:
                        masked = int((is_target_pair).sum().item())
                        print(f"[neg] Noise-pool masked {masked}/{noisy.size(0)} target-pair samples (val)")
                        logged_mask = True
                else:
                    neg = neg + noise_vals.mean()

            loss = (LOSS_MSE_WEIGHT * mse) + (LOSS_COSINE_WEIGHT * cos) + (NEGATIVE_CONTRASTIVE_WEIGHT * neg)

            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()
            total_neg += neg.item()

    n_batches = max(1, len(loader))
    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches, total_neg / n_batches


def _plot_loss_history(history, save_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2.0)
    axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2.0)
    axes[0].set_title("Total Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["train_mse"], label="Train MSE")
    axes[1].plot(epochs, history["val_mse"], label="Val MSE")
    axes[1].plot(epochs, history["train_cos"], label="Train Cos")
    axes[1].plot(epochs, history["val_cos"], label="Val Cos")
    axes[1].plot(epochs, history["train_neg"], label="Train Neg")
    axes[1].plot(epochs, history["val_neg"], label="Val Neg")
    axes[1].set_title("Loss Components by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    _set_seed(RANDOM_SEED)
    script_dir = Path(__file__).parent

    save_dir = Path(MODEL_SAVE_DIR)
    if not save_dir.is_absolute():
        save_dir = script_dir / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(DVECTOR_CACHE_DIR)
    if not cache_dir.is_absolute():
        cache_dir = script_dir / cache_dir
    if DVECTOR_CACHE_ENABLED:
        cache_dir.mkdir(parents=True, exist_ok=True)

    librispeech_path = Path(LIBRISPEECH_PATH)
    if not librispeech_path.is_absolute():
        librispeech_path = script_dir / librispeech_path

    overlap_target_dirs = [str(Path(d)) for d in OVERLAP_TARGET_DIRS]
    overlap_non_target_dirs = [str(Path(d)) for d in OVERLAP_NON_TARGET_DIRS]

    if MAIN_SPEAKERS is None or (OTHER_SPEAKERS is None and AUTO_DETECT_OTHER_SPEAKERS):
        detected_main, detected_other = detect_speakers_from_overlap_samples(overlap_target_dirs)
        main_speakers = MAIN_SPEAKERS if MAIN_SPEAKERS is not None else detected_main
        other_speakers = OTHER_SPEAKERS if OTHER_SPEAKERS is not None else detected_other
    else:
        main_speakers = MAIN_SPEAKERS
        other_speakers = OTHER_SPEAKERS or []

    if not main_speakers:
        raise ValueError("No main speakers detected or configured")

    print(f"\n[config] Main speakers: {main_speakers}")
    print(f"[config] Overlap-derived other speakers: {len(other_speakers)}")

    print("\n[data] Loading overlap d-vectors (target overlaps)...")
    overlap_cache_config = {
        "source": "overlap_target",
        "overlap_dirs": overlap_target_dirs,
        "sample_rate": SAMPLE_RATE,
        "device": DEVICE,
    }
    overlap_dvectors = _extract_with_cache(
        cache_dir,
        "overlap_target_dvectors",
        overlap_cache_config,
        lambda: extract_overlap_utterance_dvectors(
            overlap_target_dirs, sample_rate=SAMPLE_RATE, device=DEVICE
        ),
    )

    clean_main_dvectors_dict = {}
    if SINGLE_SPEAKER_DATASETS:
        print("\n[data] Loading clean target d-vectors from single speaker datasets...")
        clean_target_cache_config = {
            "source": "clean_target_single_speaker",
            "datasets": SINGLE_SPEAKER_DATASETS,
            "n_utterances_per_speaker": N_SINGLE_UTTERANCES_PER_SPEAKER,
            "sample_rate": SAMPLE_RATE,
            "device": DEVICE,
        }
        clean_main_dvectors_dict = _extract_with_cache(
            cache_dir,
            "clean_target_single_speaker_dvectors",
            clean_target_cache_config,
            lambda: extract_single_speaker_dataset_dvectors(
                SINGLE_SPEAKER_DATASETS,
                n_utterances_per_speaker=N_SINGLE_UTTERANCES_PER_SPEAKER,
                sample_rate=SAMPLE_RATE,
                device=DEVICE,
            ),
        )
    else:
        print("\n[data] Loading clean target d-vectors from LibriSpeech...")
        clean_target_cache_config = {
            "source": "clean_target_librispeech",
            "librispeech_path": str(librispeech_path.resolve()),
            "speaker_ids": list(main_speakers),
            "n_utterances_per_speaker": N_SINGLE_UTTERANCES_PER_SPEAKER,
            "sample_rate": SAMPLE_RATE,
            "device": DEVICE,
        }
        clean_main_dvectors_dict = _extract_with_cache(
            cache_dir,
            "clean_target_librispeech_dvectors",
            clean_target_cache_config,
            lambda: extract_single_utterance_dvectors(
                str(librispeech_path),
                main_speakers,
                N_SINGLE_UTTERANCES_PER_SPEAKER,
                SAMPLE_RATE,
                DEVICE,
            ),
        )

    clean_main_dvectors = _flatten_clean_dict(clean_main_dvectors_dict)
    if len(clean_main_dvectors) == 0:
        raise ValueError("No clean target d-vectors found")

    pair_groups = {}
    noise_negative_pool = None
    overlap_pairs = []
    non_target_pairs = []
    clean_non_target_pairs = []

    debug_writer = None
    if SAVE_DEBUG_PAIR_SAMPLES:
        debug_dir = Path(DEBUG_PAIR_DIR)
        if DEBUG_PAIR_SKIP_IF_EXISTS and debug_dir.exists() and list(debug_dir.rglob("*.wav")):
            print(f"[debug] Skipping pair sample export; existing files found in {debug_dir}")
        else:
            debug_writer = _DebugPairWriter(
                output_dir=DEBUG_PAIR_DIR,
                samples_per_type=DEBUG_PAIR_SAMPLES_PER_TYPE,
                sample_rate=SAMPLE_RATE,
            )

    if SINGLE_SPEAKER_DATASETS:
        clean_audio_files = _discover_clean_audio_files(SINGLE_SPEAKER_DATASETS)
    else:
        clean_audio_files = _discover_librispeech_speaker_files(librispeech_path, main_speakers)
    clean_audio_index = _build_audio_index(clean_audio_files)

    if DEVICE == "cuda" and torch.cuda.is_available():
        pair_encoder_device = torch.device("cuda")
    else:
        pair_encoder_device = torch.device("cpu")
    pair_encoder = VoiceEncoder(device=pair_encoder_device)

    if _pair_enabled("clean_target_plus_noise"):
        print("\n[pairs] Building clean target + babble noise -> clean pairs...")
        if not clean_audio_files:
            raise ValueError("No clean target audio files found for clean+noise pairs")

        noise_files = discover_musan_speech_noise_files(MUSAN_SPEECH_NOISE_ROOT)
        if not noise_files:
            raise ValueError("No MUSAN speech noise files found for clean+noise pairs")

        clean_noise_cache_config = {
            "source": "clean_target_plus_noise",
            "clean_datasets": list(SINGLE_SPEAKER_DATASETS) if SINGLE_SPEAKER_DATASETS else [],
            "clean_speakers": list(main_speakers),
            "musan_speech_root": str(MUSAN_SPEECH_NOISE_ROOT),
            "sample_rate": SAMPLE_RATE,
            "chunk_duration_sec": PAIR_CHUNK_DURATION_SEC,
            "snr_range_db": list(MUSAN_BABBLE_SNR_RANGE_DB),
            "pair_target_count": PAIR_TARGET_COUNTS.get("clean_target_plus_noise"),
            "random_seed": RANDOM_SEED,
            "device": DEVICE,
        }

        clean_noise_pairs = _extract_with_cache(
            cache_dir,
            "clean_target_plus_noise_pairs",
            clean_noise_cache_config,
            lambda: _extract_clean_target_plus_noise_pairs(
                clean_audio_files=clean_audio_files,
                noise_files=noise_files,
                sample_rate=SAMPLE_RATE,
                chunk_duration_sec=PAIR_CHUNK_DURATION_SEC,
                snr_values_db=MUSAN_BABBLE_SNR_RANGE_DB,
                max_pairs=_pair_target_count("clean_target_plus_noise"),
                random_seed=RANDOM_SEED,
                device=DEVICE,
                debug_writer=debug_writer,
            ),
        )

        pair_groups["clean_target_plus_noise"] = clean_noise_pairs

    if _pair_enabled("overlap_target"):
        print("\n[pairs] Building target overlap -> clean target pairs...")
        overlap_pairs = create_training_pairs(overlap_dvectors, clean_main_dvectors, main_speakers, n_pairs_per_dir=-1)
        overlap_pairs_cache_config = {
            "source": "overlap_target_chunked_pairs",
            "overlap_dirs": list(overlap_target_dirs),
            "overlap_pairs_hash": _hash_list([f"{ok}:{ck}" for ok, ck in overlap_pairs]),
            "clean_audio_hash": _hash_list(clean_audio_files),
            "sample_rate": SAMPLE_RATE,
            "chunk_duration_sec": PAIR_CHUNK_DURATION_SEC,
            "pair_target_count": _pair_target_count("overlap_target"),
        }
        pair_groups["overlap_target"] = _extract_with_cache(
            cache_dir,
            "overlap_target_chunked_pairs",
            overlap_pairs_cache_config,
            lambda: _build_overlap_target_pairs_from_audio(
                pair_encoder,
                overlap_dvectors,
                overlap_pairs,
                clean_audio_index,
                SAMPLE_RATE,
                PAIR_CHUNK_DURATION_SEC,
                target_count=_pair_target_count("overlap_target"),
                debug_writer=debug_writer,
            ),
        )

    if _pair_enabled("overlap_non_target_self") and overlap_non_target_dirs:
        print("\n[pairs] Building non-target overlap -> self pairs...")
        non_target_pairs_cache_config = {
            "source": "overlap_non_target_chunked_pairs",
            "overlap_dirs": list(overlap_non_target_dirs),
            "sample_rate": SAMPLE_RATE,
            "chunk_duration_sec": PAIR_CHUNK_DURATION_SEC,
            "pair_target_count": _pair_target_count("overlap_non_target_self"),
            "random_seed": RANDOM_SEED,
        }
        non_target_pairs = _extract_with_cache(
            cache_dir,
            "overlap_non_target_chunked_pairs",
            non_target_pairs_cache_config,
            lambda: _build_non_target_overlap_pairs_from_audio(
                pair_encoder,
                overlap_non_target_dirs,
                SAMPLE_RATE,
                PAIR_CHUNK_DURATION_SEC,
                target_count=_pair_target_count("overlap_non_target_self"),
                random_seed=RANDOM_SEED,
                debug_writer=debug_writer,
            ),
        )
        pair_groups["overlap_non_target_self"] = non_target_pairs

    if _pair_enabled("clean_target_identity"):
        print("\n[pairs] Building clean target identity pairs...")
        target_count = _pair_target_count("clean_target_identity")
        clean_target_cache_config = {
            "source": "clean_target_identity_chunked_pairs",
            "clean_audio_hash": _hash_list(clean_audio_files),
            "sample_rate": SAMPLE_RATE,
            "chunk_duration_sec": PAIR_CHUNK_DURATION_SEC,
            "pair_target_count": target_count,
            "random_seed": RANDOM_SEED,
        }
        pair_groups["clean_target_identity"] = _extract_with_cache(
            cache_dir,
            "clean_target_identity_chunked_pairs",
            clean_target_cache_config,
            lambda: _build_identity_pairs_from_audio(
                pair_encoder,
                clean_audio_files,
                SAMPLE_RATE,
                PAIR_CHUNK_DURATION_SEC,
                target_count=target_count,
                random_seed=RANDOM_SEED,
                pair_type="clean_target_identity",
                debug_writer=debug_writer,
            ),
        )

    if _pair_enabled("clean_non_target_identity"):
        print("\n[pairs] Discovering clean non-target speakers from LibriSpeech...")
        clean_non_target_speakers = discover_all_available_speakers(
            librispeech_path,
            exclude_speakers=set(main_speakers or []),
            librispeech_subsets=ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS,
        )
        if not clean_non_target_speakers:
            print("\n[pairs] No non-target speakers available for clean identity pairs")
        else:
            print("\n[pairs] Building clean non-target identity pairs...")
            target_count = _pair_target_count("clean_non_target_identity")
            other_audio_files = _iter_librispeech_speaker_files(
                librispeech_path,
                clean_non_target_speakers,
                subsets=ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS,
            )
            audio_count = sum(1 for _ in other_audio_files)
            print(
                "[pairs] Non-target speaker count: "
                f"{len(clean_non_target_speakers)} | audio files found: {audio_count} | "
                f"subsets: {ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS}"
            )
            other_audio_files = _iter_librispeech_speaker_files(
                librispeech_path,
                clean_non_target_speakers,
                subsets=ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS,
            )
            subset_tokens = [f"subset:{name}" for name in ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS]
            speaker_tokens = [f"speaker:{sid}" for sid in sorted(clean_non_target_speakers or [])]
            clean_non_target_cache_config = {
                "source": "clean_non_target_identity_chunked_pairs",
                "clean_audio_hash": _hash_list(speaker_tokens + subset_tokens),
                "librispeech_subsets": list(ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS),
                "sample_rate": SAMPLE_RATE,
                "chunk_duration_sec": PAIR_CHUNK_DURATION_SEC,
                "pair_target_count": target_count,
                "random_seed": RANDOM_SEED,
            }
            clean_non_target_pairs = _extract_with_cache(
                cache_dir,
                "clean_non_target_identity_chunked_pairs",
                clean_non_target_cache_config,
                lambda: _build_identity_pairs_from_audio(
                    pair_encoder,
                    other_audio_files,
                    SAMPLE_RATE,
                    PAIR_CHUNK_DURATION_SEC,
                    target_count=target_count,
                    random_seed=RANDOM_SEED,
                    pair_type="clean_non_target_identity",
                    debug_writer=debug_writer,
                ),
            )
            if not clean_non_target_pairs:
                print("\n[pairs] Warning: no audio found for clean non-target identity pairs")
            else:
                pair_groups["clean_non_target_identity"] = clean_non_target_pairs

    if _pair_enabled("noise_to_silence"):
        print("\n[pairs] Building babble noise -> silence pairs...")
        silence_cache_config = {
            "source": "silence_identity",
            "n_samples": max(200, N_NOISE_SAMPLES_PER_TYPE),
            "duration_per_sample": PAIR_CHUNK_DURATION_SEC,
            "sample_rate": SAMPLE_RATE,
            "device": DEVICE,
        }
        silence_dvectors = _extract_with_cache(
            cache_dir,
            "silence_dvectors",
            silence_cache_config,
            lambda: extract_silence_dvectors(
                n_samples=max(200, N_NOISE_SAMPLES_PER_TYPE),
                duration_per_sample=PAIR_CHUNK_DURATION_SEC,
                sample_rate=SAMPLE_RATE,
                device=DEVICE,
            ),
        )
        noise_cache_config = {
            "source": "speech_noise_identity",
            "musan_speech_root": MUSAN_SPEECH_NOISE_ROOT,
            "n_noise_samples": N_NOISE_SAMPLES_PER_TYPE,
            "noise_sample_duration": PAIR_CHUNK_DURATION_SEC,
            "sample_rate": SAMPLE_RATE,
            "random_seed": RANDOM_SEED,
        }
        noise_dvectors = _extract_with_cache(
            cache_dir,
            "speech_noise_dvectors",
            noise_cache_config,
            lambda: _extract_speech_noise_dvectors(
                pair_encoder,
                MUSAN_SPEECH_NOISE_ROOT,
                N_NOISE_SAMPLES_PER_TYPE,
                SAMPLE_RATE,
                PAIR_CHUNK_DURATION_SEC,
                random_seed=RANDOM_SEED,
                debug_writer=debug_writer,
            ),
        )
        silence_list = list(silence_dvectors.values())
        noise_pairs = []
        for noise in noise_dvectors.values():
            silence = random.choice(silence_list)
            noise_pairs.append((noise["dvector"], silence["dvector"]))
        pair_groups["noise_to_silence"] = noise_pairs

        if USE_NOISE_NEGATIVE_POOL:
            negatives = []
            if 'noise_dvectors' in locals() and noise_dvectors:
                negatives.extend([v["dvector"] for v in noise_dvectors.values()])
            if clean_non_target_pairs:
                negatives.extend([p[0] for p in clean_non_target_pairs])
            if non_target_pairs:
                negatives.extend([p[0] for p in non_target_pairs])

            if negatives:
                neg_pool = np.stack(negatives).astype(np.float32)
                noise_negative_pool = torch.from_numpy(neg_pool).to(DEVICE)
            else:
                noise_negative_pool = None

    if pair_groups:
        print("\n[pairs] Raw pair counts (pre-balance):")
        for name, pairs in pair_groups.items():
            print(f"  - {name}: {len(pairs)}")

    enabled_groups = {k: v for k, v in pair_groups.items() if v}
    if not enabled_groups:
        raise ValueError("No training pairs created. Check flags and data paths.")

    enabled_groups, counts = _balance_pair_groups(
        enabled_groups,
        strategy=BALANCE_STRATEGY,
        max_pairs_per_type=MAX_PAIRS_PER_TYPE,
        pair_target_counts=PAIR_TARGET_COUNTS,
    )

    print("\n[pairs] Pair counts after balance:")
    for name, count in counts.items():
        print(f"  - {name}: {count}")

    target_pair_types = {
        name for name in ("clean_target_plus_noise", "overlap_target", "clean_target_identity")
        if _pair_enabled(name)
    }
    inputs, targets, is_target_pair = _build_inputs_targets(enabled_groups, target_pair_types)

    (
        train_inputs,
        temp_inputs,
        train_targets,
        temp_targets,
        train_flags,
        temp_flags,
    ) = train_test_split(
        inputs,
        targets,
        is_target_pair,
        test_size=VALIDATION_SPLIT + TEST_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    if TEST_SPLIT > 0:
        val_ratio = VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)
        (
            val_inputs,
            test_inputs,
            val_targets,
            test_targets,
            val_flags,
            test_flags,
        ) = train_test_split(
            temp_inputs,
            temp_targets,
            temp_flags,
            test_size=1.0 - val_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    else:
        val_inputs, val_targets, val_flags = temp_inputs, temp_targets, temp_flags
        test_inputs = np.empty((0, inputs.shape[1]))
        test_targets = np.empty((0, targets.shape[1]))
        test_flags = np.empty((0,), dtype=bool)

    train_loader = DataLoader(
        PairDataset(train_inputs, train_targets, train_flags),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        PairDataset(val_inputs, val_targets, val_flags),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    pretrained_model_path = str(PRETRAINED_MODEL_PATH).strip()
    if pretrained_model_path:
        print("\n[model] Loading greedy layerwise pretrained model...")
        model, model_config = load_autoencoder(pretrained_model_path, DEVICE)
        model = model.to(DEVICE)
    else:
        input_dim = int(train_inputs.shape[1])
        model_config = _build_random_init_model_config(input_dim)
        print("\n[model] No pretrained model path provided; initializing random weights...")
        print(
            "  Architecture: greedy_layerwise_stacked | "
            f"input_dim={model_config['input_dim']} | "
            f"greedy_hidden_dims={model_config['greedy_hidden_dims']} | "
            f"dropout={model_config['dropout_rate']} | "
            f"norm={model_config['norm_type']} | "
            f"activation={model_config['activation_type']}"
        )
        model = GreedyLayerwiseStackedDvectorAutoencoder(
            input_dim=model_config["input_dim"],
            greedy_hidden_dims=model_config["greedy_hidden_dims"],
            dropout_rate=model_config["dropout_rate"],
            norm_type=model_config["norm_type"],
            activation_type=model_config["activation_type"],
        ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = None
    if USE_LR_SCHEDULER:
        try:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=LR_DECAY_FACTOR,
                patience=LR_DECAY_PATIENCE,
                min_lr=LR_MIN_LR,
                verbose=True,
            )
        except Exception:
            scheduler = None

    best_val = float("inf")
    patience = 0
    best_path = save_dir / "best_model.pth"
    loss_plot_path = save_dir / "training_loss_by_epoch.png"
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "train_cos": [],
        "val_cos": [],
        "train_neg": [],
        "val_neg": [],
    }

    # Smoothed validation negative component to reduce volatility in decisions
    smoothed_val_neg = None

    for epoch in range(NUM_EPOCHS):
        train_loss, train_mse, train_cos, train_neg = _train_epoch(
            model, train_loader, optimizer, noise_negative_pool=noise_negative_pool
        )
        val_loss, val_mse, val_cos, val_neg = _eval_epoch(
            model, val_loader, noise_negative_pool=noise_negative_pool
        )

        # current learning rate (first param group)
        try:
            current_lr = optimizer.param_groups[0].get("lr", float("nan"))
        except Exception:
            current_lr = float("nan")

        print(
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | LR {current_lr:.1e} | "
            f"Train {train_loss:.6f} (mse {train_mse:.6f}, cos {train_cos:.6f}, neg {train_neg:.6f}) | "
            f"Val {val_loss:.6f} (mse {val_mse:.6f}, cos {val_cos:.6f}, neg {val_neg:.6f})"
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_mse"].append(float(train_mse))
        history["val_mse"].append(float(val_mse))
        history["train_cos"].append(float(train_cos))
        history["val_cos"].append(float(val_cos))
        history["train_neg"].append(float(train_neg))
        history["val_neg"].append(float(val_neg))

        # Update EMA for validation negative component
        if smoothed_val_neg is None:
            smoothed_val_neg = float(val_neg)
        else:
            smoothed_val_neg = float(VAL_NEG_EMA_ALPHA) * smoothed_val_neg + (1.0 - float(VAL_NEG_EMA_ALPHA)) * float(val_neg)

        # Compute adjusted validation loss using smoothed negative term
        adjusted_val_loss = (
            LOSS_MSE_WEIGHT * float(val_mse)
            + LOSS_COSINE_WEIGHT * float(val_cos)
            + NEGATIVE_CONTRASTIVE_WEIGHT * float(smoothed_val_neg)
        )

        # Use adjusted (smoothed-neg) validation loss for best-model selection
        if adjusted_val_loss < best_val:
            best_val = adjusted_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": float(val_loss),
                    "adjusted_val_loss": float(adjusted_val_loss),
                    "smoothed_val_neg": float(smoothed_val_neg),
                    "epoch": epoch + 1,
                },
                best_path,
            )
            patience = 0
        else:
            patience += 1

        # Step LR scheduler based on validation loss
        # Step LR scheduler based on adjusted (smoothed-neg) validation loss
        if scheduler is not None:
            try:
                scheduler.step(adjusted_val_loss)
            except Exception:
                pass

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    final_path = save_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    _plot_loss_history(history, loss_plot_path)

    config = {
        "pretrained_model_path": PRETRAINED_MODEL_PATH,
        "model_config": model_config,
        "input_dim": int(model_config.get("input_dim", 256)),
        "model_type": model_config.get("model_type", "greedy_layerwise_stacked"),
        "greedy_hidden_dims": model_config.get("greedy_hidden_dims", []),
        "dropout_rate": float(model_config.get("dropout_rate", 0.1)),
        "norm_type": model_config.get("norm_type", "layernorm"),
        "activation_type": model_config.get("activation_type", "tanh"),
        "pair_counts": counts,
        "pair_target_counts": PAIR_TARGET_COUNTS,
        "balance_strategy": BALANCE_STRATEGY,
        "max_pairs_per_type": MAX_PAIRS_PER_TYPE,
        "include_clean_target_plus_noise": _pair_enabled("clean_target_plus_noise"),
        "include_noise_to_silence": _pair_enabled("noise_to_silence"),
        "include_target_overlap": _pair_enabled("overlap_target"),
        "include_non_target_overlap": _pair_enabled("overlap_non_target_self"),
        "include_clean_target": _pair_enabled("clean_target_identity"),
        "include_clean_non_target": _pair_enabled("clean_non_target_identity"),
        "musan_speech_noise_root": MUSAN_SPEECH_NOISE_ROOT,
        "musan_babble_snr_range_db": MUSAN_BABBLE_SNR_RANGE_DB,
        "pair_chunk_duration_sec": PAIR_CHUNK_DURATION_SEC,
        "save_debug_pair_samples": SAVE_DEBUG_PAIR_SAMPLES,
        "debug_pair_samples_per_type": DEBUG_PAIR_SAMPLES_PER_TYPE,
        "debug_pair_dir": DEBUG_PAIR_DIR,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "loss_mse_weight": LOSS_MSE_WEIGHT,
        "loss_cosine_weight": LOSS_COSINE_WEIGHT,
        "loss_negative_weight": NEGATIVE_CONTRASTIVE_WEIGHT,
        "loss_negative_margin": NEGATIVE_CONTRASTIVE_MARGIN,
        "random_seed": RANDOM_SEED,
    }

    with open(save_dir / "config.pkl", "wb") as f:
        pickle.dump(config, f)

    summary_path = save_dir / "config_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FINETUNE GREEDY D-VECTOR AE (MIXED PAIRS)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Pretrained model: {PRETRAINED_MODEL_PATH}\n")
        f.write(f"Balance strategy: {BALANCE_STRATEGY}\n")
        f.write(f"Max pairs per type: {MAX_PAIRS_PER_TYPE}\n")
        f.write(f"Pair target counts: {PAIR_TARGET_COUNTS}\n")
        f.write("Pair counts:\n")
        for name, count in counts.items():
                f.write(f"  - {name}: {count}\n")
        f.write(f"Include clean+noise: {_pair_enabled('clean_target_plus_noise')}\n")
        f.write(f"Debug pair samples: {SAVE_DEBUG_PAIR_SAMPLES} ({DEBUG_PAIR_SAMPLES_PER_TYPE})\n")
        f.write(f"LR: {LEARNING_RATE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Best val loss: {best_val:.6f}\n")
        f.write(f"Loss plot: {loss_plot_path}\n")

        # Also save a full, machine-readable config file with all settings
        config_txt_path = save_dir / "config.txt"
        try:
            with open(config_txt_path, "w", encoding="utf-8") as cf:
                json.dump(config, cf, sort_keys=True, indent=2)
                cf.write("\n")
        except Exception as e:
            print(f"⚠️  Failed saving config.txt: {e}")

    print("\nDone.")
    print(f"  Best model: {best_path}")
    print(f"  Final model: {final_path}")
    print(f"  Loss plot: {loss_plot_path}")
    print(f"  Config (pkl): {save_dir / 'config.pkl'}")
    print(f"  Config (txt): {save_dir / 'config.txt'}")


if __name__ == "__main__":
    main()
