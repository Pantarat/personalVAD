#!/usr/bin/env python3
"""
D-Vector Autoencoder for Main Speaker Extraction
Train an autoencoder to extract main speaker d-vector from overlapped speech

This pipeline:
1. Loads main speaker single utterance d-vectors (clean/target)
2. Loads overlap utterance d-vectors (main + other speaker as input)
3. Creates training pairs: overlap (main + other) -> clean main speaker
4. Trains an autoencoder to predict main speaker's clean d-vector from overlap
5. Evaluates reconstruction quality
6. Saves trained model for future use

QUICK START - Training Schemes:
================================
Set TRAINING_SCHEME to one of these 6 predefined schemes:

1. 'mainOnly' 
   - Main speaker singles + overlaps → clean main
   - Silence → silence
   - Use: Simple baseline, single speaker extraction

2. 'mainOnly_otherSingles'
   - Same as mainOnly + 300 additional identity speakers
   - Use: Better generalization to unseen speakers

3. 'manyMain'
   - Multiple main speakers (can differ from dataset)
   - All main speakers → their clean versions
   - Use: Multi-speaker extraction

4. 'manyMain_otherSingles'
   - Same as manyMain + 300 additional identity speakers
   - Use: Multi-speaker with better generalization

5. 'sumNotmain'
   - Main speaker → clean main
   - All non-main (singles + overlaps) → global average
   - Use: Learn to reject non-target speakers

6. 'sumNotmain_otherSingles'
   - Same as sumNotmain + additional identity speakers in average
   - Use: Best rejection of non-target speakers

All schemes include silence identity pairs.
See TRAINING SCHEME SELECTOR section in configuration for details.
"""

import numpy as np
from autoencoder_utils import DvectorAutoencoder, load_autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import glob
import random
import librosa
from resemblyzer import VoiceEncoder
from collections import defaultdict
import pickle
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# TRAINING SCHEME SELECTOR
# ============================================================================
# Select one of the predefined training schemes below by setting TRAINING_SCHEME
# This will automatically configure all dataset composition parameters
#
# Available schemes:
# 1. 'mainOnly' - Only main speaker identity + overlaps -> clean main
#    - Main speaker clean singles -> same clean (identity)
#    - Main speaker overlaps -> clean main (denoising)
#    - Silence -> silence (identity)
#
# 2. 'mainOnly_otherSingles' - Same as mainOnly + additional identity speakers
#    - Everything from mainOnly
#    - Additional random speakers -> same (identity, for generalization)
#
# 3. 'manyMain' - Multiple main speakers (different from dataset)
#    - Multiple main speakers identity + overlaps -> their clean versions
#    - Silence -> silence (identity)
#    - NOTE: Main speakers may differ from those in overlap dataset
#
# 4. 'manyMain_otherSingles' - Same as manyMain + additional identity speakers
#    - Everything from manyMain
#    - Additional random speakers -> same (identity, for generalization)
#
# 5. 'sumNotmain' - Non-main speakers averaged
#    - Main speaker identity + overlaps -> clean main
#    - Other non-main singles/overlaps -> global average (rejection)
#    - Silence -> silence (identity)
#
# 6. 'sumNotmain_otherSingles' - Same as sumNotmain + additional identity speakers
#    - Everything from sumNotmain
#    - Additional identity speakers -> global average (included in average)
#
TRAINING_SCHEME = 'mainOnly_otherSingles'  # Choose: 'mainOnly', 'mainOnly_otherSingles', 'manyMain', 'manyMain_otherSingles', 'sumNotmain', 'sumNotmain_otherSingles'
INCLUDE_NOISE_IDENTITY = True  # Include noise samples -> silence mapping

# Scheme configurations (DO NOT EDIT - automatically set based on TRAINING_SCHEME)
SCHEME_CONFIGS = {
    'mainOnly': {
        'INCLUDE_IDENTITY_PAIRS': True,
        'INCLUDE_OTHER_SPEAKERS_IDENTITY': False,
        'INCLUDE_SILENCE_IDENTITY': False,
        'INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS': False,
        'INCLUDE_OTHER_MAIN_SPEAKERS_AVG': False,
        'INCLUDE_NON_TARGET_OVERLAPS': False,
    },
    'mainOnly_otherSingles': {
        'INCLUDE_IDENTITY_PAIRS': True,
        'INCLUDE_OTHER_SPEAKERS_IDENTITY': True,
        'INCLUDE_SILENCE_IDENTITY': False,
        'INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS': False,
        'INCLUDE_OTHER_MAIN_SPEAKERS_AVG': False,
        'INCLUDE_NON_TARGET_OVERLAPS': False,
    },
    'manyMain': {
        'INCLUDE_IDENTITY_PAIRS': True,
        'INCLUDE_OTHER_SPEAKERS_IDENTITY': True,
        'INCLUDE_SILENCE_IDENTITY': True,
        'INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS': False,
        'INCLUDE_OTHER_MAIN_SPEAKERS_AVG': False,
        'INCLUDE_NON_TARGET_OVERLAPS': False,
    },
    'manyMain_otherSingles': {
        'INCLUDE_IDENTITY_PAIRS': True,
        'INCLUDE_OTHER_SPEAKERS_IDENTITY': False,
        'INCLUDE_SILENCE_IDENTITY': True,
        'INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS': True,
        'INCLUDE_OTHER_MAIN_SPEAKERS_AVG': False,
        'INCLUDE_NON_TARGET_OVERLAPS': False,
    },
    'sumNotmain': {
        'INCLUDE_IDENTITY_PAIRS': True,
        'INCLUDE_OTHER_SPEAKERS_IDENTITY': True,
        'INCLUDE_SILENCE_IDENTITY': True,
        'INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS': False,
        'INCLUDE_OTHER_MAIN_SPEAKERS_AVG': True,
        'INCLUDE_NON_TARGET_OVERLAPS': False,
    },
    'sumNotmain_otherSingles': {
        'INCLUDE_IDENTITY_PAIRS': True,
        'INCLUDE_OTHER_SPEAKERS_IDENTITY': True,
        'INCLUDE_SILENCE_IDENTITY': True,
        'INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS': True,
        'INCLUDE_OTHER_MAIN_SPEAKERS_AVG': True,
        'INCLUDE_NON_TARGET_OVERLAPS': False,
    },
}

# Apply selected scheme configuration
if TRAINING_SCHEME not in SCHEME_CONFIGS:
    raise ValueError(f"Invalid TRAINING_SCHEME: {TRAINING_SCHEME}. Must be one of: {list(SCHEME_CONFIGS.keys())}")

_scheme_config = SCHEME_CONFIGS[TRAINING_SCHEME]
INCLUDE_IDENTITY_PAIRS = _scheme_config['INCLUDE_IDENTITY_PAIRS']
INCLUDE_OTHER_SPEAKERS_IDENTITY = _scheme_config['INCLUDE_OTHER_SPEAKERS_IDENTITY']
INCLUDE_SILENCE_IDENTITY = _scheme_config['INCLUDE_SILENCE_IDENTITY']
INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS = _scheme_config['INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS']
INCLUDE_OTHER_MAIN_SPEAKERS_AVG = _scheme_config['INCLUDE_OTHER_MAIN_SPEAKERS_AVG']
INCLUDE_NON_TARGET_OVERLAPS = _scheme_config['INCLUDE_NON_TARGET_OVERLAPS']

# Only print configuration when running as main script, not when importing
if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"TRAINING SCHEME: {TRAINING_SCHEME}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Main speaker identity pairs: {INCLUDE_IDENTITY_PAIRS}")
    print(f"  - Silence identity pairs: {INCLUDE_SILENCE_IDENTITY}")
    print(f"  - Additional identity speakers: {INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS}")
    print(f"  - Other main speakers averaging: {INCLUDE_OTHER_MAIN_SPEAKERS_AVG}")
    print(f"  - Other speakers identity: {INCLUDE_OTHER_SPEAKERS_IDENTITY}")
    print(f"  - Non-target overlaps: {INCLUDE_NON_TARGET_OVERLAPS}")
    print(f"{'='*70}\n")

# ============================================================================
# DATA PATHS AND SPEAKER CONFIGURATION
# ============================================================================

# Paths
LIBRISPEECH_PATH = '../../data/LibriSpeech'
# Overlap samples directories - can be a single directory (string) or list of directories
# If list, model will load overlap samples from all directories
# OVERLAP_SAMPLES_DIRS = ['test_outputs/84+30spk_300','test_outputs/116+30spk_300','test_outputs/700+30spk_300','test_outputs/1255+30spk_300','test_outputs/1585+30spk_300','test_outputs/1630+30spk_300']  # e.g., ['dir1', 'dir2', 'dir3'] for multiple
OVERLAP_SAMPLES_DIRS = ['test_outputs/data/61/61_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp_2000']  # e.g., ['dir1', 'dir2', 'dir3'] for multiple
NON_TARGET_OVERLAP_SAMPLES_DIR = 'test_outputs/non_target_overlap_samples'
MODEL_SAVE_DIR = 'test_outputs/models/greedy_finetune/dvector_ae-61_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_greedyPretrain-2000_1e-5_100ep'  # Directory to save trained model and config
# MODEL_SAVE_DIR = 'test_outputs/test'

# Single speaker dataset directories (HIGHEST PRIORITY for main speaker clean utterances)
# Set to a list of single speaker dataset paths to load clean main speaker utterances
# If specified, will use these instead of extracting from LibriSpeech
# These should be the same datasets used to generate the overlap samples
SINGLE_SPEAKER_DATASETS = [
    '../../data/speaker_61/train1',
    '../../data/speaker_61/train2',
    '../../data/speaker_61/train3',
    '../../data/speaker_61/train4',
    '../../data/speaker_61/train5',
    '../../data/speaker_61/train6',
    '../../data/speaker_61/train7',
    '../../data/speaker_61/train8',
    '../../data/speaker_61/train9',
    '../../data/speaker_61/train10',
    '../../data/speaker_61/train11',
    '../../data/speaker_61/train12',
    '../../data/speaker_61/train13',
    '../../data/speaker_61/train14',
    '../../data/speaker_61/train15',
    '../../data/speaker_61/train16',
    '../../data/speaker_61/train17',
    '../../data/speaker_61/train18',
    '../../data/speaker_61/train19',
    '../../data/speaker_61/train20',
    '../../data/speaker_61/train21',
    '../../data/speaker_61/train22',
    '../../data/speaker_61/train23',
    '../../data/speaker_61/train24',
    '../../data/speaker_61/train25',
    # '../../data/speaker_174/train',
]  # List of single speaker dataset paths, or None to extract from LibriSpeech

# Main speakers (should match overlap generation) - THESE ARE THE TARGETS
# Can be a single speaker ID (string) or list of speaker IDs
# If list, model will learn to extract any of these speakers from overlap
# Set to None to auto-detect from speaker_config.json in overlap directories or single speaker datasets
# MAIN_SPEAKERS = ['84', '116', '700', '1255', '1585', '1630']  # e.g., ['84'] or ['84', '174', '251'] for multiple main speakers
MAIN_SPEAKERS = None # For main 174 only
# MAIN_SPEAKERS = None  # Set to None to auto-detect from overlap directories
N_MAIN_SPEAKERS = None  # Optional: If MAIN_SPEAKERS is None and speaker_config.json doesn't exist, randomly select this many speakers

# Other speakers configuration
# Option 1: Explicitly list speakers
OTHER_SPEAKERS = None  # Set to list like ['174', '251', ...] to specify explicitly
# OTHER_SPEAKERS = ['174', '251', '422', '652', '777', '1272', '1462', '1673', '1919', '1988', '1993', '2035', '2078', '2086', '2277', '2412', '2428', '2803', '2902', '3000', '3081', '3170', '3536', '3752', '3853', '5338', '5536', '5694', '5895', '6241']

# Option 2: Load from overlap sample directories (if OTHER_SPEAKERS is None)
# Will automatically extract speakers from overlap samples
AUTO_DETECT_OTHER_SPEAKERS = True  # If True and OTHER_SPEAKERS is None, detect from overlap samples

# Noise data configuration (MUSAN and RIRS)
# Include noise samples mapped to silence (noise -> silence identity)
# INCLUDE_NOISE_IDENTITY = True  # Include noise samples -> silence mapping
MUSAN_ROOT = '../../kaldi/egs/pvad/musan'  # Path to MUSAN dataset root (with train/test splits)
RIRS_ROOT = '../../kaldi/egs/pvad/RIRS_NOISES'  # Path to RIRS_NOISES dataset root
N_NOISE_SAMPLES_PER_TYPE = 100  # Number of noise samples to use per type (music, noise, speech from MUSAN)
N_RIR_SAMPLES = 500  # Number of RIR samples to use (set to 0 to disable, RIRs are not really "noise")
NOISE_SAMPLE_DURATION = 1.6  # Duration in seconds for each noise sample

# ============================================================================
# DATASET COMPOSITION PARAMETERS
# ============================================================================
# NOTE: Most of these are automatically set by TRAINING_SCHEME above
# Only modify if you need custom configuration beyond the predefined schemes

# Number of single utterances for MAIN SPEAKER (target for reconstruction)
N_SINGLE_UTTERANCES_PER_SPEAKER = 50

# Number of pairs to use (per speaker/category)
N_IDENTITY_PAIRS = -1  # Number of identity pairs to use PER MAIN SPEAKER (-1 = use all available clean utterances)
N_OTHER_SPEAKERS_IDENTITY_PAIRS = -1  # Number of identity pairs to use PER OTHER SPEAKER (from overlap) (-1 = use all available)
N_SILENCE_IDENTITY_PAIRS = 200  # Number of silence identity pairs to use (-1 = use all available)
N_OVERLAP_PAIRS = -1  # Number of overlap pairs to use PER DIRECTORY (-1 = use all available from each directory)

# Additional identity speakers configuration (used when INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS=True)
N_ADDITIONAL_IDENTITY_SPEAKERS = 100  # Number of additional speakers to use for identity pairs
N_ADDITIONAL_IDENTITY_UTTERANCES_PER_SPEAKER = 10  # Number of utterances per additional speaker
N_ADDITIONAL_IDENTITY_PAIRS_PER_SPEAKER = 500  # Number of identity pairs per additional speaker (-1 = use all available)
ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS = [
    # 'train-clean-100',
    # 'train-clean-360',
    'train-other-500'
]  # LibriSpeech subsets to sample additional identity speakers from (None = default subsets)
ADDITIONAL_IDENTITY_PARTITION_DURATION = 1.6  # Seconds; >0 splits each selected utterance into fixed segments for d-vector extraction

# Other main speakers configuration (used when INCLUDE_OTHER_MAIN_SPEAKERS_AVG=True)
N_OTHER_MAIN_SPEAKERS_PAIRS_PER_SPEAKER = -1  # Number of pairs per other main speaker (singles + overlaps -> average, -1 = all)

# Non-target overlap configuration (used when INCLUDE_NON_TARGET_OVERLAPS=True)
N_NON_TARGET_OVERLAP_SAMPLES = 300  # Number of non-target overlap samples to use
NON_TARGET_STRATEGY = 'zero'  # 'zero': predict zero vector, 'mean': predict mean of other speakers, 'random': random noise

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.1  # Fraction of data for validation (e.g., 0.2 = 20%)
TEST_SPLIT = 0.01  # Fraction of data for test (e.g., 0.15 = 15%)
EARLY_STOPPING_PATIENCE = 10

# Fine-tuning from pre-trained model
# Set to path of pre-trained model directory (e.g., from train_dvector_autoencoder_identity.py)
# to initialize weights before training. Set to None to train from scratch.
# PRETRAINED_MODEL_PATH = 'test_outputs/dvector_ae_identity_1100x50Dev_noOV_1,6s_20-2-27' # e.g., 'test_outputs/dvector_ae_identity'
PRETRAINED_MODEL_PATH = 'test_outputs/models/greedy/dvector_ae_greedy_layerwise_14_6-5-26'  # Uncomment to enable fine-tuning

# Model architecture
HIDDEN_DIMS = [192, 192]  # Encoder-bottleneck-decoder (increased bottleneck capacity)
DROPOUT_RATE = 0.05
AE_NORM_TYPE = 'layernorm'
AE_USE_RESIDUAL = False
AE_RESIDUAL_SCALE_INIT = 0.5

# Mixed reconstruction loss: total = MSE_WEIGHT * MSE + COSINE_WEIGHT * (1 - cosine)
LOSS_MSE_WEIGHT = 0.7
LOSS_COSINE_WEIGHT = 0.3
LOSS_COSINE_EPS = 1e-8

# Negative contrastive term (push recon away from mismatched targets)
NEGATIVE_CONTRASTIVE_WEIGHT = 0.2
NEGATIVE_CONTRASTIVE_MARGIN = 0.2

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Audio settings
SAMPLE_RATE = 16000

# Random seed
RANDOM_SEED = 42

# D-vector extraction cache
DVECTOR_CACHE_ENABLED = True
DVECTOR_CACHE_DIR = 'test_outputs/dvector_cache'
DVECTOR_CACHE_VERSION = 1

# ============================================================================
# NOTE: This script reuses DvectorAutoencoder from autoencoder_utils so training
# and inference architectures remain aligned.
# ============================================================================


# class DvectorAutoencoder(nn.Module):
#     """
#     Autoencoder to map overlap d-vectors to clean d-vectors
    
#     Architecture:
#     - Encoder: d-vector (256) -> hidden layers -> bottleneck
#     - Decoder: bottleneck -> hidden layers -> d-vector (256)
#     """
    
#     def __init__(self, input_dim=256, hidden_dims=[128, 64, 128], dropout_rate=0.2):
#         super(DvectorAutoencoder, self).__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dims = hidden_dims
        
#         # Encoder
#         encoder_layers = []
#         prev_dim = input_dim
        
#         # First half of hidden_dims for encoder
#         n_encoder_layers = len(hidden_dims) // 2
#         for i in range(n_encoder_layers):
#             encoder_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
#             encoder_layers.append(nn.BatchNorm1d(hidden_dims[i]))
#             encoder_layers.append(nn.ReLU())
#             encoder_layers.append(nn.Dropout(dropout_rate))
#             prev_dim = hidden_dims[i]
        
#         # Bottleneck
#         bottleneck_dim = hidden_dims[n_encoder_layers]
#         encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
#         encoder_layers.append(nn.BatchNorm1d(bottleneck_dim))
#         encoder_layers.append(nn.ReLU())
        
#         self.encoder = nn.Sequential(*encoder_layers)
        
#         # Decoder
#         decoder_layers = []
#         prev_dim = bottleneck_dim
        
#         # Second half of hidden_dims for decoder
#         for i in range(n_encoder_layers + 1, len(hidden_dims)):
#             decoder_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
#             decoder_layers.append(nn.BatchNorm1d(hidden_dims[i]))
#             decoder_layers.append(nn.ReLU())
#             decoder_layers.append(nn.Dropout(dropout_rate))
#             prev_dim = hidden_dims[i]
        
#         # Output layer with ReLU to ensure non-negative d-vector values
#         decoder_layers.append(nn.Linear(prev_dim, input_dim))
#         decoder_layers.append(nn.ReLU())  # Ensure all output dimensions >= 0
        
#         self.decoder = nn.Sequential(*decoder_layers)
    
#     def forward(self, x):
#         """Forward pass"""
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
    
#     def encode(self, x):
#         """Encode to bottleneck representation"""
#         return self.encoder(x)
    
#     def decode(self, z):
#         """Decode from bottleneck representation"""
#         return self.decoder(z)


class DvectorPairDataset(Dataset):
    """
    Dataset of (input_dvector, target_dvector) pairs
    Supports four types of pairs:
    1. Identity pairs: clean main speaker -> same clean main speaker (identity mapping)
    2. Overlap pairs: overlap (main+other) -> clean main speaker (denoising)
    3. Non-target pairs: overlap (other1+other2) -> zero/noise (rejection)
    4. Other main speakers pairs: (single or overlap) -> average d-vector (averaging)
    """

    # Sample kind IDs used by downstream asymmetric training.
    SAMPLE_KIND_TARGET_OVERLAP = 0
    SAMPLE_KIND_TARGET_IDENTITY = 1
    SAMPLE_KIND_NON_TARGET_OVERLAP = 2
    SAMPLE_KIND_OTHER = 3
    
    def __init__(self, overlap_dvectors, clean_dvectors, speaker_pairs,
                 identity_pairs=None, non_target_dvectors=None, non_target_pairs=None, non_target_target=None,
                 other_main_speakers_pairs=None, other_main_speakers_averages=None,
                 main_speaker_ids=None):
        """
        Args:
            overlap_dvectors: Dict of full overlap d-vectors (main + other)
            clean_dvectors: Dict of clean main speaker d-vectors
            speaker_pairs: List of (overlap_key, clean_key) tuples for full overlaps
            identity_pairs: List of (clean_key, clean_key) tuples for identity mapping
            non_target_dvectors: Dict of non-target overlap d-vectors (other1 + other2)
            non_target_pairs: List of (non_target_key, None) tuples
            non_target_target: Single target vector for all non-target samples (zero/mean/random)
            other_main_speakers_pairs: List of (input_key, speaker_id, type) tuples for other main speakers
            other_main_speakers_averages: Dict of {speaker_id: average_dvector} for other main speakers
            main_speaker_ids: List of speaker IDs treated as target speakers
        """
        self.overlap_dvectors = overlap_dvectors
        self.clean_dvectors = clean_dvectors
        self.speaker_pairs = speaker_pairs
        self.identity_pairs = identity_pairs or []
        self.non_target_dvectors = non_target_dvectors or {}
        self.non_target_pairs = non_target_pairs or []
        self.non_target_target = non_target_target
        self.other_main_speakers_pairs = other_main_speakers_pairs or []
        self.other_main_speakers_averages = other_main_speakers_averages or {}
        self.main_speaker_ids = set(str(s) for s in (main_speaker_ids or []))
        
        # Combined pairs: identity + overlap + non-target + other main speakers
        self.all_pairs = self.identity_pairs + speaker_pairs + self.non_target_pairs + self.other_main_speakers_pairs
    
    def __len__(self):
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        pair = self.all_pairs[idx]
        # print(f"Fetching pair {idx}/{len(self.all_pairs)}: {pair}")
        target_speaker_id = None
        sample_kind = self.SAMPLE_KIND_OTHER
        
        # Check if it's a tuple with 4 elements (identity pair with metadata)
        if len(pair) == 4:
            input_key, target_key, speaker_id, pair_type = pair
            target_speaker_id = speaker_id
            
            if pair_type == 'identity':
                # Identity pair: clean -> clean (same)
                input_dvec = self.clean_dvectors[input_key]['dvector']
                target_dvec = self.clean_dvectors[target_key]['dvector']
                if str(speaker_id) in self.main_speaker_ids:
                    sample_kind = self.SAMPLE_KIND_TARGET_IDENTITY
                else:
                    sample_kind = self.SAMPLE_KIND_OTHER
            else:
                # This shouldn't happen, but handle gracefully
                input_dvec = self.clean_dvectors[input_key]['dvector']
                target_dvec = self.clean_dvectors[target_key]['dvector']
                sample_kind = self.SAMPLE_KIND_OTHER
        
        # Check if it's a tuple with 3 elements (other main speakers pair or global average pair)
        elif len(pair) == 3:
            # Other main speakers pair: (input_key, speaker_id, type) -> average
            input_key, speaker_id, pair_type = pair
            target_speaker_id = None  # target is average/global representation (non-target objective)
            
            if pair_type in ['single', 'other_identity', 'additional_identity']:
                # Single utterance (other main speaker, other identity, or additional identity)
                input_dvec = self.clean_dvectors[input_key]['dvector']
            else:  # pair_type == 'overlap'
                # Overlap with other main speaker as main
                input_dvec = self.overlap_dvectors[input_key]['dvector']
            
            # Check if using global average (speaker_id == 'global_avg')
            if speaker_id == 'global_avg':
                # Target is the single global average for all non-main-speaker utterances
                target_dvec = self.other_main_speakers_averages['global_avg']
            else:
                # Target is the per-speaker average d-vector
                target_dvec = self.other_main_speakers_averages[speaker_id]

            sample_kind = self.SAMPLE_KIND_OTHER
        
        elif pair[1] is None:
            # Non-target overlap pair
            non_target_key = pair[0]
            input_dvec = self.non_target_dvectors[non_target_key]['dvector']
            target_dvec = self.non_target_target  # Use single shared target
            target_speaker_id = None
            sample_kind = self.SAMPLE_KIND_NON_TARGET_OVERLAP
        else:
            # Full overlap pair (main + other -> main)
            overlap_key, clean_key = pair
            input_dvec = self.overlap_dvectors[overlap_key]['dvector']
            target_dvec = self.clean_dvectors[clean_key]['dvector']
            target_speaker_id = self.clean_dvectors[clean_key].get('speaker', None)
            sample_kind = self.SAMPLE_KIND_TARGET_OVERLAP

        is_target = 1.0 if (target_speaker_id is not None and str(target_speaker_id) in self.main_speaker_ids) else 0.0
        
        return (
            torch.FloatTensor(input_dvec),
            torch.FloatTensor(target_dvec),
            torch.tensor(is_target, dtype=torch.float32),
            torch.tensor(sample_kind, dtype=torch.long)
        )


def _normalize_for_hash(value):
    """Normalize nested config values into deterministic JSON-serializable structures."""
    if isinstance(value, Path):
        return str(value.resolve())

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


def _resolve_path_str(path_value, script_dir):
    """Resolve a path to absolute canonical string for cache keys."""
    path = Path(path_value)
    if not path.is_absolute():
        path = script_dir / path
    return str(path.resolve())


def _cache_file_for_config(cache_dir, cache_name, cache_config):
    """Build deterministic cache file path from cache name + config."""
    payload = {
        'cache_name': cache_name,
        'cache_version': DVECTOR_CACHE_VERSION,
        'config': _normalize_for_hash(cache_config),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    digest = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()[:16]
    return cache_dir / f"{cache_name}_{digest}.pkl"


def _extract_with_cache(cache_dir, cache_name, cache_config, extractor_fn):
    """Load extracted d-vectors from cache when config matches; otherwise extract and save."""
    normalized_config = _normalize_for_hash(cache_config)
    cache_file = _cache_file_for_config(cache_dir, cache_name, normalized_config)

    if DVECTOR_CACHE_ENABLED and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_payload = pickle.load(f)

            if isinstance(cached_payload, dict) and 'config' in cached_payload and 'data' in cached_payload:
                cached_config = cached_payload.get('config')
                cached_version = cached_payload.get('cache_version')
                if cached_version == DVECTOR_CACHE_VERSION and cached_config == normalized_config:
                    print(f"\n♻️  Cache hit for {cache_name}: {cache_file}")
                    return cached_payload['data']
                print(f"\n⚠️  Cache config/version mismatch for {cache_name}, regenerating")
            else:
                # Legacy/simple cache payload: treat loaded object as the data.
                print(f"\n♻️  Cache hit for {cache_name}: {cache_file}")
                return cached_payload
        except Exception as e:
            print(f"\n⚠️  Failed reading cache for {cache_name}: {e}")
            print("   Re-extracting d-vectors...")

    print(f"\n🔄 Cache miss for {cache_name}; extracting d-vectors...")
    data = extractor_fn()

    if DVECTOR_CACHE_ENABLED:
        cache_payload = {
            'cache_version': DVECTOR_CACHE_VERSION,
            'config': normalized_config,
            'data': data,
        }
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_payload, f)
            print(f"✓ Cached {cache_name} to: {cache_file}")
        except Exception as e:
            print(f"⚠️  Could not write cache for {cache_name}: {e}")

    return data


def extract_silence_dvectors(n_samples=500, duration_per_sample=1.0, sample_rate=16000, device='cuda'):
    """Extract d-vectors from silence (zero audio) frames"""
    print(f"\n🔇 Extracting SILENCE d-vectors...")
    print(f"  Number of samples: {n_samples}")
    print(f"  Duration per sample: {duration_per_sample}s")
    
    # Initialize encoder
    if torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cuda')
        print(f"  Using device: CUDA")
    else:
        device_obj = torch.device('cpu')
        print(f"  Using device: CPU")
    
    encoder = VoiceEncoder(device=device_obj)
    print(f"✓ Voice encoder loaded")
    
    results = {}
    
    for i in range(n_samples):
        try:
            # Create silence audio
            n_samples_audio = int(duration_per_sample * sample_rate)
            audio = np.zeros(n_samples_audio, dtype=np.float32)
            
            # Extract mel spectrogram
            fbanks = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
            ).astype('float32').T
            
            # Get d-vector
            with torch.no_grad():
                fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
                dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()
            
            # Normalize
            dvector = dvector / np.linalg.norm(dvector)
            
            silence_key = f"silence_{i:04d}"
            results[silence_key] = {
                'dvector': dvector,
                'speaker': 'silence',
                'type': 'silence'
            }
            
        except Exception as e:
            print(f"    ⚠️  Error extracting silence {i}: {e}")
            continue
    
    print(f"✓ Total silence d-vectors: {len(results)}")
    
    return results


def extract_noise_dvectors(musan_root, rirs_root, n_noise_samples_per_type=100, n_rir_samples=50,
                           noise_sample_duration=1.0, sample_rate=16000, device='cuda'):
    """
    Extract d-vectors from noise samples (MUSAN and RIRS TRAIN PARTITIONS ONLY)
    These will be mapped to silence during training
    
    IMPORTANT: Only uses pre-partitioned TRAIN sets to avoid data leakage:
      - MUSAN: musan_music_train, musan_noise_train, musan_speech_train
      - RIRS: simulated_rirs/{room_type}/rir_list_train
    
    Args:
        musan_root: Path to MUSAN dataset root (with musan_*_train folders)
        rirs_root: Path to RIRS_NOISES dataset root
        n_noise_samples_per_type: Number of samples per MUSAN type (music, noise, speech)
        n_rir_samples: Number of RIR samples to use
        noise_sample_duration: Duration to extract from each noise file
        sample_rate: Audio sample rate
        device: Device for d-vector extraction
    
    Returns:
        Dict of noise d-vectors
    """
    print(f"\n🔊 Extracting NOISE d-vectors (TRAIN PARTITIONS ONLY)...")
    print(f"  MUSAN root: {musan_root}")
    print(f"  RIRS root: {rirs_root}")
    print(f"  Samples per type: {n_noise_samples_per_type}")
    print(f"  RIR samples: {n_rir_samples}")
    print(f"  Duration per sample: {noise_sample_duration}s")
    
    # Initialize encoder
    if torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cuda')
        print(f"  Using device: CUDA")
    else:
        device_obj = torch.device('cpu')
        print(f"  Using device: CPU")
    
    encoder = VoiceEncoder(device=device_obj)
    print(f"✓ Voice encoder loaded")
    
    results = {}
    script_dir = Path(__file__).parent
    
    # Resolve paths
    musan_path = Path(musan_root)
    if not musan_path.is_absolute():
        musan_path = script_dir / musan_path
    
    rirs_path = Path(rirs_root)
    if not rirs_path.is_absolute():
        rirs_path = script_dir / rirs_path
    
    # MUSAN noise types to load (TRAIN PARTITIONS ONLY - ensures no test data leakage)
    musan_types = {
        'music': musan_path / 'musan_music_train',
        'noise': musan_path / 'musan_noise_train',
        'speech': musan_path / 'musan_speech_train'
    }
    
    # Validation: Ensure we're using train partitions only
    # NOTE: check directory name only (not full absolute path), otherwise paths
    # containing folders like "AE_test" can trigger false positives.
    for noise_type, noise_dir in musan_types.items():
        dir_name = noise_dir.name.lower()
        if dir_name.endswith('_test'):
            raise ValueError(f"ERROR: Attempting to use TEST partition for {noise_type}: {noise_dir}")
        if not dir_name.endswith('_train'):
            raise ValueError(f"ERROR: Directory name must end with '_train': {noise_dir}")
    
    # Extract from MUSAN
    for noise_type, noise_dir in musan_types.items():
        if not noise_dir.exists():
            print(f"  ⚠️  {noise_type} directory not found: {noise_dir}")
            continue
        
        print(f"\n  Processing MUSAN {noise_type}...")
        
        # Read wav.scp to get audio files
        wav_scp = noise_dir / 'wav.scp'
        if not wav_scp.exists():
            print(f"    ⚠️  wav.scp not found in {noise_dir}")
            continue
        
        # Parse wav.scp
        audio_files = []
        with open(wav_scp, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id = parts[0]
                    # Handle both pipe commands and direct paths
                    if '|' in line:
                        # Skip pipe commands, too complex to parse
                        continue
                    else:
                        audio_path = parts[1]
                        # Paths in wav.scp are like "musan/speech/us-gov/speech-us-gov-0120.wav"
                        # They are relative to the parent of the musan directory
                        audio_path_obj = Path(audio_path)
                        if not audio_path_obj.is_absolute():
                            # Resolve relative to parent of musan_path
                            audio_path_obj = musan_path.parent / audio_path
                        audio_files.append((utt_id, str(audio_path_obj)))
        
        if not audio_files:
            print(f"    ⚠️  No valid audio files found in wav.scp")
            continue
        
        print(f"    Found {len(audio_files)} audio files")
        
        # Sample randomly
        n_to_sample = min(n_noise_samples_per_type, len(audio_files))
        np.random.seed(RANDOM_SEED)
        sampled_files = np.random.choice(len(audio_files), size=n_to_sample, replace=False)
        
        # Extract d-vectors
        for idx in sampled_files:
            utt_id, audio_path = audio_files[idx]
            
            try:
                # Verify file exists before loading
                if not Path(audio_path).exists():
                    print(f"    ⚠️  File not found: {audio_path}")
                    continue
                
                # Load audio
                audio, sr = librosa.load(audio_path, sr=sample_rate, duration=noise_sample_duration)
                
                if len(audio) == 0:
                    continue
                
                # Ensure minimum duration
                min_samples = int(noise_sample_duration * sample_rate)
                if len(audio) < min_samples:
                    audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
                
                # Extract mel spectrogram
                fbanks = librosa.feature.melspectrogram(
                    y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
                ).astype('float32').T
                
                # Get d-vector
                with torch.no_grad():
                    fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
                    dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()
                
                # Normalize
                dvector = dvector / np.linalg.norm(dvector)
                
                noise_key = f"noise_{noise_type}_{utt_id}"
                results[noise_key] = {
                    'dvector': dvector,
                    'type': f'noise_{noise_type}',
                    'source': str(audio_path)
                }
                
            except Exception as e:
                print(f"    ⚠️  Error processing {utt_id}: {e}")
                continue
        
        print(f"    ✓ Extracted {sum(1 for k in results if k.startswith(f'noise_{noise_type}'))} d-vectors")
    
    # Extract from RIRS (if requested) - TRAIN PARTITIONS ONLY
    if n_rir_samples > 0:
        print(f"\n  Processing RIRS (reverb as noise - TRAIN PARTITION ONLY)...")
        
        # Try to find RIR lists in simulated_rirs
        rir_lists = []
        simulated_rirs = rirs_path / 'simulated_rirs'
        
        if simulated_rirs.exists():
            # Look for rir_list_train files in room types (TRAIN ONLY)
            for room_type in ['smallroom', 'mediumroom', 'largeroom']:
                rir_list_file = simulated_rirs / room_type / 'rir_list_train'
                # Validation: Ensure we're using train partition.
                # Check file name only to avoid false positives from parent dirs like "AE_test".
                rir_list_name = rir_list_file.name.lower()
                if rir_list_name.endswith('_test'):
                    raise ValueError(f"ERROR: Attempting to use TEST partition: {rir_list_file}")
                if not rir_list_name.endswith('_train'):
                    raise ValueError(f"ERROR: RIR list must be from train partition: {rir_list_file}")
                rir_list_file = simulated_rirs / room_type / 'rir_list_train'
                if rir_list_file.exists():
                    with open(rir_list_file, 'r') as f:
                        for line in f:
                            rir_path = line.strip()
                            if rir_path:
                                # Make path absolute relative to RIRS root
                                full_path = rirs_path / rir_path
                                if full_path.exists():
                                    rir_lists.append((room_type, str(full_path)))
        
        if rir_lists:
            print(f"    Found {len(rir_lists)} RIR files")
            
            # Sample randomly
            n_to_sample = min(n_rir_samples, len(rir_lists))
            np.random.seed(RANDOM_SEED + 1)  # Different seed for RIRs
            sampled_indices = np.random.choice(len(rir_lists), size=n_to_sample, replace=False)
            
            for idx in sampled_indices:
                room_type, rir_path = rir_lists[idx]
                
                try:
                    # Load RIR
                    rir, sr = librosa.load(rir_path, sr=sample_rate, duration=noise_sample_duration)
                    
                    if len(rir) == 0:
                        continue
                    
                    # Ensure minimum duration
                    min_samples = int(noise_sample_duration * sample_rate)
                    if len(rir) < min_samples:
                        rir = np.pad(rir, (0, min_samples - len(rir)), mode='constant')
                    
                    # Extract mel spectrogram
                    fbanks = librosa.feature.melspectrogram(
                        y=rir, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
                    ).astype('float32').T
                    
                    # Get d-vector
                    with torch.no_grad():
                        fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
                        dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()
                    
                    # Normalize
                    dvector = dvector / np.linalg.norm(dvector)
                    
                    rir_key = f"noise_rir_{room_type}_{idx}"
                    results[rir_key] = {
                        'dvector': dvector,
                        'type': f'noise_rir_{room_type}',
                        'source': rir_path
                    }
                    
                except Exception as e:
                    print(f"    ⚠️  Error processing RIR: {e}")
                    continue
            
            print(f"    ✓ Extracted {sum(1 for k in results if k.startswith('noise_rir'))} RIR d-vectors")
        else:
            print(f"    ⚠️  No RIR files found in {simulated_rirs}")
    
    print(f"\n✓ Total noise d-vectors: {len(results)}")
    print(f"  Breakdown:")
    for noise_type in ['music', 'noise', 'speech']:
        count = sum(1 for k in results if k.startswith(f'noise_{noise_type}'))
        if count > 0:
            print(f"    - MUSAN {noise_type}: {count}")
    rir_count = sum(1 for k in results if k.startswith('noise_rir'))
    if rir_count > 0:
        print(f"    - RIRS reverb: {rir_count}")
    
    return results


def extract_single_speaker_dataset_dvectors(dataset_dirs, n_utterances_per_speaker=-1,
                                            sample_rate=16000, device='cuda'):
    """
    Extract d-vectors from single speaker dataset directories
    
    Args:
        dataset_dirs: List of single speaker dataset directory paths
        n_utterances_per_speaker: Number of utterances to use per speaker (-1 = use all)
        sample_rate: Audio sample rate
        device: Device for d-vector extraction
    
    Returns:
        Dict of {speaker_id: {utterance_key: data}} for all speakers
    """
    print(f"\n🎤 Extracting CLEAN d-vectors from single speaker datasets...")
    print(f"  Number of datasets: {len(dataset_dirs)}")
    if n_utterances_per_speaker > 0:
        print(f"  Max utterances per speaker: {n_utterances_per_speaker}")
    else:
        print(f"  Using all available utterances")
    
    script_dir = Path(__file__).parent

    # Extract d-vectors fresh each run
    print(f"\n🔄 Extracting d-vectors...")
    
    # Initialize encoder
    if torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cuda')
        print(f"  Using device: CUDA")
    else:
        device_obj = torch.device('cpu')
        print(f"  Using device: CPU")
    
    encoder = VoiceEncoder(device=device_obj)
    print(f"✓ Voice encoder loaded")
    
    results_by_speaker = defaultdict(dict)
    
    for dataset_path_str in dataset_dirs:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.is_absolute():
            dataset_path = script_dir / dataset_path
        
        if not dataset_path.exists():
            print(f"  ⚠️  Dataset not found: {dataset_path}")
            continue
        
        print(f"\n  Loading dataset: {dataset_path}")
        
        # Extract speaker ID from utt2spk file
        utt2spk_path = dataset_path / 'utt2spk'
        speaker_id = None
        if utt2spk_path.exists():
            with open(utt2spk_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    speaker_id = first_line.split()[1]
        
        if not speaker_id:
            print(f"    ⚠️  Could not extract speaker ID from {dataset_path}")
            continue
        
        print(f"    Speaker ID: {speaker_id}")
        
        # Load audio files from the audio directory
        audio_dir = dataset_path / 'audio'
        if not audio_dir.exists():
            print(f"    ⚠️  Audio directory not found: {audio_dir}")
            continue
        
        audio_files = sorted(list(audio_dir.glob('*.flac')))
        
        if not audio_files:
            print(f"    ⚠️  No audio files found in {audio_dir}")
            continue
        
        print(f"    Found {len(audio_files)} audio files")
        
        # Sample if requested
        if n_utterances_per_speaker > 0 and len(audio_files) > n_utterances_per_speaker:
            np.random.seed(RANDOM_SEED)
            audio_files = list(np.random.choice(audio_files, size=n_utterances_per_speaker, replace=False))
            print(f"    Sampling {n_utterances_per_speaker} utterances")
        
        # Extract d-vectors
        for audio_file in audio_files:
            try:
                audio, sr = librosa.load(str(audio_file), sr=sample_rate)
                
                utterance_key = audio_file.stem
                
                # Extract mel spectrogram
                fbanks = librosa.feature.melspectrogram(
                    y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
                ).astype('float32').T
                
                # Get d-vector
                with torch.no_grad():
                    fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
                    dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()
                
                # Normalize
                dvector = dvector / np.linalg.norm(dvector)
                
                results_by_speaker[speaker_id][utterance_key] = {
                    'dvector': dvector,
                    'speaker': speaker_id,
                    'dataset': str(dataset_path)
                }
                
            except Exception as e:
                print(f"    ⚠️  Error processing {audio_file.name}: {e}")
                continue
        
        print(f"    ✓ Extracted {len(results_by_speaker[speaker_id])} d-vectors for speaker {speaker_id}")
    
    total = sum(len(v) for v in results_by_speaker.values())
    print(f"\n✓ Total clean d-vectors from single speaker datasets: {total}")
    print(f"✓ Speakers covered: {', '.join(sorted(results_by_speaker.keys()))}")
    
    return results_by_speaker


def extract_single_utterance_dvectors(librispeech_path, speaker_ids, n_utterances_per_speaker,
                                     sample_rate=16000, device='cuda',
                                     librispeech_subsets=None, partition_duration=None):
    """
    Extract d-vectors from single speaker utterances
    """
    print(f"\n🎤 Extracting CLEAN d-vectors from single utterances...")
    print(f"  Speakers: {', '.join(speaker_ids)}")
    print(f"  Utterances per speaker: {n_utterances_per_speaker}")

    if librispeech_subsets is None:
        librispeech_subsets = ['dev-clean', 'dev-other', 'test-clean', 'test-other',
                               'train-clean-100', 'train-clean-360', 'train-other-500']
    else:
        librispeech_subsets = list(librispeech_subsets)

    partition_enabled = partition_duration is not None and partition_duration > 0
    if partition_enabled:
        print(f"  Partition duration: {partition_duration}s (enabled)")
    else:
        print(f"  Partition duration: disabled (full utterance)")
    print(f"  LibriSpeech subsets: {', '.join(librispeech_subsets)}")
    
    script_dir = Path(__file__).parent
    
    results_by_speaker = defaultdict(dict)

    speakers_to_extract = list(speaker_ids)
    print(f"\n🔄 Extracting d-vectors for {len(speakers_to_extract)} speakers...")
    
    # Initialize encoder
    if torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cuda')
        print(f"  Using device: CUDA")
    else:
        device_obj = torch.device('cpu')
        print(f"  Using device: CPU")
    
    encoder = VoiceEncoder(device=device_obj)
    print(f"✓ Voice encoder loaded")
    
    librispeech_path = Path(librispeech_path)
    
    for speaker_id in speakers_to_extract:
        print(f"\n  Processing Speaker {speaker_id}...")
        
        # Find audio files across selected LibriSpeech subsets
        audio_files = []

        for subset in librispeech_subsets:
            speaker_pattern = str(librispeech_path / subset / speaker_id / "*" / "*.flac")
            files = glob.glob(speaker_pattern)
            audio_files.extend(files)
        
        if not audio_files:
            print(f"    ⚠️  No audio files found in any subset")
            continue
        
        print(f"    Found {len(audio_files)} audio files across selected subsets")
        
        # Sample files
        n_samples = min(n_utterances_per_speaker, len(audio_files))
        np.random.seed(RANDOM_SEED)
        sampled_files = np.random.choice(audio_files, size=n_samples, replace=False)
        
        # Extract d-vectors
        n_added_for_speaker = 0
        for audio_file in sampled_files:
            try:
                audio, sr = librosa.load(audio_file, sr=sample_rate)

                stem = Path(audio_file).stem

                # Build one or more segments from this utterance
                segments = []
                if partition_enabled:
                    segment_samples = int(partition_duration * sample_rate)
                    if segment_samples <= 0:
                        raise ValueError(f"Invalid partition_duration: {partition_duration}")

                    for seg_idx, start_idx in enumerate(range(0, len(audio), segment_samples)):
                        segment_audio = audio[start_idx:start_idx + segment_samples]
                        if len(segment_audio) == 0:
                            continue
                        if len(segment_audio) < segment_samples:
                            # Pad final segment to fixed duration for consistent comparison
                            segment_audio = np.pad(segment_audio, (0, segment_samples - len(segment_audio)), mode='constant')

                        segment_key = f"{stem}_seg{seg_idx:03d}"
                        segments.append((segment_key, segment_audio))
                else:
                    segments.append((stem, audio))

                for utterance_key, segment_audio in segments:
                    # Extract mel spectrogram
                    fbanks = librosa.feature.melspectrogram(
                        y=segment_audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
                    ).astype('float32').T

                    # Get d-vector
                    with torch.no_grad():
                        fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
                        dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()

                    # Normalize
                    dvector = dvector / np.linalg.norm(dvector)

                    results_by_speaker[speaker_id][utterance_key] = {
                        'dvector': dvector,
                        'speaker': speaker_id
                    }
                    n_added_for_speaker += 1
                
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                continue

        print(f"    ✓ Extracted {len(results_by_speaker[speaker_id])} d-vectors ({n_added_for_speaker} newly added)")
    
    total = sum(len(v) for v in results_by_speaker.values())
    print(f"\n✓ Total clean d-vectors: {total}")
    
    return results_by_speaker


def discover_all_available_speakers(librispeech_path, exclude_speakers=None, librispeech_subsets=None):
    """
    Discover all available speakers across all LibriSpeech subsets
    
    Args:
        librispeech_path: Path to LibriSpeech directory
        exclude_speakers: Set or list of speaker IDs to exclude
    
    Returns:
        List of speaker IDs (as strings)
    """
    print(f"\n🔍 Discovering available speakers in LibriSpeech...")
    
    librispeech_path = Path(librispeech_path)
    
    # Common LibriSpeech subsets to search
    if librispeech_subsets is None:
        subsets = ['dev-clean', 'dev-other',
                   'train-clean-100', 'train-clean-360', 'train-other-500']
    else:
        subsets = list(librispeech_subsets)
    
    all_speakers = set()
    found_subsets = []
    
    for subset in subsets:
        subset_path = librispeech_path / subset
        if subset_path.exists():
            # Get all speaker directories
            speaker_dirs = [d for d in subset_path.iterdir() if d.is_dir() and d.name.isdigit()]
            subset_speakers = [d.name for d in speaker_dirs]
            all_speakers.update(subset_speakers)
            found_subsets.append(subset)
            print(f"  Found {len(subset_speakers)} speakers in {subset}")
    
    if not found_subsets:
        print(f"  ⚠️  No LibriSpeech subsets found in: {librispeech_path}")
        return []
    
    print(f"  Searched {len(found_subsets)} subsets: {', '.join(found_subsets)}")
    
    # Convert to sorted list
    all_speakers = sorted(list(all_speakers))
    
    # Filter out excluded speakers
    if exclude_speakers is not None:
        exclude_set = set(exclude_speakers) if not isinstance(exclude_speakers, set) else exclude_speakers
        all_speakers = [spk for spk in all_speakers if spk not in exclude_set]
    
    print(f"✓ Found {len(all_speakers)} unique speakers across all subsets (after exclusions)")
    
    return all_speakers


def detect_speakers_from_overlap_samples(overlap_dirs):
    """Detect which speakers appear in overlap samples"""
    print(f"\n🔍 Detecting speakers from overlap samples...")
    
    # Normalize to list
    if isinstance(overlap_dirs, str):
        overlap_dirs = [overlap_dirs]
    
    all_main_speakers = set()
    all_other_speakers = set()
    
    for overlap_dir in overlap_dirs:
        overlap_path = Path(overlap_dir)
        
        # Try to load speaker_config.json first (if generated by new script)
        speaker_config_path = overlap_path / 'speaker_config.json'
        if speaker_config_path.exists():
            with open(speaker_config_path, 'r') as f:
                speaker_config = json.load(f)
                # Handle new format with list of main speakers
                if speaker_config.get('main_speakers'):
                    all_main_speakers.update(speaker_config['main_speakers'])
                    print(f"  From {overlap_path.name}/speaker_config.json: {len(speaker_config['main_speakers'])} main speaker(s)")
                # Handle old format with single main speaker
                elif speaker_config.get('main_speaker'):
                    all_main_speakers.add(speaker_config['main_speaker'])
                    print(f"  From {overlap_path.name}/speaker_config.json: 1 main speaker")
                
                # Load other speakers if available
                if speaker_config.get('other_speakers'):
                    all_other_speakers.update(speaker_config['other_speakers'])
            continue
        
        # Fallback: read from utt2spk to extract main speakers
        utt2spk_path = overlap_path / 'utt2spk'
        if utt2spk_path.exists():
            with open(utt2spk_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        speaker_id = parts[1]
                        all_main_speakers.add(speaker_id)
            print(f"  From {overlap_path.name}/utt2spk: Detected main speakers from utterance labels")
            continue
        
        # Fallback: read from metadata.json
        metadata_path = overlap_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_list = json.load(f)
            
            for metadata in metadata_list:
                main_spk = metadata.get('main_speaker', '')
                other_spks = metadata.get('other_speakers', [])
                if main_spk:
                    all_main_speakers.add(main_spk)
                if isinstance(other_spks, list):
                    all_other_speakers.update(other_spks)
                elif other_spks:  # Single speaker (older format)
                    all_other_speakers.add(other_spks)
            
            print(f"  From {overlap_path.name}/metadata.json: Detected speakers from metadata")
            continue
        
        print(f"  ⚠️  No speaker information found in {overlap_path.name}")
    
    print(f"✓ Detected {len(all_main_speakers)} main speaker(s){': ' + ', '.join(sorted(all_main_speakers)) if all_main_speakers else ' (none in metadata)'}")
    print(f"✓ Detected {len(all_other_speakers)} other speakers (from config if available)")
    
    return sorted(list(all_main_speakers)), sorted(list(all_other_speakers))


def extract_overlap_utterance_dvectors(overlap_dirs, sample_rate=16000, device='cuda'):
    """
    Extract d-vectors from overlap samples
    
    Args:
        overlap_dirs: Single directory path (str) or list of directory paths
        sample_rate: Audio sample rate
        device: Device for d-vector extraction
    
    Returns:
        Dict of overlap d-vectors from all directories
    """
    # Normalize to list
    if isinstance(overlap_dirs, str):
        overlap_dirs = [overlap_dirs]
    
    print(f"\n🎤 Extracting OVERLAP d-vectors from {len(overlap_dirs)} directory(ies)...")
    
    # Initialize encoder once
    if torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cuda')
        print(f"  Using device: CUDA")
    else:
        device_obj = torch.device('cpu')
        print(f"  Using device: CPU")
    
    encoder = VoiceEncoder(device=device_obj)
    
    all_results = {}
    
    for dir_idx, overlap_dir in enumerate(overlap_dirs, 1):
        print(f"\n  Processing directory {dir_idx}/{len(overlap_dirs)}: {overlap_dir}")
        
        # Load metadata
        metadata_path = Path(overlap_dir) / 'metadata.json'
        if not metadata_path.exists():
            print(f"    ⚠️  Metadata not found: {metadata_path}")
            continue
        
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        print(f"    Samples in directory: {len(metadata_list)}")
        
        audio_dir = Path(overlap_dir) / 'audio'
        dir_results = {}
        
        for i, metadata in enumerate(metadata_list):
            try:
                output_name = metadata['output_name']
                audio_path = audio_dir / f"{output_name}.wav"
                
                if not audio_path.exists():
                    continue
                
                # Load and extract
                audio, sr = librosa.load(audio_path, sr=sample_rate)
                
                fbanks = librosa.feature.melspectrogram(
                    y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
                ).astype('float32').T
                
                with torch.no_grad():
                    fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
                    dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()
                
                dvector = dvector / np.linalg.norm(dvector)
                
                # Use directory-prefixed sample_id to ensure uniqueness across directories
                unique_key = f"{Path(overlap_dir).name}_{metadata['sample_id']}"
                
                other_speakers = metadata.get('other_speakers', [])
                main_utterance = metadata.get('main_utterance')
                if isinstance(main_utterance, str) and len(main_utterance) > 0:
                    # Normalize to stem so it matches clean d-vector keys extracted from file stems
                    main_utterance = Path(main_utterance).stem

                dir_results[unique_key] = {
                    'dvector': dvector,
                    'speaker': other_speakers[0] if len(other_speakers) > 0 else '',
                    'other_speaker': other_speakers[0] if len(other_speakers) > 0 else '',
                    'other_speakers': other_speakers,
                    'main_speaker': metadata.get('main_speaker', ''),
                    'main_utterance': main_utterance,
                    'sample_id': metadata.get('sample_id', ''),
                    'output_name': metadata.get('output_name', ''),
                    'source_dir': overlap_dir
                }
                
            except Exception as e:
                continue
        
        print(f"    ✓ Extracted {len(dir_results)} d-vectors from this directory")
        all_results.update(dir_results)
    
    print(f"\n✓ Total overlap d-vectors from all directories: {len(all_results)}")
    
    return all_results


def extract_non_target_overlap_dvectors(overlap_dir, n_samples=-1, sample_rate=16000, device='cuda'):
    """Extract d-vectors from non-target overlap samples (2 speakers, no target)"""
    print(f"\n🎤 Extracting NON-TARGET OVERLAP d-vectors...")
    
    # Load metadata
    metadata_path = Path(overlap_dir) / 'metadata.json'
    if not metadata_path.exists():
        print(f"  ⚠️  Metadata not found: {metadata_path}")
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"  Total samples available: {len(metadata_list)}")
    
    # Sample if requested
    if n_samples > 0 and len(metadata_list) > n_samples:
        np.random.seed(RANDOM_SEED)
        metadata_list = list(np.random.choice(metadata_list, size=n_samples, replace=False))
        print(f"  Using {len(metadata_list)} samples")
    
    # Initialize encoder
    if torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cuda')
    else:
        device_obj = torch.device('cpu')
    
    encoder = VoiceEncoder(device=device_obj)
    
    results = {}
    audio_dir = Path(overlap_dir) / 'audio'
    
    for metadata in metadata_list:
        try:
            output_name = metadata['output_name']
            if not output_name.endswith('.wav'):
                output_name = output_name + '.wav'
            audio_path = audio_dir / output_name
            
            if not audio_path.exists():
                continue
            
            # Load and extract
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            fbanks = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
            ).astype('float32').T
            
            with torch.no_grad():
                fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
                dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()
            
            dvector = dvector / np.linalg.norm(dvector)
            
            # Get overlapping speakers
            overlapping = metadata.get('overlapping_speakers', [])
            if isinstance(overlapping, list):
                speakers = ','.join(str(s) for s in overlapping)
            else:
                speakers = str(overlapping)
            
            results[metadata['sample_id']] = {
                'dvector': dvector,
                'speakers': speakers,
                'is_non_target': True
            }
            
        except Exception as e:
            continue
    
    print(f"✓ Total non-target overlap d-vectors: {len(results)}")
    
    return results


def create_non_target_representation(dvector_dim, strategy='zero', other_speakers_dvectors=None):
    """
    Create target representation for non-target overlap samples
    
    Args:
        dvector_dim: Dimension of d-vector
        strategy: 'zero', 'mean', or 'random'
        other_speakers_dvectors: Dict of other speakers' clean d-vectors (for 'mean' strategy)
    
    Returns:
        Target d-vector for non-target overlaps
    """
    if strategy == 'zero':
        # Zero vector - model learns to output zeros for non-target
        return np.zeros(dvector_dim, dtype=np.float32)
    
    elif strategy == 'mean' and other_speakers_dvectors is not None:
        # Mean of other speakers - represents "not main speaker"
        all_dvectors = []
        for speaker_dvectors in other_speakers_dvectors.values():
            for data in speaker_dvectors.values():
                all_dvectors.append(data['dvector'])
        
        if len(all_dvectors) > 0:
            mean_dvector = np.mean(all_dvectors, axis=0)
            return mean_dvector / np.linalg.norm(mean_dvector)
        else:
            return np.zeros(dvector_dim, dtype=np.float32)
    
    elif strategy == 'random':
        # Random normalized vector - model learns to reject non-target
        random_vec = np.random.randn(dvector_dim).astype(np.float32)
        return random_vec / np.linalg.norm(random_vec)
    
    else:
        return np.zeros(dvector_dim, dtype=np.float32)


def compute_other_main_speakers_averages(clean_dvectors_by_speaker, overlap_dvectors, main_speakers, other_main_speakers):
    """
    Compute average d-vectors for other main speakers (from their singles and overlaps)
    
    NOTE: This computes the average from both singles AND overlaps for each speaker.
    However, only the single utterances will be mapped to this average during training.
    Overlaps will still map to clean main speaker via create_training_pairs().
    
    Args:
        clean_dvectors_by_speaker: Dict of {speaker_id: {utterance_key: data}}
        overlap_dvectors: Dict of overlap d-vectors
        main_speakers: List of main speaker IDs (the true targets)
        other_main_speakers: List of other main speaker IDs (to average)
    
    Returns:
        Dict of {speaker_id: average_dvector} for other main speakers
    """
    print(f"\n📊 Computing average d-vectors for other main speakers...")
    
    averages = {}
    
    for speaker_id in other_main_speakers:
        all_dvectors = []
        
        # Collect single utterance d-vectors
        if speaker_id in clean_dvectors_by_speaker:
            for data in clean_dvectors_by_speaker[speaker_id].values():
                all_dvectors.append(data['dvector'])
        
        # Collect overlap d-vectors where this speaker is the main speaker
        for data in overlap_dvectors.values():
            if data.get('main_speaker') == speaker_id:
                all_dvectors.append(data['dvector'])
        
        if len(all_dvectors) > 0:
            avg_dvector = np.mean(all_dvectors, axis=0)
            avg_dvector = avg_dvector / np.linalg.norm(avg_dvector)  # Normalize
            averages[speaker_id] = avg_dvector
            print(f"  Speaker {speaker_id}: {len(all_dvectors)} samples averaged")
        else:
            print(f"  ⚠️  Speaker {speaker_id}: No samples found")
    
    print(f"✓ Computed averages for {len(averages)} other main speakers")
    return averages


def compute_global_non_main_average(clean_dvectors_by_speaker, overlap_dvectors, main_speaker_id, 
                                   other_speakers_dvectors_dict=None, additional_speakers_dvectors_dict=None):
    """
    Compute a SINGLE global average d-vector from ALL non-main-speaker utterances
    This includes:
    - All other main speakers' singles
    - All other main speakers' overlaps (where they are the main speaker)
    - All other speakers' singles (for identity)
    - All additional identity speakers' singles
    - All overlaps where the main speaker is NOT the target main speaker
    
    Args:
        clean_dvectors_by_speaker: Dict of {speaker_id: {utterance_key: data}}
        overlap_dvectors: Dict of overlap d-vectors
        main_speaker_id: The primary target speaker ID (e.g., '84')
        other_speakers_dvectors_dict: Dict of other speakers d-vectors (for identity)
        additional_speakers_dvectors_dict: Dict of additional identity speakers d-vectors
    
    Returns:
        Single averaged d-vector representing all non-main-speaker utterances
    """
    print(f"\n📊 Computing GLOBAL average d-vector for all non-main-speaker utterances...")
    print(f"  Target main speaker: {main_speaker_id}")
    
    all_dvectors = []
    counts = {'other_main_singles': 0, 'other_main_overlaps': 0, 'other_speakers': 0, 
              'additional_speakers': 0, 'non_target_overlaps': 0}
    
    # 1. Collect singles from all speakers EXCEPT the main target speaker
    for speaker_id, speaker_dvectors in clean_dvectors_by_speaker.items():
        if speaker_id != main_speaker_id:
            for data in speaker_dvectors.values():
                all_dvectors.append(data['dvector'])
                counts['other_main_singles'] += 1
    
    # 2. Collect overlaps where target speaker does NOT appear (neither as main nor as overlapping speaker)
    for data in overlap_dvectors.values():
        main_spk = data.get('main_speaker')
        other_spk = data.get('speaker')  # The overlapping speaker
        
        # Only include if target speaker is neither the main nor the overlapping speaker
        if main_spk != main_speaker_id and other_spk != main_speaker_id:
            all_dvectors.append(data['dvector'])
            counts['other_main_overlaps'] += 1
    
    # 3. Collect other speakers (for identity mapping)
    if other_speakers_dvectors_dict:
        for speaker_id, speaker_dvectors in other_speakers_dvectors_dict.items():
            for data in speaker_dvectors.values():
                all_dvectors.append(data['dvector'])
                counts['other_speakers'] += 1
    
    # 4. Collect additional identity speakers
    if additional_speakers_dvectors_dict:
        for speaker_id, speaker_dvectors in additional_speakers_dvectors_dict.items():
            for data in speaker_dvectors.values():
                all_dvectors.append(data['dvector'])
                counts['additional_speakers'] += 1
    
    if len(all_dvectors) == 0:
        print(f"  ⚠️  Warning: No non-main-speaker utterances found!")
        return None
    
    # Compute global average
    global_avg = np.mean(all_dvectors, axis=0)
    global_avg = global_avg / np.linalg.norm(global_avg)  # Normalize
    
    print(f"  Averaged {len(all_dvectors)} total d-vectors:")
    print(f"    - Other main speakers' singles: {counts['other_main_singles']}")
    print(f"    - Other main speakers' overlaps: {counts['other_main_overlaps']}")
    print(f"    - Other speakers (identity): {counts['other_speakers']}")
    print(f"    - Additional speakers (identity): {counts['additional_speakers']}")
    print(f"✓ Computed single global average for all non-main-speaker utterances")
    
    return global_avg


def create_other_main_speakers_pairs(clean_dvectors_by_speaker, overlap_dvectors, other_main_speakers, n_pairs_per_speaker=-1):
    """
    Create pairs for other main speakers: (singles + overlaps) -> average d-vector
    This teaches the model to map other main speakers' singles AND overlaps to their average representation
    
    Args:
        clean_dvectors_by_speaker: Dict of {speaker_id: {utterance_key: data}}
        overlap_dvectors: Dict of overlap d-vectors
        other_main_speakers: List of other main speaker IDs
        n_pairs_per_speaker: Number of pairs per speaker (-1 = use all available)
    
    Returns:
        List of (input_key, speaker_id, type) tuples where speaker_id indicates which average to use as target
    """
    print(f"\n🔗 Creating pairs for other main speakers (singles + overlaps -> average)...")
    
    all_pairs = []
    
    for speaker_id in other_main_speakers:
        speaker_pairs = []
        
        # Collect single utterance keys
        if speaker_id in clean_dvectors_by_speaker:
            for key in clean_dvectors_by_speaker[speaker_id].keys():
                speaker_pairs.append((key, speaker_id, 'single'))
        
        # Collect overlap keys where this speaker is the main speaker
        for key, data in overlap_dvectors.items():
            if data.get('main_speaker') == speaker_id:
                speaker_pairs.append((key, speaker_id, 'overlap'))
        
        if len(speaker_pairs) == 0:
            print(f"  Speaker {speaker_id}: No samples found")
            continue
        
        # Sample if requested
        if n_pairs_per_speaker > 0 and len(speaker_pairs) < n_pairs_per_speaker:
            # Not enough samples, use all
            selected_pairs = speaker_pairs
        elif n_pairs_per_speaker > 0:
            # Sample without replacement
            selected_pairs = list(np.random.choice(len(speaker_pairs), size=n_pairs_per_speaker, replace=False))
            selected_pairs = [speaker_pairs[i] for i in selected_pairs]
        else:
            # Use all
            selected_pairs = speaker_pairs
        
        all_pairs.extend(selected_pairs)
        
        # Count singles vs overlaps
        n_singles = sum(1 for _, _, type_ in selected_pairs if type_ == 'single')
        n_overlaps = sum(1 for _, _, type_ in selected_pairs if type_ == 'overlap')
        print(f"  Speaker {speaker_id}: {len(selected_pairs)} pairs ({n_singles} singles + {n_overlaps} overlaps)")
    
    print(f"✓ Created {len(all_pairs)} pairs for other main speakers")
    return all_pairs


def create_global_average_pairs(clean_dvectors_by_speaker, overlap_dvectors, main_speaker_id,
                                other_speakers_dvectors_dict=None, additional_speakers_dvectors_dict=None,
                                n_pairs_per_speaker=-1):
    """
    Create pairs that map ALL non-main-speaker utterances to a SINGLE global average
    This includes:
    - All other main speakers' singles
    - All other main speakers' overlaps
    - All other speakers' singles (identity)
    - All additional identity speakers' singles
    
    Args:
        clean_dvectors_by_speaker: Dict of {speaker_id: {utterance_key: data}}
        overlap_dvectors: Dict of overlap d-vectors
        main_speaker_id: The primary target speaker ID (e.g., '84')
        other_speakers_dvectors_dict: Dict of other speakers d-vectors (for identity)
        additional_speakers_dvectors_dict: Dict of additional identity speakers d-vectors
        n_pairs_per_speaker: Number of pairs per speaker (-1 = use all available)
    
    Returns:
        List of (input_key, 'global_avg', type) tuples - all map to the same global average
    """
    print(f"\n🔗 Creating global average pairs (all non-main-speaker -> single global average)...")
    print(f"  Target main speaker: {main_speaker_id}")
    
    all_pairs = []
    
    # 1. Singles from all speakers EXCEPT the main target speaker
    for speaker_id, speaker_dvectors in clean_dvectors_by_speaker.items():
        if speaker_id != main_speaker_id:
            speaker_keys = list(speaker_dvectors.keys())
            
            # Sample if requested
            if n_pairs_per_speaker > 0 and len(speaker_keys) > n_pairs_per_speaker:
                selected_keys = list(np.random.choice(speaker_keys, size=n_pairs_per_speaker, replace=False))
            else:
                selected_keys = speaker_keys
            
            for key in selected_keys:
                all_pairs.append((key, 'global_avg', 'single'))
    
    # 2. Overlaps where target speaker does NOT appear (neither as main nor as overlapping speaker)
    non_target_overlaps = []
    for key, data in overlap_dvectors.items():
        main_spk = data.get('main_speaker')
        other_spk = data.get('speaker')  # The overlapping speaker
        
        # Only include if target speaker is neither the main nor the overlapping speaker
        if main_spk != main_speaker_id and other_spk != main_speaker_id:
            non_target_overlaps.append((key, 'global_avg', 'overlap'))
    
    all_pairs.extend(non_target_overlaps)
    
    # 3. Other speakers (for identity mapping)
    if other_speakers_dvectors_dict:
        for speaker_id, speaker_dvectors in other_speakers_dvectors_dict.items():
            speaker_keys = list(speaker_dvectors.keys())
            
            # Sample if requested
            if n_pairs_per_speaker > 0 and len(speaker_keys) > n_pairs_per_speaker:
                selected_keys = list(np.random.choice(speaker_keys, size=n_pairs_per_speaker, replace=False))
            else:
                selected_keys = speaker_keys
            
            for key in selected_keys:
                all_pairs.append((key, 'global_avg', 'other_identity'))
    
    # 4. Additional identity speakers
    if additional_speakers_dvectors_dict:
        for speaker_id, speaker_dvectors in additional_speakers_dvectors_dict.items():
            speaker_keys = list(speaker_dvectors.keys())
            
            # Sample if requested
            if n_pairs_per_speaker > 0 and len(speaker_keys) > n_pairs_per_speaker:
                selected_keys = list(np.random.choice(speaker_keys, size=n_pairs_per_speaker, replace=False))
            else:
                selected_keys = speaker_keys
            
            for key in selected_keys:
                all_pairs.append((key, 'global_avg', 'additional_identity'))
    
    # Count by type
    type_counts = defaultdict(int)
    for _, _, type_ in all_pairs:
        type_counts[type_] += 1
    
    print(f"  Created {len(all_pairs)} total pairs:")
    for type_, count in sorted(type_counts.items()):
        print(f"    - {type_}: {count}")
    
    print(f"✓ All {len(all_pairs)} pairs will map to the SAME global average")
    return all_pairs


def create_identity_pairs(clean_dvectors_by_speaker, n_pairs_per_speaker=-1):
    """
    Create identity pairs: clean speaker -> same clean speaker
    This teaches the model to preserve clean speaker d-vectors unchanged
    
    Args:
        clean_dvectors_by_speaker: Dict of {speaker_id: {utterance_key: data}} or flat dict of d-vectors
        n_pairs_per_speaker: Number of pairs to create PER SPEAKER (-1 = use all available)
    
    Returns:
        List of (clean_key, clean_key, speaker_id, 'identity') tuples
    """
    all_identity_pairs = []
    
    # Check if input is nested (by speaker) or flat
    if len(clean_dvectors_by_speaker) == 0:
        return []
    
    first_value = next(iter(clean_dvectors_by_speaker.values()))
    
    # If nested structure (speaker_id -> utterances)
    if isinstance(first_value, dict) and 'dvector' not in first_value:
        # Process per speaker
        for speaker_id, speaker_dvectors in clean_dvectors_by_speaker.items():
            speaker_keys = list(speaker_dvectors.keys())
            
            if len(speaker_keys) == 0:
                continue
            
            # Determine how many pairs to create for this speaker
            if n_pairs_per_speaker == -1 or n_pairs_per_speaker >= len(speaker_keys):
                # Use all available clean utterances for this speaker
                speaker_pairs = [(key, key, speaker_id, 'identity') for key in speaker_keys]
            else:
                # Sample subset for this speaker
                sampled_keys = np.random.choice(speaker_keys, size=min(n_pairs_per_speaker, len(speaker_keys)), replace=False)
                speaker_pairs = [(key, key, speaker_id, 'identity') for key in sampled_keys]
            
            all_identity_pairs.extend(speaker_pairs)
    else:
        # Flat structure (all utterances together) - legacy behavior
        # Extract speaker_id from the data itself
        clean_keys = list(clean_dvectors_by_speaker.keys())
        
        if n_pairs_per_speaker == -1 or n_pairs_per_speaker >= len(clean_keys):
            all_identity_pairs = [(key, key, clean_dvectors_by_speaker[key].get('speaker', 'unknown'), 'identity') for key in clean_keys]
        else:
            sampled_keys = np.random.choice(clean_keys, size=n_pairs_per_speaker, replace=False)
            all_identity_pairs = [(key, key, clean_dvectors_by_speaker[key].get('speaker', 'unknown'), 'identity') for key in sampled_keys]
    
    return all_identity_pairs


def create_training_pairs(overlap_dvectors, clean_main_dvectors, main_speaker_ids, n_pairs_per_dir=-1):
    """
    Create (overlap, clean_main) pairs for training
    
    Each overlap sample (main + other) is paired with the EXACT clean main utterance
    that was used during overlap generation (`main_utterance` in metadata).
    ONLY processes the FIRST main speaker (main_speaker_ids[0])
    Other main speakers are handled by create_other_main_speakers_pairs()
    
    Args:
        overlap_dvectors: Dict of overlap d-vectors (with 'source_dir' field)
        clean_main_dvectors: Dict of clean main speaker d-vectors
        main_speaker_ids: List of main speaker IDs (only first one is used)
        n_pairs_per_dir: Number of pairs to create PER SOURCE DIRECTORY (-1 = use all available from each)
    
    Returns:
        List of (overlap_key, clean_key) tuples
    """
    print(f"\n🔗 Creating overlap training pairs (overlap -> clean main speaker)...")
    print(f"  Matching policy: STRICT exact main utterance match from overlap metadata")
    # Only use the first main speaker
    primary_speaker = main_speaker_ids[0] if len(main_speaker_ids) > 0 else None
    if primary_speaker is None:
        print(f"  ⚠️ No main speakers provided!")
        return []
    print(f"  Primary target speaker: {primary_speaker} (others handled separately)")
    
    pairs = []
    skipped_missing_main_utt = 0
    skipped_missing_clean_match = 0
    
    if len(clean_main_dvectors) == 0:
        print(f"  ⚠️ No clean main speaker utterances found!")
        return pairs
    
    # Group overlap samples by source directory (only primary speaker)
    overlaps_by_dir = defaultdict(list)
    for key, data in overlap_dvectors.items():
        main_spk = data.get('main_speaker', '')
        
        # Check if primary speaker is the designated main speaker
        # OR if main_speaker is null/empty, check if primary speaker is in other_speakers
        if main_spk == primary_speaker:
            source_dir = data.get('source_dir', 'unknown')
            overlaps_by_dir[source_dir].append(key)
        elif not main_spk:  # main_speaker is null/empty
            # Check if primary speaker appears in other_speakers list
            other_spks = data.get('other_speakers', [])
            if primary_speaker in other_spks:
                source_dir = data.get('source_dir', 'unknown')
                overlaps_by_dir[source_dir].append(key)
    
    if len(overlaps_by_dir) == 0:
        print(f"  ⚠️ No overlap samples found with primary speaker {primary_speaker}!")
        return pairs
    
    print(f"  Found overlap samples from {len(overlaps_by_dir)} directory(ies)")
    
    # Create pairs for each directory
    for source_dir, dir_overlap_keys in overlaps_by_dir.items():
        print(f"\n  Processing directory: {Path(source_dir).name}")
        print(f"    Available samples: {len(dir_overlap_keys)}")
        
        # Determine how many pairs to create from this directory
        if n_pairs_per_dir == -1 or n_pairs_per_dir >= len(dir_overlap_keys):
            # Use all overlap samples from this directory
            selected_overlap_keys = dir_overlap_keys
        else:
            # Sample subset from this directory
            selected_overlap_keys = list(np.random.choice(dir_overlap_keys, size=n_pairs_per_dir, replace=False))
        
        print(f"    Using: {len(selected_overlap_keys)} samples")
        
        for overlap_key in selected_overlap_keys:
            overlap_item = overlap_dvectors[overlap_key]

            # Main speaker and exact clean utterance used in mixing
            main_spk = overlap_item.get('main_speaker', '')
            main_utt = overlap_item.get('main_utterance', None)

            # Require explicit main utterance reference for strict pairing
            if not main_utt:
                skipped_missing_main_utt += 1
                continue

            # Find exact clean key match by utterance key + same main speaker
            if (
                main_utt in clean_main_dvectors
                and clean_main_dvectors[main_utt].get('speaker', '') == main_spk
            ):
                pairs.append((overlap_key, main_utt))
            else:
                skipped_missing_clean_match += 1
    
    print(f"\n✓ Created {len(pairs)} total overlap training pairs")
    print(f"  All pairs: overlap (main + other) -> clean main speaker")

    if skipped_missing_main_utt > 0:
        print(f"  ⚠️ Skipped {skipped_missing_main_utt} overlap samples: missing main_utterance in metadata")
    if skipped_missing_clean_match > 0:
        print(f"  ⚠️ Skipped {skipped_missing_clean_match} overlap samples: exact clean main utterance not found")

    if skipped_missing_main_utt > 0 or skipped_missing_clean_match > 0:
        raise ValueError(
            "Strict overlap->clean pairing failed: every overlap sample must provide a valid "
            "`main_utterance` and that utterance must exist in clean_main_dvectors for the same main speaker. "
            "Fix metadata/clean utterance alignment or disable strict matching logic in create_training_pairs()."
        )

    if len(pairs) == 0:
        print(f"  ❌ No valid overlap->clean pairs after strict utterance matching")
    
    # Show distribution by source directory and other speaker
    dir_counts = defaultdict(int)
    for overlap_key, _ in pairs:
        source_dir = overlap_dvectors[overlap_key].get('source_dir', 'unknown')
        dir_counts[Path(source_dir).name] += 1
    
    print(f"\n  Pairs by source directory:")
    for dir_name, count in sorted(dir_counts.items()):
        print(f"    {dir_name}: {count} pairs")
    
    return pairs


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir):
    """Train the autoencoder"""
    print(f"\n🏋️  Training autoencoder...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_mse_loss': [],
        'train_cosine_loss': [],
        'train_negative_loss': [],
        'val_loss': [],
        'val_mse_loss': [],
        'val_cosine_loss': [],
        'val_negative_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_cosine_loss = 0.0
        train_negative_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) >= 2:
                overlap, clean = batch[0], batch[1]
            else:
                raise ValueError("Unexpected batch format in train_loader")
            overlap = overlap.to(device)
            clean = clean.to(device)
            clean_norm = F.normalize(clean, p=2, dim=1, eps=LOSS_COSINE_EPS)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(overlap)
            mse_loss = F.mse_loss(reconstructed, clean_norm)
            cosine_loss = 1.0 - F.cosine_similarity(
                reconstructed, clean_norm, dim=1, eps=LOSS_COSINE_EPS
            ).mean()
            if overlap.size(0) > 1:
                shuffle = torch.randperm(overlap.size(0), device=overlap.device)
                neg_target = clean_norm[shuffle]
                neg_cos = F.cosine_similarity(
                    reconstructed, neg_target, dim=1, eps=LOSS_COSINE_EPS
                )
                negative_loss = F.relu(neg_cos - NEGATIVE_CONTRASTIVE_MARGIN).mean()
            else:
                negative_loss = torch.tensor(0.0, device=overlap.device)
            loss = (
                (LOSS_MSE_WEIGHT * mse_loss)
                + (LOSS_COSINE_WEIGHT * cosine_loss)
                + (NEGATIVE_CONTRASTIVE_WEIGHT * negative_loss)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse_loss += mse_loss.item()
            train_cosine_loss += cosine_loss.item()
            train_negative_loss += negative_loss.item()
        
        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_cosine_loss /= len(train_loader)
        train_negative_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse_loss = 0.0
        val_cosine_loss = 0.0
        val_negative_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) >= 2:
                    overlap, clean = batch[0], batch[1]
                else:
                    raise ValueError("Unexpected batch format in val_loader")
                overlap = overlap.to(device)
                clean = clean.to(device)
                clean_norm = F.normalize(clean, p=2, dim=1, eps=LOSS_COSINE_EPS)
                
                reconstructed = model(overlap)
                mse_loss = F.mse_loss(reconstructed, clean_norm)
                cosine_loss = 1.0 - F.cosine_similarity(
                    reconstructed, clean_norm, dim=1, eps=LOSS_COSINE_EPS
                ).mean()
                if overlap.size(0) > 1:
                    shuffle = torch.randperm(overlap.size(0), device=overlap.device)
                    neg_target = clean_norm[shuffle]
                    neg_cos = F.cosine_similarity(
                        reconstructed, neg_target, dim=1, eps=LOSS_COSINE_EPS
                    )
                    negative_loss = F.relu(neg_cos - NEGATIVE_CONTRASTIVE_MARGIN).mean()
                else:
                    negative_loss = torch.tensor(0.0, device=overlap.device)
                loss = (
                    (LOSS_MSE_WEIGHT * mse_loss)
                    + (LOSS_COSINE_WEIGHT * cosine_loss)
                    + (NEGATIVE_CONTRASTIVE_WEIGHT * negative_loss)
                )
                val_loss += loss.item()
                val_mse_loss += mse_loss.item()
                val_cosine_loss += cosine_loss.item()
                val_negative_loss += negative_loss.item()
        
        val_loss /= len(val_loader)
        val_mse_loss /= len(val_loader)
        val_cosine_loss /= len(val_loader)
        val_negative_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_mse_loss'].append(train_mse_loss)
        history['train_cosine_loss'].append(train_cosine_loss)
        history['train_negative_loss'].append(train_negative_loss)
        history['val_loss'].append(val_loss)
        history['val_mse_loss'].append(val_mse_loss)
        history['val_cosine_loss'].append(val_cosine_loss)
        history['val_negative_loss'].append(val_negative_loss)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Mixed: {train_loss:.6f} (MSE {train_mse_loss:.6f}, Cos {train_cosine_loss:.6f}, Neg {train_negative_loss:.6f}), "
                f"Val Mixed: {val_loss:.6f} (MSE {val_mse_loss:.6f}, Cos {val_cosine_loss:.6f}, Neg {val_negative_loss:.6f}), "
                f"LR: {current_lr:.6f}"
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = Path(save_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mse_loss': train_mse_loss,
                'train_cosine_loss': train_cosine_loss,
                'val_mse_loss': val_mse_loss,
                'val_cosine_loss': val_cosine_loss,
            }, best_model_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    
    return history, best_val_loss


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    print(f"\n📊 Evaluating model on test set...")
    
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    total_mse_loss = 0.0
    total_cosine_loss = 0.0
    total_negative_loss = 0.0
    
    all_overlap = []
    all_clean = []
    all_reconstructed = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) >= 2:
                overlap, clean = batch[0], batch[1]
            else:
                raise ValueError("Unexpected batch format in test_loader")
            overlap = overlap.to(device)
            clean = clean.to(device)
            clean_norm = F.normalize(clean, p=2, dim=1, eps=LOSS_COSINE_EPS)
            
            reconstructed = model(overlap)
            mse_loss = F.mse_loss(reconstructed, clean_norm)
            cosine_loss = 1.0 - F.cosine_similarity(
                reconstructed, clean_norm, dim=1, eps=LOSS_COSINE_EPS
            ).mean()
            if overlap.size(0) > 1:
                shuffle = torch.randperm(overlap.size(0), device=overlap.device)
                neg_target = clean_norm[shuffle]
                neg_cos = F.cosine_similarity(
                    reconstructed, neg_target, dim=1, eps=LOSS_COSINE_EPS
                )
                negative_loss = F.relu(neg_cos - NEGATIVE_CONTRASTIVE_MARGIN).mean()
            else:
                negative_loss = torch.tensor(0.0, device=overlap.device)
            loss = (
                (LOSS_MSE_WEIGHT * mse_loss)
                + (LOSS_COSINE_WEIGHT * cosine_loss)
                + (NEGATIVE_CONTRASTIVE_WEIGHT * negative_loss)
            )
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_cosine_loss += cosine_loss.item()
            total_negative_loss += negative_loss.item()
            
            all_overlap.append(overlap.cpu().numpy())
            all_clean.append(clean_norm.cpu().numpy())
            all_reconstructed.append(reconstructed.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    avg_mse_loss = total_mse_loss / len(test_loader)
    avg_cosine_loss = total_cosine_loss / len(test_loader)
    avg_negative_loss = total_negative_loss / len(test_loader)
    
    # Concatenate all batches
    all_overlap = np.concatenate(all_overlap, axis=0)
    all_clean = np.concatenate(all_clean, axis=0)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # Compute cosine similarity
    cosine_sims = []
    for i in range(len(all_clean)):
        clean_norm = all_clean[i] / np.linalg.norm(all_clean[i])
        recon_norm = all_reconstructed[i] / np.linalg.norm(all_reconstructed[i])
        cos_sim = np.dot(clean_norm, recon_norm)
        cosine_sims.append(cos_sim)
    
    cosine_sims = np.array(cosine_sims)
    
    print(f"\n  Test Results:")
    print(f"    Mixed Loss: {avg_loss:.6f}")
    print(f"    MSE Loss: {avg_mse_loss:.6f}")
    print(f"    Cosine Loss: {avg_cosine_loss:.6f}")
    print(f"    Negative Loss: {avg_negative_loss:.6f}")
    print(f"    Cosine Similarity: {cosine_sims.mean():.4f} ± {cosine_sims.std():.4f}")
    print(f"    Min Cosine Sim: {cosine_sims.min():.4f}")
    print(f"    Max Cosine Sim: {cosine_sims.max():.4f}")
    
    return {
        'mixed_loss': avg_loss,
        'mse_loss': avg_mse_loss,
        'cosine_loss': avg_cosine_loss,
        'negative_loss': avg_negative_loss,
        'cosine_similarity': cosine_sims,
        'overlap': all_overlap,
        'clean': all_clean,
        'reconstructed': all_reconstructed
    }


def plot_training_history(history, save_path):
    """Plot training and validation loss"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mixed Loss', fontsize=12)
    axes[0].set_title('Training History', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1].plot(history['learning_rate'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training history plot to: {save_path}")
    plt.close()


def plot_reconstruction_quality(eval_results, save_path):
    """Plot reconstruction quality metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cosine similarity distribution
    axes[0].hist(eval_results['cosine_similarity'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(eval_results['cosine_similarity'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {eval_results['cosine_similarity'].mean():.3f}")
    axes[0].set_xlabel('Cosine Similarity', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Reconstruction Quality Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter: overlap vs reconstructed similarity to clean
    overlap_sims = []
    recon_sims = []
    
    for i in range(len(eval_results['clean'])):
        clean_norm = eval_results['clean'][i] / np.linalg.norm(eval_results['clean'][i])
        overlap_norm = eval_results['overlap'][i] / np.linalg.norm(eval_results['overlap'][i])
        recon_norm = eval_results['reconstructed'][i] / np.linalg.norm(eval_results['reconstructed'][i])
        
        overlap_sims.append(np.dot(clean_norm, overlap_norm))
        recon_sims.append(np.dot(clean_norm, recon_norm))
    
    axes[1].scatter(overlap_sims, recon_sims, alpha=0.6, s=30)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (no improvement)')
    axes[1].set_xlabel('Overlap → Clean Similarity', fontsize=12)
    axes[1].set_ylabel('Reconstructed → Clean Similarity', fontsize=12)
    axes[1].set_title('Reconstruction Improvement', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved reconstruction quality plot to: {save_path}")
    plt.close()


def save_model_and_config(model, config, save_dir):
    """Save final model and configuration"""
    save_dir = Path(save_dir)
    
    # Save model weights
    model_path = save_dir / 'final_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved final model to: {model_path}")
    
    # Save configuration (pickle)
    config_path = save_dir / 'config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✓ Saved configuration to: {config_path}")
    
    # Save human-readable configuration summary
    config_txt_path = save_dir / 'config_summary.txt'
    with open(config_txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("D-VECTOR AUTOENCODER TRAINING CONFIGURATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("TRAINING SCHEME\n")
        f.write("-" * 70 + "\n")
        f.write(f"Scheme: {config['training_scheme']}\n")
        
        # Add scheme description
        scheme_descriptions = {
            'mainOnly': 'Main speaker identity + overlaps only',
            'mainOnly_otherSingles': 'Main speaker + additional identity speakers',
            'manyMain': 'Multiple main speakers (may differ from dataset)',
            'manyMain_otherSingles': 'Multiple main speakers + additional identity',
            'sumNotmain': 'Non-main speakers averaged to global representation',
            'sumNotmain_otherSingles': 'Non-main averaging + additional identity'
        }
        f.write(f"Description: {scheme_descriptions.get(config['training_scheme'], 'Custom')}\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Input dimension: {config['input_dim']}\n")
        f.write(f"Hidden dimensions: {config['hidden_dims']}\n")
        f.write(f"Dropout rate: {config['dropout_rate']}\n\n")
        f.write(f"Normalization: {config.get('norm_type', 'batchnorm')}\n")
        f.write(f"Residual connection: {config.get('use_residual', False)}\n")
        f.write(f"Residual init scale: {config.get('residual_scale_init', 0.5)}\n\n")
        
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write(f"Learning rate: {config['learning_rate']}\n")
        f.write(f"Number of epochs: {config['num_epochs']}\n")
        f.write(f"Loss MSE weight: {config.get('loss_mse_weight', 'n/a')}\n")
        f.write(f"Loss cosine weight: {config.get('loss_cosine_weight', 'n/a')}\n")
        f.write(f"Loss negative weight: {config.get('loss_negative_weight', 'n/a')}\n")
        f.write(f"Loss negative margin: {config.get('loss_negative_margin', 'n/a')}\n")
        f.write(f"Validation split: {config['validation_split']:.1%}\n")
        f.write(f"Test split: {config['test_split']:.1%}\n")
        f.write(f"Early stopping patience: {config['early_stopping_patience']}\n")
        f.write(f"Random seed: {config['random_seed']}\n")
        f.write(f"Device: {config['device']}\n")
        if config.get('pretrained_model_path'):
            f.write(f"Fine-tuned from: {config['pretrained_model_path']}\n")
            if config.get('pretrained_model_type'):
                f.write(f"Pretrained model type: {config['pretrained_model_type']}\n")
        else:
            f.write(f"Training from scratch: Yes\n")
        f.write("\n")
        
        f.write("DATA SOURCES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Sample rate: {config['sample_rate']} Hz\n")
        
        # Main speaker data source
        if config.get('single_speaker_datasets'):
            f.write(f"Main speaker data source: Single speaker datasets\n")
            f.write(f"  Number of datasets: {len(config['single_speaker_datasets'])}\n")
            for dataset_path in config['single_speaker_datasets']:
                f.write(f"    - {dataset_path}\n")
        else:
            f.write(f"Main speaker data source: LibriSpeech\n")
            f.write(f"  Path: {config['librispeech_path']}\n")
        
        f.write(f"Single utterances per speaker: {config['n_single_utterances_per_speaker']}\n")
        f.write(f"Overlap sample directories ({len(config['overlap_samples_dirs'])}):\n")
        for i, dir_path in enumerate(config['overlap_samples_dirs'], 1):
            f.write(f"  {i}. {Path(dir_path).name}\n")
        f.write("\n")
        
        f.write("SPEAKER CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Main speakers (primary targets): {', '.join(config['main_speakers'])}\n")
        f.write(f"  Count: {len(config['main_speakers'])}\n")
        f.write(f"  Detection: {'Auto-detected from overlap samples' if config.get('main_speakers_auto_detected', False) else 'Manually configured'}\n")
        
        # Handle auto-detected vs explicit speakers
        if isinstance(config['other_speakers_configured'], str):
            f.write(f"\nOther speakers: {config['other_speakers_configured']}\n")
        else:
            f.write(f"\nTotal OTHER_SPEAKERS configured: {len(config['other_speakers_configured'])}\n")
            f.write(f"  First 10: {', '.join(config['other_speakers_configured'][:10])}\n")
            if len(config['other_speakers_configured']) > 10:
                f.write(f"  ... and {len(config['other_speakers_configured']) - 10} more\n")
        
        f.write(f"\nSpeaker source: {config['other_speakers_source']}\n")
        f.write(f"\nSpeakers used in overlaps ({len(config['overlap_speakers_used'])}):\n")
        f.write(f"  {', '.join(sorted(config['overlap_speakers_used']))}\n")
        
        # Add speaker separation guarantee
        f.write(f"\n🔒 SPEAKER SEPARATION GUARANTEE:\n")
        f.write(f"  ✓ Main speakers: {len(config['main_speakers'])} speakers (targets for extraction)\n")
        f.write(f"  ✓ Overlapping speakers: {len(config['overlap_speakers_used'])} speakers (appear in overlap samples)\n")
        if isinstance(config['other_speakers_configured'], list):
            f.write(f"  ✓ Other speakers: {len(config['other_speakers_configured'])} speakers (identity mapping)\n")
        if config.get('additional_identity_speakers_used'):
            f.write(f"  ✓ Additional identity speakers: {len(config['additional_identity_speakers_used'])} speakers (generalization)\n")
        f.write(f"  ✓ STRICT SEPARATION: All categories are mutually exclusive (no speaker appears in multiple categories)\n")
        
        if config['main_identity_speakers_used']:
            f.write(f"\nMain speakers used for identity mapping ({len(config['main_identity_speakers_used'])}):\n")
            f.write(f"  {', '.join(config['main_identity_speakers_used'])}\n")
        
        if config['other_identity_speakers_used']:
            f.write(f"\nOther speakers used for identity mapping ({len(config['other_identity_speakers_used'])}):\n")
            speakers_list = config['other_identity_speakers_used']
            f.write(f"  First 10: {', '.join(speakers_list[:10])}\n")
            if len(speakers_list) > 10:
                f.write(f"  ... and {len(speakers_list) - 10} more\n")
        
        if config.get('additional_identity_speakers_used'):
            f.write(f"\nAdditional speakers used for identity mapping ({len(config['additional_identity_speakers_used'])}):\n")
            speakers_list = config['additional_identity_speakers_used']
            f.write(f"  First 10: {', '.join(speakers_list[:10])}\n")
            if len(speakers_list) > 10:
                f.write(f"  ... and {len(speakers_list) - 10} more\n")
        
        if config.get('has_silence_identity_pairs', False):
            f.write(f"\nSilence frames used for identity mapping: Yes\n")
        
        if config['other_main_speakers_used']:
            f.write(f"\nOther main speakers (averaged) ({len(config['other_main_speakers_used'])}):\n")
            f.write(f"  {', '.join(config['other_main_speakers_used'])}\n")
        f.write("\n")
        
        f.write("DATASET COMPOSITION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Identity pairs (main): {'Yes' if config['include_identity_pairs'] else 'No'}")
        if config['include_identity_pairs']:
            f.write(f" (n={config['n_identity_pairs'] if config['n_identity_pairs'] > 0 else 'all'})\n")
        else:
            f.write("\n")
        
        f.write(f"Identity pairs (other): {'Yes' if config['include_other_speakers_identity'] else 'No'}")
        if config['include_other_speakers_identity']:
            f.write(f" (n={config['n_other_speakers_identity_pairs'] if config['n_other_speakers_identity_pairs'] > 0 else 'all'})\n")
        else:
            f.write("\n")
        
        f.write(f"Identity pairs (silence): {'Yes' if config.get('include_silence_identity', False) else 'No'}")
        if config.get('include_silence_identity', False):
            f.write(f" (n={config.get('n_silence_identity_pairs', 0) if config.get('n_silence_identity_pairs', 0) > 0 else 'all'})\n")
        else:
            f.write("\n")
        
        f.write(f"Identity pairs (noise->silence): {'Yes' if config.get('include_noise_identity', False) else 'No'}")
        if config.get('include_noise_identity', False):
            f.write(f" (MUSAN {config.get('n_noise_samples_per_type', 0)}/type, RIRs {config.get('n_rir_samples', 0)}, duration {config.get('noise_sample_duration', 1.0)}s)\n")
        else:
            f.write("\n")
        
        f.write(f"Identity pairs (additional): {'Yes' if config.get('include_additional_identity_speakers', False) else 'No'}")
        if config.get('include_additional_identity_speakers', False):
            f.write(f" ({config.get('n_additional_identity_speakers', 0)} speakers, {config.get('n_additional_identity_utterances_per_speaker', 0)} utt/spk, n={config.get('n_additional_identity_pairs_per_speaker', 0) if config.get('n_additional_identity_pairs_per_speaker', 0) > 0 else 'all'})\n")
        else:
            f.write("\n")
        
        f.write(f"Overlap pairs: Yes (n={config['n_overlap_pairs'] if config['n_overlap_pairs'] > 0 else 'all'} per directory)\n")
        
        f.write(f"Other main speakers averaging: {'Yes' if config['include_other_main_speakers_avg'] else 'No'}")
        if config['include_other_main_speakers_avg']:
            f.write(f" (n={config['n_other_main_speakers_pairs_per_speaker'] if config['n_other_main_speakers_pairs_per_speaker'] > 0 else 'all'})\n")
        else:
            f.write("\n")
        
        f.write(f"Non-target overlaps: {'Yes' if config['include_non_target_overlaps'] else 'No'}")
        if config['include_non_target_overlaps']:
            f.write(f" (strategy={config['non_target_strategy']}, n={config['n_non_target_overlap_samples']})\n")
        else:
            f.write("\n")
        f.write("\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples:\n")
        f.write(f"  Train: {config['train_size']} ({config['train_identity_pairs']} identity + {config['train_overlap_pairs']} overlap")
        if config['train_other_main_pairs'] > 0:
            f.write(f" + {config['train_other_main_pairs']} other-main")
        if config['train_non_target_pairs'] > 0:
            f.write(f" + {config['train_non_target_pairs']} non-target")
        f.write(")\n")
        
        f.write(f"  Val:   {config['val_size']} ({config['val_identity_pairs']} identity + {config['val_overlap_pairs']} overlap")
        if config['val_other_main_pairs'] > 0:
            f.write(f" + {config['val_other_main_pairs']} other-main")
        if config['val_non_target_pairs'] > 0:
            f.write(f" + {config['val_non_target_pairs']} non-target")
        f.write(")\n")
        
        f.write(f"  Test:  {config['test_size']} ({config['test_identity_pairs']} identity + {config['test_overlap_pairs']} overlap")
        if config['test_other_main_pairs'] > 0:
            f.write(f" + {config['test_other_main_pairs']} other-main")
        if config['test_non_target_pairs'] > 0:
            f.write(f" + {config['test_non_target_pairs']} non-target")
        f.write(")\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best validation loss: {config['best_val_loss']:.6f}\n")
        if config.get('test_mixed_loss') is not None:
            f.write(f"Test mixed loss: {config['test_mixed_loss']:.6f}\n")
        f.write(f"Test MSE loss: {config['test_mse_loss']:.6f}\n")
        if config.get('test_cosine_loss') is not None:
            f.write(f"Test cosine loss: {config['test_cosine_loss']:.6f}\n")
        f.write(f"Test cosine similarity: {config['test_cosine_similarity_mean']:.4f} ± {config['test_cosine_similarity_std']:.4f}\n\n")
        
        f.write("TRAINING OBJECTIVES\n")
        f.write("-" * 70 + "\n")
        obj_num = 1
        if config['include_identity_pairs']:
            f.write(f"{obj_num}. Identity: Main speakers {', '.join(config['main_speakers'])} (clean → same)\n")
            obj_num += 1
        if config['include_other_speakers_identity']:
            f.write(f"{obj_num}. Identity: Other speakers (clean → same)\n")
            obj_num += 1
        if config.get('include_silence_identity', False):
            f.write(f"{obj_num}. Identity: Silence frames (silence → same)\n")
            obj_num += 1
        if config.get('include_noise_identity', False):
            f.write(f"{obj_num}. Mapping: Noise samples (noise → silence)\n")
            obj_num += 1
        f.write(f"{obj_num}. Denoising: Overlap (main + other) → Clean main (primary speaker only)\n")
        obj_num += 1
        if config['include_other_main_speakers_avg']:
            f.write(f"{obj_num}. Averaging: Other main speakers (singles + overlaps → average)\n")
            obj_num += 1
        if config['include_non_target_overlaps']:
            f.write(f"{obj_num}. Rejection: Non-target overlaps → {config['non_target_strategy']} representation\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Saved human-readable config summary to: {config_txt_path}")


def main():
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Resolve paths
    script_dir = Path(__file__).parent
    librispeech_path = Path(LIBRISPEECH_PATH)
    if not librispeech_path.is_absolute():
        librispeech_path = script_dir / librispeech_path
    
    # Normalize overlap directories to list and resolve paths
    overlap_dirs = OVERLAP_SAMPLES_DIRS if isinstance(OVERLAP_SAMPLES_DIRS, list) else [OVERLAP_SAMPLES_DIRS]
    overlap_dirs_resolved = []
    for overlap_dir in overlap_dirs:
        overlap_path = Path(overlap_dir)
        if not overlap_path.is_absolute():
            overlap_path = script_dir / overlap_path
        overlap_dirs_resolved.append(overlap_path)
    
    save_dir = Path(MODEL_SAVE_DIR)
    if not save_dir.is_absolute():
        save_dir = script_dir / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(DVECTOR_CACHE_DIR)
    if not cache_dir.is_absolute():
        cache_dir = script_dir / cache_dir
    if DVECTOR_CACHE_ENABLED:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("D-VECTOR AUTOENCODER TRAINING PIPELINE")
    print("Extract Main Speaker from Overlap")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  LibriSpeech: {librispeech_path}")
    print(f"  Overlap samples directories: {len(overlap_dirs_resolved)}")
    for idx, overlap_dir in enumerate(overlap_dirs_resolved, 1):
        print(f"    {idx}. {overlap_dir}")
    if INCLUDE_NON_TARGET_OVERLAPS:
        print(f"  Non-target overlap samples: {NON_TARGET_OVERLAP_SAMPLES_DIR}")
        print(f"  Non-target strategy: {NON_TARGET_STRATEGY}")
        print(f"  Non-target samples: {N_NON_TARGET_OVERLAP_SAMPLES}")
    print(f"  Model save dir: {save_dir}")
    print(f"  D-vector cache: {'enabled' if DVECTOR_CACHE_ENABLED else 'disabled'}")
    if DVECTOR_CACHE_ENABLED:
        print(f"  D-vector cache dir: {cache_dir}")
    
    # Determine main speakers (auto-detect if not specified)
    main_speakers = MAIN_SPEAKERS
    main_speakers_auto_detected = False
    
    if main_speakers is None:
        print(f"\n🔍 Auto-detecting main speakers from overlap samples...")
        detected_main, detected_other = detect_speakers_from_overlap_samples(overlap_dirs_resolved)
        
        if detected_main:
            main_speakers = detected_main
            main_speakers_auto_detected = True
            print(f"  ✓ Detected {len(main_speakers)} main speaker(s): {', '.join(main_speakers)}")
        elif N_MAIN_SPEAKERS is not None and N_MAIN_SPEAKERS > 0:
            # Random selection if no detection and N_MAIN_SPEAKERS is specified
            print(f"  No main speakers detected, randomly selecting {N_MAIN_SPEAKERS} from LibriSpeech...")
            available_speakers = discover_all_available_speakers(librispeech_path, set())
            if N_MAIN_SPEAKERS > len(available_speakers):
                print(f"  ⚠️  Warning: Requested {N_MAIN_SPEAKERS} main speakers but only {len(available_speakers)} available")
                main_speakers = available_speakers
            else:
                main_speakers = random.sample(available_speakers, N_MAIN_SPEAKERS)
            main_speakers_auto_detected = True
            print(f"  ✓ Selected main speakers: {', '.join(sorted(main_speakers))}")
        else:
            raise ValueError("No main speakers specified, detected, or random selection configured. "
                           "Set MAIN_SPEAKERS, ensure speaker_config.json exists in overlap dirs, or set N_MAIN_SPEAKERS")
        
        # Update AUTO_DETECT_OTHER_SPEAKERS to reuse detection
        if AUTO_DETECT_OTHER_SPEAKERS and detected_other:
            other_speakers_from_detection = detected_other
        else:
            other_speakers_from_detection = None
    else:
        # Normalize to list if single speaker provided
        if isinstance(main_speakers, str):
            main_speakers = [main_speakers]
        other_speakers_from_detection = None
    
    print(f"  Main speakers (TARGET): {', '.join(main_speakers)}")
    
    # Determine other speakers
    other_speakers = OTHER_SPEAKERS
    if other_speakers is None and AUTO_DETECT_OTHER_SPEAKERS:
        if other_speakers_from_detection is not None:
            # Reuse from main speaker detection
            other_speakers = other_speakers_from_detection
            print(f"  ✓ Using {len(other_speakers)} detected other speakers (from main speaker detection)")
        else:
            print(f"\n🔍 Auto-detecting other speakers from overlap samples...")
            _, detected_other = detect_speakers_from_overlap_samples(overlap_dirs_resolved)
            other_speakers = detected_other
            print(f"  ✓ Will use {len(other_speakers)} detected other speakers")
            print(f"  ℹ️  Note: Main speakers already configured as: {', '.join(main_speakers)}")
    
    if other_speakers is not None:
        print(f"  Other speakers ({len(other_speakers)}): {', '.join(other_speakers[:10])}{'...' if len(other_speakers) > 10 else ''}")
    else:
        print(f"  Other speakers: Will be determined from overlap samples")
    
    print(f"\n  Dataset composition:")
    if INCLUDE_IDENTITY_PAIRS:
        print(f"    Identity pairs - Main speaker (clean->clean): {N_IDENTITY_PAIRS if N_IDENTITY_PAIRS > 0 else 'all available'}")
    if INCLUDE_OTHER_SPEAKERS_IDENTITY:
        print(f"    Identity pairs - Other speakers (clean->clean): {N_OTHER_SPEAKERS_IDENTITY_PAIRS if N_OTHER_SPEAKERS_IDENTITY_PAIRS > 0 else 'all available'}")
    if INCLUDE_SILENCE_IDENTITY:
        print(f"    Identity pairs - Silence frames (silence->silence): {N_SILENCE_IDENTITY_PAIRS if N_SILENCE_IDENTITY_PAIRS > 0 else 'all available'}")
    if INCLUDE_NOISE_IDENTITY:
        print(f"    Identity pairs - Noise samples (noise->silence): MUSAN {N_NOISE_SAMPLES_PER_TYPE}/type, RIRs {N_RIR_SAMPLES}")
    if INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS:
        print(f"    Identity pairs - Additional speakers ({N_ADDITIONAL_IDENTITY_SPEAKERS} speakers, {N_ADDITIONAL_IDENTITY_UTTERANCES_PER_SPEAKER} utt/spk): {N_ADDITIONAL_IDENTITY_PAIRS_PER_SPEAKER if N_ADDITIONAL_IDENTITY_PAIRS_PER_SPEAKER > 0 else 'all available'}")
    print(f"    Overlap pairs (overlap->clean): {N_OVERLAP_PAIRS if N_OVERLAP_PAIRS > 0 else 'all available'} (primary speaker only)")
    if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        print(f"    Other main speakers (singles+overlaps->average): {N_OTHER_MAIN_SPEAKERS_PAIRS_PER_SPEAKER if N_OTHER_MAIN_SPEAKERS_PAIRS_PER_SPEAKER > 0 else 'all available'} per speaker")
    print(f"\n  Training parameters:")
    print(f"    Architecture: {HIDDEN_DIMS}")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Learning rate: {LEARNING_RATE}")
    print(f"    Epochs: {NUM_EPOCHS}")
    print(f"    Val split: {VALIDATION_SPLIT:.0%}")
    print(f"    Test split: {TEST_SPLIT:.0%}")
    print(f"    Device: {DEVICE}")
    print(f"\n🎯 Training objectives:")
    objective_num = 1
    if INCLUDE_IDENTITY_PAIRS:
        if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
            print(f"   {objective_num}. Identity: Clean PRIMARY main speaker {main_speakers[0]} -> same (preserve clean)")
        else:
            print(f"   {objective_num}. Identity: Clean main speakers {', '.join(main_speakers)} -> same (preserve clean)")
        objective_num += 1
    if INCLUDE_OTHER_SPEAKERS_IDENTITY and not INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        print(f"   {objective_num}. Identity: Clean other speakers -> same (preserve clean)")
        objective_num += 1
    if INCLUDE_SILENCE_IDENTITY and not INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        print(f"   {objective_num}. Identity: Silence frames -> same silence (preserve silence)")
        objective_num += 1
    if INCLUDE_NOISE_IDENTITY and not INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        print(f"   {objective_num}. Mapping: Noise samples -> silence (reject noise as silence)")
        objective_num += 1
    if INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS and not INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        print(f"   {objective_num}. Identity: Additional {N_ADDITIONAL_IDENTITY_SPEAKERS} speakers -> same (preserve clean, generalization)")
        objective_num += 1
    print(f"   {objective_num}. Denoising: Overlap ({main_speakers[0]} + other) -> Clean {main_speakers[0]}")
    objective_num += 1
    if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        print(f"   {objective_num}. Global Averaging: ALL non-target utterances -> SINGLE global average")
        print(f"      - Other main speakers' singles (e.g., {', '.join(main_speakers[1:])})") if len(main_speakers) > 1 else None
        print(f"      - Other main speakers' overlaps")
        print(f"      - Other speakers' singles (identity)")
        print(f"      - Additional identity speakers' singles")
        objective_num += 1
    if INCLUDE_NON_TARGET_OVERLAPS:
        print(f"   {objective_num}. Rejection: Non-target overlaps (other1 + other2) -> {NON_TARGET_STRATEGY} representation")
    # Priority: Use single speaker datasets if specified, otherwise extract from LibriSpeech
    if SINGLE_SPEAKER_DATASETS:
        print(f"\n📦 Using single speaker datasets for main speaker clean d-vectors...")
        single_dataset_dirs_resolved = [_resolve_path_str(p, script_dir) for p in SINGLE_SPEAKER_DATASETS]
        clean_main_cache_config = {
            'source': 'single_speaker_datasets',
            'dataset_dirs': single_dataset_dirs_resolved,
            'n_utterances_per_speaker': N_SINGLE_UTTERANCES_PER_SPEAKER,
            'sample_rate': SAMPLE_RATE,
            'device': DEVICE,
            'random_seed': RANDOM_SEED,
        }
        clean_main_dvectors_dict = _extract_with_cache(
            cache_dir,
            'clean_main_single_dataset',
            clean_main_cache_config,
            lambda: extract_single_speaker_dataset_dvectors(
                single_dataset_dirs_resolved,
                N_SINGLE_UTTERANCES_PER_SPEAKER,
                SAMPLE_RATE,
                DEVICE
            )
        )
        
        # Verify all main speakers are covered
        loaded_speakers = set(clean_main_dvectors_dict.keys())
        main_speakers_set = set(main_speakers)
        if not main_speakers_set.issubset(loaded_speakers):
            missing = main_speakers_set - loaded_speakers
            print(f"\n⚠️  Warning: Some main speakers not found in single speaker datasets: {missing}")
            print(f"   Available speakers: {loaded_speakers}")
            print(f"   Will extract missing speakers from LibriSpeech...")
            
            # Extract missing speakers from LibriSpeech
            if missing:
                missing_speakers = sorted(list(missing))
                missing_cache_config = {
                    'source': 'librispeech_missing_main',
                    'librispeech_path': str(librispeech_path.resolve()),
                    'speaker_ids': missing_speakers,
                    'n_utterances_per_speaker': N_SINGLE_UTTERANCES_PER_SPEAKER,
                    'sample_rate': SAMPLE_RATE,
                    'device': DEVICE,
                    'random_seed': RANDOM_SEED,
                    'librispeech_subsets': None,
                    'partition_duration': None,
                }
                missing_dvectors = _extract_with_cache(
                    cache_dir,
                    'clean_main_librispeech_missing',
                    missing_cache_config,
                    lambda: extract_single_utterance_dvectors(
                        str(librispeech_path),
                        missing_speakers,
                        N_SINGLE_UTTERANCES_PER_SPEAKER,
                        SAMPLE_RATE,
                        DEVICE
                    )
                )
                clean_main_dvectors_dict.update(missing_dvectors)
    else:
        print(f"\n📦 Extracting main speaker clean d-vectors from LibriSpeech...")
        main_librispeech_cache_config = {
            'source': 'librispeech_main',
            'librispeech_path': str(librispeech_path.resolve()),
            'speaker_ids': list(main_speakers),
            'n_utterances_per_speaker': N_SINGLE_UTTERANCES_PER_SPEAKER,
            'sample_rate': SAMPLE_RATE,
            'device': DEVICE,
            'random_seed': RANDOM_SEED,
            'librispeech_subsets': None,
            'partition_duration': None,
        }
        clean_main_dvectors_dict = _extract_with_cache(
            cache_dir,
            'clean_main_librispeech',
            main_librispeech_cache_config,
            lambda: extract_single_utterance_dvectors(
                str(librispeech_path),
                main_speakers,  # Main speakers - these are the TARGETS
                N_SINGLE_UTTERANCES_PER_SPEAKER,
                SAMPLE_RATE,
                DEVICE
            )
        )
    
    # Extract overlap samples first to identify which speakers appear in overlaps
    overlap_cache_config = {
        'source': 'overlap_samples',
        'overlap_dirs': [str(d.resolve()) for d in overlap_dirs_resolved],
        'sample_rate': SAMPLE_RATE,
        'device': DEVICE,
    }
    overlap_dvectors = _extract_with_cache(
        cache_dir,
        'overlap_dvectors',
        overlap_cache_config,
        lambda: extract_overlap_utterance_dvectors(
            overlap_dirs_resolved,
            SAMPLE_RATE,
            DEVICE
        )
    )
    
    # Extract d-vectors for OTHER SPEAKERS (for identity mapping as non-target)
    # Include overlap-generator "other speakers" so the model sees explicit clean
    # non-target identities for speakers present in overlap mixtures.
    clean_other_dvectors_dict = {}
    if INCLUDE_OTHER_SPEAKERS_IDENTITY:
        # Identify speakers that appear in overlaps as the secondary/other speaker.
        overlap_other_speakers = set()
        for data in overlap_dvectors.values():
            other_spk = data.get('other_speaker', '')
            if other_spk:
                overlap_other_speakers.add(other_spk)

        configured_other_speakers = set(other_speakers) if other_speakers is not None else set()
        candidate_other_speakers = configured_other_speakers | overlap_other_speakers
        non_target_identity_speakers = sorted(
            [spk for spk in candidate_other_speakers if spk not in main_speakers]
        )

        if len(non_target_identity_speakers) > 0:
            print(f"\n📋 Other-speaker identity mapping (non-target):")
            print(f"   Configured other speakers: {len(configured_other_speakers)}")
            print(f"   Overlap-generator other speakers: {len(overlap_other_speakers)} ({', '.join(sorted(overlap_other_speakers)[:5])}{'...' if len(overlap_other_speakers) > 5 else ''})")
            print(f"   Final non-target identity speakers: {len(non_target_identity_speakers)} ({', '.join(non_target_identity_speakers[:5])}{'...' if len(non_target_identity_speakers) > 5 else ''})")

            other_identity_cache_config = {
                'source': 'librispeech_other_identity',
                'librispeech_path': str(librispeech_path.resolve()),
                'speaker_ids': non_target_identity_speakers,
                'n_utterances_per_speaker': N_SINGLE_UTTERANCES_PER_SPEAKER,
                'sample_rate': SAMPLE_RATE,
                'device': DEVICE,
                'random_seed': RANDOM_SEED,
                'librispeech_subsets': None,
                'partition_duration': None,
            }

            clean_other_dvectors_dict = _extract_with_cache(
                cache_dir,
                'clean_other_identity_librispeech',
                other_identity_cache_config,
                lambda: extract_single_utterance_dvectors(
                    str(librispeech_path),
                    non_target_identity_speakers,
                    N_SINGLE_UTTERANCES_PER_SPEAKER,
                    SAMPLE_RATE,
                    DEVICE
                )
            )
        else:
            print(f"\n⚠️  No non-target other speakers available after excluding main speakers")
            print(f"   Skipping other speakers identity mapping")
    
    # Extract silence d-vectors (for identity mapping)
    silence_dvectors_dict = {}
    if INCLUDE_SILENCE_IDENTITY:
        silence_n_samples = max(N_SILENCE_IDENTITY_PAIRS, 100) if N_SILENCE_IDENTITY_PAIRS > 0 else 500
        silence_cache_config = {
            'source': 'silence_identity',
            'n_samples': silence_n_samples,
            'duration_per_sample': 1.0,
            'sample_rate': SAMPLE_RATE,
            'device': DEVICE,
        }
        silence_dvectors_dict = _extract_with_cache(
            cache_dir,
            'silence_dvectors',
            silence_cache_config,
            lambda: extract_silence_dvectors(
                n_samples=silence_n_samples,
                duration_per_sample=1.0,
                sample_rate=SAMPLE_RATE,
                device=DEVICE
            )
        )
    
    # Extract noise d-vectors (for noise->silence mapping)
    noise_dvectors_dict = {}
    if INCLUDE_NOISE_IDENTITY:
        print(f"\n🎵 Extracting noise d-vectors from partitioned train sets...")
        noise_cache_config = {
            'source': 'noise_identity',
            'musan_root': _resolve_path_str(MUSAN_ROOT, script_dir),
            'rirs_root': _resolve_path_str(RIRS_ROOT, script_dir),
            'n_noise_samples_per_type': N_NOISE_SAMPLES_PER_TYPE,
            'n_rir_samples': N_RIR_SAMPLES,
            'noise_sample_duration': NOISE_SAMPLE_DURATION,
            'sample_rate': SAMPLE_RATE,
            'device': DEVICE,
            'random_seed': RANDOM_SEED,
        }
        noise_dvectors_dict = _extract_with_cache(
            cache_dir,
            'noise_dvectors',
            noise_cache_config,
            lambda: extract_noise_dvectors(
                musan_root=MUSAN_ROOT,
                rirs_root=RIRS_ROOT,
                n_noise_samples_per_type=N_NOISE_SAMPLES_PER_TYPE,
                n_rir_samples=N_RIR_SAMPLES,
                noise_sample_duration=NOISE_SAMPLE_DURATION,
                sample_rate=SAMPLE_RATE,
                device=DEVICE
            )
        )
        print(f"✓ Extracted {len(noise_dvectors_dict)} noise d-vectors")
    
    # Extract additional identity speakers (separate from main and overlapping speakers)
    additional_identity_dvectors_dict = {}
    # STRICT SEPARATION VALIDATION
    print(f"\n🔒 Validating speaker category separation...")
    
    # Identify all speakers that appear in overlaps
    overlap_speakers = set()
    for data in overlap_dvectors.values():
        speaker_field = data.get('speaker', '')
        if speaker_field:
            overlap_speakers.add(speaker_field)
        other_spk = data.get('other_speaker', '')
        if other_spk:
            overlap_speakers.add(other_spk)
        main_spk = data.get('main_speaker', '')
        if main_spk:
            overlap_speakers.add(main_spk)
    
    main_speakers_set = set(main_speakers)
    other_speakers_set = set(other_speakers) if other_speakers else set()
    
    # Validation 1: Main speakers vs Overlapping speakers
    main_overlap_intersection = main_speakers_set & overlap_speakers
    if main_overlap_intersection:
        print(f"  ⚠️  Warning: {len(main_overlap_intersection)} main speaker(s) also appear in overlaps (expected if they are targets)")
        # This is actually expected - main speakers should appear in overlaps
    
    # Validation 2: Other speakers vs Main speakers
    other_main_intersection = other_speakers_set & main_speakers_set
    if other_main_intersection:
        print(f"  ❌ ERROR: {len(other_main_intersection)} speaker(s) appear in BOTH main and other categories: {', '.join(sorted(other_main_intersection))}")
        raise ValueError("Main speakers and other speakers must not intersect!")
    
    print(f"  ✓ Main speakers ({len(main_speakers_set)}): {', '.join(sorted(main_speakers_set))}")
    print(f"  ✓ Overlapping speakers ({len(overlap_speakers)}): {len(overlap_speakers)} unique speakers in overlap samples")
    if other_speakers_set:
        print(f"  ✓ Other speakers ({len(other_speakers_set)}): Configured for identity mapping")
    
    if INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS:
        print(f"\n🎯 Extracting ADDITIONAL identity speakers (separate from main and overlap speakers)...")
        print(f"  Source subsets: {', '.join(ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS) if ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS else 'default'}")
        print(f"  Partition duration: {ADDITIONAL_IDENTITY_PARTITION_DURATION}s" if ADDITIONAL_IDENTITY_PARTITION_DURATION and ADDITIONAL_IDENTITY_PARTITION_DURATION > 0 else "  Partition duration: disabled")
        
        # Create exclusion list: main speakers + overlap speakers + other speakers
        exclude_speakers = main_speakers_set | overlap_speakers | other_speakers_set
        
        print(f"  Excluding {len(exclude_speakers)} speakers (main + overlap + other configured)")
        print(f"    Main: {len(main_speakers_set)} speakers")
        if len(overlap_speakers) > 0:
            print(f"    Overlap: {len(overlap_speakers)} speakers ({', '.join(sorted(list(overlap_speakers))[:5])}{'...' if len(overlap_speakers) > 5 else ''})")
        if other_speakers_set:
            print(f"    Other configured: {len(other_speakers_set)} speakers")
        
        # Discover all available speakers
        all_available = discover_all_available_speakers(
            librispeech_path,
            exclude_speakers,
            librispeech_subsets=ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS
        )
        
        if len(all_available) < N_ADDITIONAL_IDENTITY_SPEAKERS:
            print(f"  ⚠️  Warning: Only {len(all_available)} speakers available, but {N_ADDITIONAL_IDENTITY_SPEAKERS} requested")
            n_to_select = len(all_available)
        else:
            n_to_select = N_ADDITIONAL_IDENTITY_SPEAKERS
        
        # Randomly select N speakers
        np.random.seed(RANDOM_SEED)
        selected_additional = np.random.choice(all_available, size=n_to_select, replace=False).tolist()
        
        print(f"  Selected {len(selected_additional)} additional identity speakers")
        print(f"    First 10: {', '.join(selected_additional[:10])}{'...' if len(selected_additional) > 10 else ''}")
        
        # FINAL VALIDATION: Ensure no intersection
        selected_additional_set = set(selected_additional)
        additional_main_intersection = selected_additional_set & main_speakers_set
        additional_overlap_intersection = selected_additional_set & overlap_speakers
        additional_other_intersection = selected_additional_set & other_speakers_set
        
        if additional_main_intersection or additional_overlap_intersection or additional_other_intersection:
            print(f"  ❌ ERROR: Additional identity speakers intersect with other categories!")
            if additional_main_intersection:
                print(f"     Intersection with main: {', '.join(sorted(additional_main_intersection))}")
            if additional_overlap_intersection:
                print(f"     Intersection with overlap: {', '.join(sorted(additional_overlap_intersection))}")
            if additional_other_intersection:
                print(f"     Intersection with other: {', '.join(sorted(additional_other_intersection))}")
            raise ValueError("Additional identity speakers must be completely separate from main, overlap, and other speakers!")
        
        print(f"  ✓ Validated: No intersection with main ({len(main_speakers_set)}), overlap ({len(overlap_speakers)}), or other ({len(other_speakers_set)}) speakers")
        
        # Extract d-vectors for these speakers
        additional_identity_cache_config = {
            'source': 'librispeech_additional_identity',
            'librispeech_path': str(librispeech_path.resolve()),
            'speaker_ids': selected_additional,
            'n_utterances_per_speaker': N_ADDITIONAL_IDENTITY_UTTERANCES_PER_SPEAKER,
            'sample_rate': SAMPLE_RATE,
            'device': DEVICE,
            'random_seed': RANDOM_SEED,
            'librispeech_subsets': list(ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS) if ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS is not None else None,
            'partition_duration': ADDITIONAL_IDENTITY_PARTITION_DURATION,
        }

        additional_identity_dvectors_dict = _extract_with_cache(
            cache_dir,
            'clean_additional_identity_librispeech',
            additional_identity_cache_config,
            lambda: extract_single_utterance_dvectors(
                str(librispeech_path),
                selected_additional,
                N_ADDITIONAL_IDENTITY_UTTERANCES_PER_SPEAKER,
                SAMPLE_RATE,
                DEVICE,
                librispeech_subsets=ADDITIONAL_IDENTITY_LIBRISPEECH_SUBSETS,
                partition_duration=ADDITIONAL_IDENTITY_PARTITION_DURATION
            )
        )
    
    if len(overlap_dvectors) == 0:
        print("\n❌ Error: No overlap samples found. Check that overlap directories exist and contain audio/metadata")
        return
    
    # Extract clean main speakers d-vectors (flatten from dict)
    clean_main_dvectors = {}
    for speaker_id in main_speakers:
        if speaker_id in clean_main_dvectors_dict and len(clean_main_dvectors_dict[speaker_id]) > 0:
            clean_main_dvectors.update(clean_main_dvectors_dict[speaker_id])
    
    if len(clean_main_dvectors) == 0:
        print(f"\n❌ Error: No clean utterances found for main speakers {', '.join(main_speakers)}")
        return
    
    print(f"\n✓ Clean main speakers d-vectors: {len(clean_main_dvectors)} (from {len([s for s in main_speakers if s in clean_main_dvectors_dict])} speakers)")
    
    # Compute global average and create pairs if enabled
    other_main_speakers_averages = {}
    other_main_speakers_pairs = []
    if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        # Use the FIRST main speaker as the primary target (e.g., '84')
        primary_main_speaker = main_speakers[0]
        print(f"\n📦 Using GLOBAL AVERAGING for all non-main-speaker utterances")
        print(f"  Primary target speaker: {primary_main_speaker}")
        print(f"  All other utterances will map to a SINGLE global average")
        
        # Compute single global average from ALL non-main-speaker utterances
        global_avg = compute_global_non_main_average(
            clean_main_dvectors_dict,
            overlap_dvectors,
            primary_main_speaker,
            other_speakers_dvectors_dict=clean_other_dvectors_dict if len(clean_other_dvectors_dict) > 0 else None,
            additional_speakers_dvectors_dict=additional_identity_dvectors_dict if len(additional_identity_dvectors_dict) > 0 else None
        )
        
        if global_avg is not None:
            # Store as 'global_avg' key
            other_main_speakers_averages['global_avg'] = global_avg
            
            # Create pairs: ALL non-main-speaker utterances -> global average
            other_main_speakers_pairs = create_global_average_pairs(
                clean_main_dvectors_dict,
                overlap_dvectors,
                primary_main_speaker,
                other_speakers_dvectors_dict=clean_other_dvectors_dict if len(clean_other_dvectors_dict) > 0 else None,
                additional_speakers_dvectors_dict=additional_identity_dvectors_dict if len(additional_identity_dvectors_dict) > 0 else None,
                n_pairs_per_speaker=N_OTHER_MAIN_SPEAKERS_PAIRS_PER_SPEAKER
            )
        else:
            print(f"\n⚠️  Warning: Could not compute global average (no non-main-speaker utterances found)")
    
    # Extract non-target overlap samples if enabled
    non_target_dvectors = {}
    non_target_pairs = []
    other_speakers_dvectors_for_non_target_mean = None
    if INCLUDE_NON_TARGET_OVERLAPS:
        print(f"\n📦 Extracting non-target overlap samples...")
        non_target_cache_config = {
            'source': 'non_target_overlap',
            'overlap_dir': _resolve_path_str(NON_TARGET_OVERLAP_SAMPLES_DIR, script_dir),
            'n_samples': N_NON_TARGET_OVERLAP_SAMPLES,
            'sample_rate': SAMPLE_RATE,
            'device': DEVICE,
            'random_seed': RANDOM_SEED,
        }

        non_target_dvectors = _extract_with_cache(
            cache_dir,
            'non_target_overlap_dvectors',
            non_target_cache_config,
            lambda: extract_non_target_overlap_dvectors(
                NON_TARGET_OVERLAP_SAMPLES_DIR,
                n_samples=N_NON_TARGET_OVERLAP_SAMPLES,
                sample_rate=SAMPLE_RATE,
                device=DEVICE
            )
        )
        
        if len(non_target_dvectors) > 0:
            print(f"✓ Non-target overlap d-vectors: {len(non_target_dvectors)}")
            
            # Get d-vector dimension from overlap samples
            sample_dvec = list(overlap_dvectors.values())[0]['dvector']
            dvector_dim = sample_dvec.shape[0]
            
            # Create non-target pairs with appropriate target representation
            # For 'mean' strategy, we need to extract other speakers' d-vectors
            other_speakers_dvectors = None
            if NON_TARGET_STRATEGY == 'mean':
                # Extract d-vectors for other speakers to compute their mean
                non_target_mean_cache_config = {
                    'source': 'librispeech_non_target_mean_speakers',
                    'librispeech_path': str(librispeech_path.resolve()),
                    'speaker_ids': list(OTHER_SPEAKERS) if OTHER_SPEAKERS is not None else None,
                    'n_utterances_per_speaker': N_SINGLE_UTTERANCES_PER_SPEAKER,
                    'sample_rate': SAMPLE_RATE,
                    'device': DEVICE,
                    'random_seed': RANDOM_SEED,
                    'librispeech_subsets': None,
                    'partition_duration': None,
                }
                other_speakers_dvectors_dict = _extract_with_cache(
                    cache_dir,
                    'non_target_mean_other_speakers_librispeech',
                    non_target_mean_cache_config,
                    lambda: extract_single_utterance_dvectors(
                        str(librispeech_path),
                        OTHER_SPEAKERS,
                        N_SINGLE_UTTERANCES_PER_SPEAKER,
                        SAMPLE_RATE,
                        DEVICE
                    )
                )
                # Flatten to list of d-vectors
                other_speakers_dvectors = []
                for speaker, dvecs in other_speakers_dvectors_dict.items():
                    other_speakers_dvectors.extend([d['dvector'] for d in dvecs])
                other_speakers_dvectors_for_non_target_mean = other_speakers_dvectors
                print(f"✓ Extracted {len(other_speakers_dvectors)} other speaker d-vectors for mean computation")
            
            # Create pairs: (overlap_key, None) - target will be computed later
            non_target_pairs = [(key, None) for key in non_target_dvectors.keys()]
            print(f"✓ Created {len(non_target_pairs)} non-target training pairs")
        else:
            print(f"⚠️  Warning: No non-target overlap samples found, continuing without them")
    
    # Create identity pairs (clean -> clean) if enabled
    identity_pairs = []
    if INCLUDE_IDENTITY_PAIRS:
        if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
            # Only create identity pairs for the PRIMARY main speaker (e.g., '84')
            # Other main speakers will be handled by global averaging
            primary_main_speaker = main_speakers[0]
            print(f"\n🔗 Creating identity training pairs for PRIMARY main speaker only (clean -> same clean)...")
            print(f"  Primary speaker: {primary_main_speaker}")
            print(f"  (Other main speakers handled by global averaging)")
            
            # Create dict with only primary speaker
            primary_speaker_dict = {primary_main_speaker: clean_main_dvectors_dict[primary_main_speaker]} if primary_main_speaker in clean_main_dvectors_dict else {}
            identity_pairs = create_identity_pairs(primary_speaker_dict, N_IDENTITY_PAIRS)
            print(f"✓ Created {len(identity_pairs)} identity pairs for primary speaker {primary_main_speaker}")
        else:
            # Original behavior: create identity pairs for ALL main speakers
            print(f"\n🔗 Creating identity training pairs for MAIN speakers (clean -> same clean)...")
            identity_pairs = create_identity_pairs(clean_main_dvectors_dict, N_IDENTITY_PAIRS)
            print(f"✓ Created {len(identity_pairs)} identity pairs for main speakers {', '.join(main_speakers)}")
            
            # Show per-speaker breakdown using speaker_id from tuple
            pairs_by_speaker = defaultdict(int)
            for key, _, speaker_id, _ in identity_pairs:
                pairs_by_speaker[speaker_id] += 1
            for speaker_id, count in sorted(pairs_by_speaker.items()):
                print(f"  Speaker {speaker_id}: {count} identity pairs")
    
    # Create identity pairs for OTHER speakers
    other_speakers_identity_pairs = []
    if INCLUDE_OTHER_SPEAKERS_IDENTITY and len(clean_other_dvectors_dict) > 0:
        if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
            print(f"\n⏭️  Skipping other speakers identity pairs (other speakers → global average instead)")
        else:
            print(f"\n🔗 Creating identity training pairs for OTHER speakers (clean -> same clean)...")
            other_speakers_identity_pairs = create_identity_pairs(clean_other_dvectors_dict, N_OTHER_SPEAKERS_IDENTITY_PAIRS)
            print(f"✓ Created {len(other_speakers_identity_pairs)} identity pairs for other speakers")
            
            # Show per-speaker breakdown using speaker_id from tuple
            pairs_by_speaker = defaultdict(int)
            for key, _, speaker_id, _ in other_speakers_identity_pairs:
                pairs_by_speaker[speaker_id] += 1
            for speaker_id, count in sorted(pairs_by_speaker.items()):
                print(f"  Speaker {speaker_id}: {count} identity pairs")
    
    # Create identity pairs for SILENCE frames
    silence_identity_pairs = []
    if INCLUDE_SILENCE_IDENTITY and len(silence_dvectors_dict) > 0:
        if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
            print(f"\n⏭️  Skipping silence identity pairs (handled by global averaging)")
        else:
            print(f"\n🔗 Creating identity training pairs for SILENCE frames (silence -> same silence)...")
            # Wrap silence dict in nested structure for compatibility with create_identity_pairs
            silence_nested = {'silence': silence_dvectors_dict}
            silence_identity_pairs = create_identity_pairs(silence_nested, N_SILENCE_IDENTITY_PAIRS)
            print(f"✓ Created {len(silence_identity_pairs)} silence identity pairs")
    
    # Create noise->silence mapping pairs (teach model to map noise to silence)
    noise_identity_pairs = []
    if INCLUDE_NOISE_IDENTITY and len(noise_dvectors_dict) > 0:
        if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
            print(f"\n⏭️  Skipping noise identity pairs (handled by global averaging)")
        else:
            print(f"\n🔗 Creating noise->silence mapping pairs (noise -> silence)...")
            # Create pairs mapping noise d-vectors to silence d-vectors
            # For each noise sample, pair it with a random silence d-vector as target
            if len(silence_dvectors_dict) == 0:
                print(f"⚠️  Warning: No silence d-vectors available for noise mapping. Skipping noise pairs.")
            else:
                silence_keys_list = list(silence_dvectors_dict.keys())
                for noise_key, noise_data in noise_dvectors_dict.items():
                    # Select random silence key as target
                    silence_target_key = np.random.choice(silence_keys_list)
                    # Create pair: (input_key, target_key, speaker_id, 'identity')
                    # Format matches other identity pairs: both keys are strings
                    noise_type = noise_data['type']
                    noise_identity_pairs.append((
                        noise_key,
                        silence_target_key,
                        f'noise_{noise_type}',
                        'identity'
                    ))
                print(f"✓ Created {len(noise_identity_pairs)} noise->silence mapping pairs")
                # Show breakdown by noise type
                noise_type_counts = defaultdict(int)
                for _, _, speaker_id, _ in noise_identity_pairs:
                    noise_type_counts[speaker_id] += 1
                for noise_type, count in sorted(noise_type_counts.items()):
                    print(f"    {noise_type}: {count} pairs")
    
    # Create identity pairs for ADDITIONAL speakers
    additional_identity_pairs = []
    if INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS and len(additional_identity_dvectors_dict) > 0:
        if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
            print(f"\n⏭️  Skipping additional speakers identity pairs (handled by global averaging)")
        else:
            print(f"\n🔗 Creating identity training pairs for ADDITIONAL speakers (clean -> same clean)...")
            additional_identity_pairs = create_identity_pairs(additional_identity_dvectors_dict, N_ADDITIONAL_IDENTITY_PAIRS_PER_SPEAKER)
            print(f"✓ Created {len(additional_identity_pairs)} identity pairs for additional speakers")
            
            # Show per-speaker breakdown using speaker_id from tuple
            pairs_by_speaker = defaultdict(int)
            for key, _, speaker_id, _ in additional_identity_pairs:
                pairs_by_speaker[speaker_id] += 1
            print(f"  Covering {len(pairs_by_speaker)} speakers")
            # Show a few examples
            example_speakers = sorted(pairs_by_speaker.items())[:5]
            for speaker_id, count in example_speakers:
                print(f"    Speaker {speaker_id}: {count} identity pairs")
            if len(pairs_by_speaker) > 5:
                print(f"    ... and {len(pairs_by_speaker) - 5} more speakers")
    
    # Merge all clean d-vectors (main + other speakers + silence + noise + additional) for dataset
    all_clean_dvectors = clean_main_dvectors.copy()
    for speaker_id, speaker_dvectors in clean_other_dvectors_dict.items():
        all_clean_dvectors.update(speaker_dvectors)
    if len(silence_dvectors_dict) > 0:
        all_clean_dvectors.update(silence_dvectors_dict)
    if len(noise_dvectors_dict) > 0:
        all_clean_dvectors.update(noise_dvectors_dict)
    for speaker_id, speaker_dvectors in additional_identity_dvectors_dict.items():
        all_clean_dvectors.update(speaker_dvectors)
    
    # Create training pairs for full overlap samples
    pairs = create_training_pairs(overlap_dvectors, clean_main_dvectors, main_speakers, N_OVERLAP_PAIRS)
    
    # Combine all identity pairs (main + other speakers + silence + noise + additional)
    all_identity_pairs = identity_pairs + other_speakers_identity_pairs + silence_identity_pairs + noise_identity_pairs + additional_identity_pairs
    
    if len(pairs) == 0 and len(all_identity_pairs) == 0 and len(other_main_speakers_pairs) == 0:
        print("\n❌ Error: No training pairs created")
        return
    
    print(f"\n✓ Total pairs created:")
    if len(identity_pairs) > 0:
        print(f"  Identity pairs (main speaker): {len(identity_pairs)}")
    if len(other_speakers_identity_pairs) > 0:
        print(f"  Identity pairs (other speakers): {len(other_speakers_identity_pairs)}")
    if len(silence_identity_pairs) > 0:
        print(f"  Identity pairs (silence): {len(silence_identity_pairs)}")
    if len(noise_identity_pairs) > 0:
        print(f"  Identity pairs (noise->silence): {len(noise_identity_pairs)}")
    if len(additional_identity_pairs) > 0:
        print(f"  Identity pairs (additional speakers): {len(additional_identity_pairs)}")
    if len(all_identity_pairs) > 0:
        print(f"  Total identity pairs: {len(all_identity_pairs)}")
    print(f"  Overlap pairs (primary speaker only): {len(pairs)}")
    if len(other_main_speakers_pairs) > 0:
        print(f"  Other main speakers pairs (singles + overlaps -> avg): {len(other_main_speakers_pairs)}")
    
    # Split into train/val/test (both identity, overlap, and non-target)
    # Split identity pairs (main + other speakers)
    train_identity_pairs = []
    val_identity_pairs = []
    test_identity_pairs = []
    if len(all_identity_pairs) > 0:
        if len(all_identity_pairs) < 2:
            # Not enough samples to split, use all for training
            train_identity_pairs = all_identity_pairs
        else:
            val_test_size = VALIDATION_SPLIT + TEST_SPLIT
            train_identity, temp_identity = train_test_split(all_identity_pairs, test_size=val_test_size, random_state=RANDOM_SEED)
            if len(temp_identity) < 2:
                # Not enough samples to split further, use for validation only
                train_identity_pairs = train_identity
                val_identity_pairs = temp_identity
            else:
                val_identity, test_identity = train_test_split(temp_identity, test_size=TEST_SPLIT/val_test_size, random_state=RANDOM_SEED)
                train_identity_pairs = train_identity
                val_identity_pairs = val_identity
                test_identity_pairs = test_identity
    
    # Split overlap pairs
    train_pairs = []
    val_pairs = []
    test_pairs = []
    if len(pairs) > 0:
        if len(pairs) < 2:
            # Not enough samples to split, use all for training
            train_pairs = pairs
        else:
            val_test_size = VALIDATION_SPLIT + TEST_SPLIT
            train_overlap, temp_overlap = train_test_split(pairs, test_size=val_test_size, random_state=RANDOM_SEED)
            if len(temp_overlap) < 2:
                # Not enough samples to split further, use for validation only
                train_pairs = train_overlap
                val_pairs = temp_overlap
            else:
                val_overlap, test_overlap = train_test_split(temp_overlap, test_size=TEST_SPLIT/val_test_size, random_state=RANDOM_SEED)
                train_pairs = train_overlap
                val_pairs = val_overlap
                test_pairs = test_overlap
    
    # Split non-target pairs if available
    train_non_target_pairs = []
    val_non_target_pairs = []
    test_non_target_pairs = []
    if len(non_target_pairs) > 0:
        if len(non_target_pairs) < 2:
            # Not enough samples to split, use all for training
            train_non_target_pairs = non_target_pairs
        else:
            val_test_size = VALIDATION_SPLIT + TEST_SPLIT
            train_nt, temp_nt = train_test_split(non_target_pairs, test_size=val_test_size, random_state=RANDOM_SEED)
            if len(temp_nt) < 2:
                # Not enough samples to split further, use for validation only
                train_non_target_pairs = train_nt
                val_non_target_pairs = temp_nt
            else:
                val_nt, test_nt = train_test_split(temp_nt, test_size=TEST_SPLIT/val_test_size, random_state=RANDOM_SEED)
                train_non_target_pairs = train_nt
                val_non_target_pairs = val_nt
                test_non_target_pairs = test_nt
    
    # Split other main speakers pairs if available
    train_other_main_pairs = []
    val_other_main_pairs = []
    test_other_main_pairs = []
    if len(other_main_speakers_pairs) > 0:
        if len(other_main_speakers_pairs) < 2:
            # Not enough samples to split, use all for training
            train_other_main_pairs = other_main_speakers_pairs
        else:
            val_test_size = VALIDATION_SPLIT + TEST_SPLIT
            train_om, temp_om = train_test_split(other_main_speakers_pairs, test_size=val_test_size, random_state=RANDOM_SEED)
            if len(temp_om) < 2:
                # Not enough samples to split further, use for validation only
                train_other_main_pairs = train_om
                val_other_main_pairs = temp_om
            else:
                val_om, test_om = train_test_split(temp_om, test_size=TEST_SPLIT/val_test_size, random_state=RANDOM_SEED)
                train_other_main_pairs = train_om
                val_other_main_pairs = val_om
                test_other_main_pairs = test_om
    
    print(f"\n📊 Dataset split (train:{1-VALIDATION_SPLIT-TEST_SPLIT:.0%} / val:{VALIDATION_SPLIT:.0%} / test:{TEST_SPLIT:.0%}):")
    
    # Train split
    train_total = len(train_identity_pairs) + len(train_pairs) + len(train_non_target_pairs) + len(train_other_main_pairs)
    print(f"  Train: {train_total} total", end="")
    components = []
    if len(train_identity_pairs) > 0:
        components.append(f"{len(train_identity_pairs)} identity")
    if len(train_pairs) > 0:
        components.append(f"{len(train_pairs)} overlap")
    if len(train_other_main_pairs) > 0:
        components.append(f"{len(train_other_main_pairs)} other-main-avg")
    if len(train_non_target_pairs) > 0:
        components.append(f"{len(train_non_target_pairs)} non-target")
    if components:
        print(f" ({', '.join(components)})")
    else:
        print()
    
    # Val split
    val_total = len(val_identity_pairs) + len(val_pairs) + len(val_non_target_pairs) + len(val_other_main_pairs)
    print(f"  Val: {val_total} total", end="")
    components = []
    if len(val_identity_pairs) > 0:
        components.append(f"{len(val_identity_pairs)} identity")
    if len(val_pairs) > 0:
        components.append(f"{len(val_pairs)} overlap")
    if len(val_other_main_pairs) > 0:
        components.append(f"{len(val_other_main_pairs)} other-main-avg")
    if len(val_non_target_pairs) > 0:
        components.append(f"{len(val_non_target_pairs)} non-target")
    if components:
        print(f" ({', '.join(components)})")
    else:
        print()
    
    # Test split
    test_total = len(test_identity_pairs) + len(test_pairs) + len(test_non_target_pairs) + len(test_other_main_pairs)
    print(f"  Test: {test_total} total", end="")
    components = []
    if len(test_identity_pairs) > 0:
        components.append(f"{len(test_identity_pairs)} identity")
    if len(test_pairs) > 0:
        components.append(f"{len(test_pairs)} overlap")
    if len(test_other_main_pairs) > 0:
        components.append(f"{len(test_other_main_pairs)} other-main-avg")
    if len(test_non_target_pairs) > 0:
        components.append(f"{len(test_non_target_pairs)} non-target")
    if components:
        print(f" ({', '.join(components)})")
    else:
        print()
    
    # Get d-vector dimension from first sample
    sample_dvec = list(overlap_dvectors.values())[0]['dvector']
    dvector_dim = sample_dvec.shape[0]
    
    # Create target representation for non-target samples (needed by dataset)
    target_representation = None
    if len(non_target_pairs) > 0:
        # Reuse previously extracted mean-speaker vectors when available.
        other_speakers_dvectors = other_speakers_dvectors_for_non_target_mean
        if NON_TARGET_STRATEGY == 'mean':
            if other_speakers_dvectors is None:
                non_target_mean_cache_config = {
                    'source': 'librispeech_non_target_mean_speakers',
                    'librispeech_path': str(librispeech_path.resolve()),
                    'speaker_ids': list(OTHER_SPEAKERS) if OTHER_SPEAKERS is not None else None,
                    'n_utterances_per_speaker': N_SINGLE_UTTERANCES_PER_SPEAKER,
                    'sample_rate': SAMPLE_RATE,
                    'device': DEVICE,
                    'random_seed': RANDOM_SEED,
                    'librispeech_subsets': None,
                    'partition_duration': None,
                }
                other_speakers_dvectors_dict = _extract_with_cache(
                    cache_dir,
                    'non_target_mean_other_speakers_librispeech',
                    non_target_mean_cache_config,
                    lambda: extract_single_utterance_dvectors(
                        str(librispeech_path),
                        OTHER_SPEAKERS,
                        N_SINGLE_UTTERANCES_PER_SPEAKER,
                        SAMPLE_RATE,
                        DEVICE
                    )
                )
                other_speakers_dvectors = []
                for speaker, dvecs in other_speakers_dvectors_dict.items():
                    other_speakers_dvectors.extend([d['dvector'] for d in dvecs])
                other_speakers_dvectors_for_non_target_mean = other_speakers_dvectors
        
        target_representation = create_non_target_representation(
            dvector_dim,
            NON_TARGET_STRATEGY,
            other_speakers_dvectors
        )
    
    # Create datasets with identity, overlap, non-target, and other main speakers support
    # Use all_clean_dvectors which includes both main and other speakers
    train_dataset = DvectorPairDataset(
        overlap_dvectors, 
        all_clean_dvectors, 
        train_pairs,
        identity_pairs=train_identity_pairs if len(train_identity_pairs) > 0 else None,
        non_target_dvectors=non_target_dvectors if len(train_non_target_pairs) > 0 else None,
        non_target_pairs=train_non_target_pairs if len(train_non_target_pairs) > 0 else None,
        non_target_target=target_representation,
        other_main_speakers_pairs=train_other_main_pairs if len(train_other_main_pairs) > 0 else None,
        other_main_speakers_averages=other_main_speakers_averages if len(other_main_speakers_averages) > 0 else None,
        main_speaker_ids=main_speakers,
    )
    val_dataset = DvectorPairDataset(
        overlap_dvectors, 
        all_clean_dvectors, 
        val_pairs,
        identity_pairs=val_identity_pairs if len(val_identity_pairs) > 0 else None,
        non_target_dvectors=non_target_dvectors if len(val_non_target_pairs) > 0 else None,
        non_target_pairs=val_non_target_pairs if len(val_non_target_pairs) > 0 else None,
        non_target_target=target_representation,
        other_main_speakers_pairs=val_other_main_pairs if len(val_other_main_pairs) > 0 else None,
        other_main_speakers_averages=other_main_speakers_averages if len(other_main_speakers_averages) > 0 else None,
        main_speaker_ids=main_speakers,
    )
    test_dataset = DvectorPairDataset(
        overlap_dvectors, 
        all_clean_dvectors, 
        test_pairs,
        identity_pairs=test_identity_pairs if len(test_identity_pairs) > 0 else None,
        non_target_dvectors=non_target_dvectors if len(test_non_target_pairs) > 0 else None,
        non_target_pairs=test_non_target_pairs if len(test_non_target_pairs) > 0 else None,
        non_target_target=target_representation,
        other_main_speakers_pairs=test_other_main_pairs if len(test_other_main_pairs) > 0 else None,
        other_main_speakers_averages=other_main_speakers_averages if len(other_main_speakers_averages) > 0 else None,
        main_speaker_ids=main_speakers,
    )
    
    # Create dataloaders
    # drop_last=True for training to avoid BatchNorm error with single-sample batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get d-vector dimension from first sample
    sample_dvec = list(overlap_dvectors.values())[0]['dvector']
    dvector_dim = sample_dvec.shape[0]
    print(f"\n  D-vector dimension: {dvector_dim}")
    
    pretrained_config = None
    model = None
    if PRETRAINED_MODEL_PATH is not None:
        pretrained_path = Path(PRETRAINED_MODEL_PATH)
        if pretrained_path.exists():
            try:
                model, pretrained_config = load_autoencoder(pretrained_path, DEVICE)
                print(f"✓ Loaded pretrained model from: {pretrained_path}")
            except Exception as e:
                print(f"⚠️  Warning: Pretrained load failed ({e}); falling back to scratch model")
                model = None
        else:
            print(f"⚠️  Warning: Pretrained model path not found: {pretrained_path}")

    if model is None:
        model = DvectorAutoencoder(
            input_dim=dvector_dim,
            hidden_dims=HIDDEN_DIMS,
            dropout_rate=DROPOUT_RATE,
            norm_type=AE_NORM_TYPE,
            use_residual=AE_USE_RESIDUAL,
            residual_scale_init=AE_RESIDUAL_SCALE_INIT,
        )
    
    print(f"\n🏗️  Model architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    if pretrained_config is not None and int(pretrained_config.get('input_dim', dvector_dim)) != int(dvector_dim):
        print(
            "⚠️  Warning: pretrained input_dim does not match current d-vector dim; "
            "check dataset or model config."
        )
    
    # Train model
    history, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        LEARNING_RATE,
        DEVICE,
        save_dir
    )
    
    # Load best model for evaluation
    best_model_path = save_dir / 'best_model.pth'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on test set
    eval_results = evaluate_model(model, test_loader, DEVICE)
    
    # Plot results
    plot_training_history(history, save_dir / 'training_history.png')
    plot_reconstruction_quality(eval_results, save_dir / 'reconstruction_quality.png')
    
    # Collect speaker information for config
    overlap_speakers_used = set()
    for data in overlap_dvectors.values():
        main_spk = data.get('main_speaker', '')
        other_spk = data.get('other_speaker', '')
        if main_spk:
            overlap_speakers_used.add(main_spk)
        if other_spk:
            overlap_speakers_used.add(other_spk)
    
    # Get speakers used in identity pairs
    main_identity_speakers = set()
    if len(identity_pairs) > 0:
        for _, _, speaker_id, _ in identity_pairs:
            main_identity_speakers.add(speaker_id)
    
    other_identity_speakers = set()
    if len(other_speakers_identity_pairs) > 0:
        for _, _, speaker_id, _ in other_speakers_identity_pairs:
            other_identity_speakers.add(speaker_id)
    
    additional_identity_speakers = set()
    if len(additional_identity_pairs) > 0:
        for _, _, speaker_id, _ in additional_identity_pairs:
            additional_identity_speakers.add(speaker_id)
    
    has_silence_identity = len(silence_identity_pairs) > 0
    
    # Get other main speakers if used
    other_main_speaker_ids_used = []
    if len(other_main_speakers_pairs) > 0:
        other_main_speaker_ids_used = list(set([speaker_id for _, speaker_id, _ in other_main_speakers_pairs]))
    
    # Save final model and config
    config = {
        # Model architecture
        'input_dim': dvector_dim,
        'hidden_dims': HIDDEN_DIMS,
        'dropout_rate': DROPOUT_RATE,
        'norm_type': AE_NORM_TYPE,
        'use_residual': AE_USE_RESIDUAL,
        'residual_scale_init': AE_RESIDUAL_SCALE_INIT,
        
        # Training configuration
        'training_scheme': TRAINING_SCHEME,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'loss_mse_weight': LOSS_MSE_WEIGHT,
        'loss_cosine_weight': LOSS_COSINE_WEIGHT,
        'loss_negative_weight': NEGATIVE_CONTRASTIVE_WEIGHT,
        'loss_negative_margin': NEGATIVE_CONTRASTIVE_MARGIN,
        'validation_split': VALIDATION_SPLIT,
        'test_split': TEST_SPLIT,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        
        # Data configuration
        'overlap_samples_dirs': [str(d) for d in overlap_dirs_resolved],
        'single_speaker_datasets': SINGLE_SPEAKER_DATASETS if SINGLE_SPEAKER_DATASETS else None,
        'librispeech_path': str(librispeech_path),
        'n_single_utterances_per_speaker': N_SINGLE_UTTERANCES_PER_SPEAKER,
        'sample_rate': SAMPLE_RATE,
        'dvector_cache_enabled': DVECTOR_CACHE_ENABLED,
        'dvector_cache_dir': str(cache_dir),
        'dvector_cache_version': DVECTOR_CACHE_VERSION,
        
        # Speaker configuration
        'main_speakers': main_speakers,
        'main_speakers_auto_detected': main_speakers_auto_detected,
        'other_speakers_configured': other_speakers if other_speakers is not None else 'auto-detected',
        'other_speakers_source': 'explicit' if OTHER_SPEAKERS is not None else 'auto-detected',
        'overlap_speakers_used': sorted(list(overlap_speakers_used)),
        'main_identity_speakers_used': sorted(list(main_identity_speakers)),
        'other_identity_speakers_used': sorted(list(other_identity_speakers)),
        'additional_identity_speakers_used': sorted(list(additional_identity_speakers)),
        'other_main_speakers_used': sorted(other_main_speaker_ids_used),
        
        # Dataset composition
        'include_identity_pairs': INCLUDE_IDENTITY_PAIRS,
        'include_other_speakers_identity': INCLUDE_OTHER_SPEAKERS_IDENTITY,
        'include_silence_identity': INCLUDE_SILENCE_IDENTITY,
        'include_noise_identity': INCLUDE_NOISE_IDENTITY,
        'include_additional_identity_speakers': INCLUDE_ADDITIONAL_IDENTITY_SPEAKERS,
        'include_other_main_speakers_avg': INCLUDE_OTHER_MAIN_SPEAKERS_AVG,
        'include_non_target_overlaps': INCLUDE_NON_TARGET_OVERLAPS,
        'n_identity_pairs': N_IDENTITY_PAIRS,
        'n_other_speakers_identity_pairs': N_OTHER_SPEAKERS_IDENTITY_PAIRS,
        'n_silence_identity_pairs': N_SILENCE_IDENTITY_PAIRS,
        'n_noise_samples_per_type': N_NOISE_SAMPLES_PER_TYPE if INCLUDE_NOISE_IDENTITY else 0,
        'n_rir_samples': N_RIR_SAMPLES if INCLUDE_NOISE_IDENTITY else 0,
        'noise_sample_duration': NOISE_SAMPLE_DURATION if INCLUDE_NOISE_IDENTITY else 0,
        'n_additional_identity_speakers': N_ADDITIONAL_IDENTITY_SPEAKERS,
        'n_additional_identity_utterances_per_speaker': N_ADDITIONAL_IDENTITY_UTTERANCES_PER_SPEAKER,
        'n_additional_identity_pairs_per_speaker': N_ADDITIONAL_IDENTITY_PAIRS_PER_SPEAKER,
        'n_overlap_pairs': N_OVERLAP_PAIRS,
        'n_other_main_speakers_pairs_per_speaker': N_OTHER_MAIN_SPEAKERS_PAIRS_PER_SPEAKER,
        'n_non_target_overlap_samples': N_NON_TARGET_OVERLAP_SAMPLES if INCLUDE_NON_TARGET_OVERLAPS else 0,
        'non_target_strategy': NON_TARGET_STRATEGY if INCLUDE_NON_TARGET_OVERLAPS else None,
        'has_silence_identity_pairs': has_silence_identity,
        'has_noise_identity_pairs': len(noise_identity_pairs) > 0,
        
        # Dataset statistics
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_identity_pairs': len(train_identity_pairs),
        'train_overlap_pairs': len(train_pairs),
        'train_other_main_pairs': len(train_other_main_pairs),
        'train_non_target_pairs': len(train_non_target_pairs),
        'val_identity_pairs': len(val_identity_pairs),
        'val_overlap_pairs': len(val_pairs),
        'val_other_main_pairs': len(val_other_main_pairs),
        'val_non_target_pairs': len(val_non_target_pairs),
        'test_identity_pairs': len(test_identity_pairs),
        'test_overlap_pairs': len(test_pairs),
        'test_other_main_pairs': len(test_other_main_pairs),
        'test_non_target_pairs': len(test_non_target_pairs),
        
        # Performance metrics
        'best_val_loss': best_val_loss,
        'test_mixed_loss': eval_results['mixed_loss'],
        'test_mse_loss': eval_results['mse_loss'],
        'test_cosine_loss': eval_results['cosine_loss'],
        'test_cosine_similarity_mean': eval_results['cosine_similarity'].mean(),
        'test_cosine_similarity_std': eval_results['cosine_similarity'].std(),
        
        # Misc
        'random_seed': RANDOM_SEED,
        'device': DEVICE,
        'pretrained_model_path': PRETRAINED_MODEL_PATH,
        'pretrained_model_type': pretrained_config.get('model_type') if pretrained_config else None,
    }
    
    save_model_and_config(model, config, save_dir)
    
    # Save evaluation results
    eval_save_path = save_dir / 'evaluation_results.pkl'
    with open(eval_save_path, 'wb') as f:
        pickle.dump(eval_results, f)
    print(f"✓ Saved evaluation results to: {eval_save_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Test Mixed Loss: {eval_results['mixed_loss']:.6f}")
    print(f"  Test MSE Loss: {eval_results['mse_loss']:.6f}")
    print(f"  Test Cosine Loss: {eval_results['cosine_loss']:.6f}")
    print(f"  Test Cosine Similarity: {eval_results['cosine_similarity'].mean():.4f} ± {eval_results['cosine_similarity'].std():.4f}")
    print(f"\n💡 Interpretation:")
    print(f"  Cosine similarity > 0.9 = Excellent main speaker extraction")
    print(f"  Cosine similarity > 0.8 = Good main speaker extraction")
    print(f"  Cosine similarity > 0.7 = Moderate main speaker extraction")
    print(f"\n📁 Model saved to: {save_dir}")
    print(f"\n🎓 Model was trained on:")
    if INCLUDE_IDENTITY_PAIRS:
        print(f"  - Identity mapping: Clean main speaker -> preserved as-is")
    if INCLUDE_OTHER_SPEAKERS_IDENTITY:
        print(f"  - Identity mapping: Clean other speakers -> preserved as-is")
    print(f"  - Overlap denoising: Overlap (X+other) -> Clean X (primary speaker only)")
    if INCLUDE_OTHER_MAIN_SPEAKERS_AVG:
        print(f"  - Other main speakers averaging: Y,Z,... singles + overlaps -> Average d-vector")
    if INCLUDE_NON_TARGET_OVERLAPS:
        print(f"  - Non-target rejection: Other speakers -> {NON_TARGET_STRATEGY}")
    print(f"\n🚀 Use the model to extract main speakers {', '.join(main_speakers)}:")
    print(f"  1. Load model: torch.load('{save_dir / 'final_model.pth'}')")
    print(f"  2. Load config: pickle.load(open('{save_dir / 'config.pkl'}', 'rb'))")
    print(f"  3. Extract main speaker: main_dvector = model(overlap_dvector_tensor)")
    print(f"\n💡 The model will:")
    print(f"  - Output clean main speaker d-vector from overlap input")
    if INCLUDE_IDENTITY_PAIRS:
        print(f"  - Preserve clean main speaker d-vectors unchanged (identity)")
    if INCLUDE_NON_TARGET_OVERLAPS:
        print(f"  - Reject non-target speakers ({NON_TARGET_STRATEGY} output)")


if __name__ == "__main__":
    main()
