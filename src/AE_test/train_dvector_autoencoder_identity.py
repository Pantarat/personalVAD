#!/usr/bin/env python3
"""
D-Vector Autoencoder for Identity Mapping (Reconstruction)
Train an autoencoder to reconstruct d-vectors (identity mapping)

This pipeline:
1. Loads various d-vectors (clean speakers, silence, overlaps)
2. Creates training pairs: input -> same input (identity/reconstruction)
3. Trains an autoencoder to reconstruct the input
4. Evaluates reconstruction quality
5. Saves trained model for future use

This is useful for:
- Learning a compressed representation of d-vectors
- Denoising d-vectors
- Testing autoencoder capacity
"""

import warnings
# Suppress FutureWarning from resemblyzer library (librosa positional args deprecation)
warnings.filterwarnings('ignore', category=FutureWarning, module='resemblyzer')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import glob
import librosa
from resemblyzer import VoiceEncoder
from collections import defaultdict, Counter
import pickle

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
LIBRISPEECH_PATH = '../../data/LibriSpeech'
# OVERLAP_SAMPLES_DIRS = []  # Overlap samples to include
OVERLAP_SAMPLES_DIRS = ['test_outputs/data/test_other_mix_1,6s_100000_balanced_4-1']  # Overlap/mix dataset dirs from generate_full_overlap_utterances.py
MODEL_SAVE_DIR = 'test_outputs/models/dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_1-4-26'  # Directory to save trained model and config

# LibriSpeech subsets to use for speaker selection
# Available: 'dev-clean', 'dev-other', 'test-clean', 'test-other',
#            'train-clean-100', 'train-clean-360', 'train-other-500'
# Use None to include all available subsets
# LIBRISPEECH_SUBSETS = None  # Use all subsets
# LIBRISPEECH_SUBSETS = ['dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360']  # Clean only
# LIBRISPEECH_SUBSETS = ['dev-other', 'test-other', 'train-other-500']  # Other/noisy only
LIBRISPEECH_SUBSETS = ['train-other-500']  # Small subset for quick testing

# Optional: balance sampled speakers by sex using LibriSpeech SPEAKERS.TXT
BALANCE_SPEAKERS_BY_SEX = True  # If True, tries to sample ~50% male and ~50% female
LIBRISPEECH_SPEAKERS_TXT = '../../data/LibriSpeech/SPEAKERS.TXT'

# Speakers to include
N_SPEAKERS = 1100  # Number of speakers to use for identity mapping
N_UTTERANCES_PER_SPEAKER = 50  # Number of utterances per speaker
DURATION_PER_SAMPLE = 1.6  # Duration per sample in seconds (min 0.3s, optimal 1.6s)

# Overlap samples
INCLUDE_OVERLAP_IDENTITY = True  # Include overlap -> same overlap (identity mapping)
N_OVERLAP_SAMPLES = -1  # Number of overlap samples to use (-1 = use all available)
OVERLAP_INCLUDE_MAIN_SPEAKER_SAMPLES = True  # Include samples where `has_main_speaker=True`
OVERLAP_INCLUDE_OTHER_MIXTURES = True  # Include non-main mixtures from generate_full_overlap_utterances.py

# Silence
INCLUDE_SILENCE_IDENTITY = False  # Include silence -> same silence (identity mapping)
N_SILENCE_SAMPLES = 20000  # Number of silence samples to use

# Noise (MUSAN + RIRS)
INCLUDE_NOISE_IDENTITY = True  # Include noise -> same noise (identity mapping)
MUSAN_ROOT = '../../data/musan'  # Path to MUSAN dataset
RIRS_ROOT = '../../data/RIRS_NOISES'  # Path to RIRS_NOISES dataset
N_NOISE_SAMPLES_PER_TYPE = 100  # Number of noise samples per type (music, noise, speech)
N_RIR_SAMPLES = 500  # Number of RIR samples
NOISE_SAMPLE_DURATION = 1.6  # Duration of noise samples in seconds (min 0.3s, optimal 1.6s)

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
NUM_EPOCHS = 500
VALIDATION_SPLIT = 0.02
TEST_SPLIT = 0
EARLY_STOPPING_PATIENCE = 50

# Model architecture
HIDDEN_DIMS = [128,64,128]  # Encoder-bottleneck-decoder
DROPOUT_RATE = 0.1

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Audio settings
SAMPLE_RATE = 16000

# Random seed
RANDOM_SEED = 42


class DvectorAutoencoder(nn.Module):
    """
    Autoencoder to reconstruct d-vectors (identity mapping)
    
    Architecture:
    - Encoder: d-vector (256) -> hidden layers -> bottleneck
    - Decoder: bottleneck -> hidden layers -> d-vector (256)
    """
    
    def __init__(self, input_dim=256, hidden_dims=[128, 64, 128], dropout_rate=0.2):
        super(DvectorAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        # First half of hidden_dims for encoder
        n_encoder_layers = len(hidden_dims) // 2
        for i in range(n_encoder_layers):
            encoder_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            encoder_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        # Bottleneck
        bottleneck_dim = hidden_dims[n_encoder_layers]
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        encoder_layers.append(nn.BatchNorm1d(bottleneck_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = bottleneck_dim
        
        # Second half of hidden_dims for decoder
        for i in range(n_encoder_layers + 1, len(hidden_dims)):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        # Output layer with ReLU to ensure non-negative d-vector values
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.ReLU())  # Ensure all output dimensions >= 0
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode to bottleneck representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from bottleneck representation"""
        return self.decoder(z)


class DvectorIdentityDataset(Dataset):
    """
    Dataset of (input_dvector, same_dvector) pairs for identity mapping
    """
    
    def __init__(self, dvectors_dict):
        """
        Args:
            dvectors_dict: Dict of d-vectors {key: {'dvector': array, ...}}
        """
        self.dvectors_dict = dvectors_dict
        self.keys = list(dvectors_dict.keys())
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        dvec = self.dvectors_dict[key]['dvector']
        
        return (
            torch.FloatTensor(dvec),
            torch.FloatTensor(dvec)  # Identity: output = input
        )


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
            # Generate silence
            n_samples_audio = int(duration_per_sample * sample_rate)
            silence_audio = np.zeros(n_samples_audio, dtype=np.float32)
            
            # Extract d-vector using resemblyzer
            with torch.no_grad():
                dvector = encoder.embed_utterance(silence_audio)
            
            # Store result
            key = f"silence_{i:05d}"
            results[key] = {
                'dvector': dvector,
                'type': 'silence',
                'duration': duration_per_sample
            }
            
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{n_samples} silence samples")
                
        except Exception as e:
            print(f"    ⚠️  Error extracting silence sample {i}: {e}")
            continue
    
    print(f"✓ Total silence d-vectors: {len(results)}")
    
    return results


def extract_noise_dvectors(musan_root, rirs_root, n_noise_samples_per_type=100, n_rir_samples=50,
                           noise_sample_duration=1.0, sample_rate=16000, device='cuda'):
    """
    Extract d-vectors from noise samples (MUSAN and RIRS TRAIN PARTITIONS ONLY)
    For identity mapping: noise -> same noise
    
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
    for noise_type, noise_dir in musan_types.items():
        if 'test' in str(noise_dir).lower():
            raise ValueError(f"ERROR: Attempting to use TEST partition for {noise_type}: {noise_dir}")
        if '_train' not in str(noise_dir):
            raise ValueError(f"ERROR: Directory name must contain '_train' suffix: {noise_dir}")
    
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
                # Validation: Ensure we're using train partition
                if 'test' in str(rir_list_file).lower():
                    raise ValueError(f"ERROR: Attempting to use TEST partition: {rir_list_file}")
                if '_train' not in str(rir_list_file):
                    raise ValueError(f"ERROR: RIR list must be from train partition: {rir_list_file}")
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


def load_speaker_sex_map(speakers_txt_path):
    """
    Load speaker -> sex mapping from LibriSpeech SPEAKERS.TXT.

    Args:
        speakers_txt_path: Path to SPEAKERS.TXT

    Returns:
        dict: {speaker_id (str): sex ('M' or 'F')}
    """
    sex_map = {}

    if not speakers_txt_path.exists():
        print(f"  ⚠️  SPEAKERS.TXT not found at {speakers_txt_path}")
        return sex_map

    with open(speakers_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:
                continue

            speaker_id = parts[0]
            sex = parts[1].upper()

            if speaker_id.isdigit() and sex in {'M', 'F'}:
                sex_map[speaker_id] = sex

    return sex_map


def select_balanced_speakers_by_sex(candidates, n_speakers, speaker_sex_map):
    """
    Select speakers with approximately balanced male/female distribution.

    Args:
        candidates: List of candidate speaker IDs
        n_speakers: Number of speakers to select
        speaker_sex_map: Dict mapping speaker_id -> 'M'/'F'

    Returns:
        list: Selected speaker IDs
    """
    male_candidates = [s for s in candidates if speaker_sex_map.get(s) == 'M']
    female_candidates = [s for s in candidates if speaker_sex_map.get(s) == 'F']
    unknown_candidates = [s for s in candidates if s not in speaker_sex_map]

    target_female = n_speakers // 2
    target_male = n_speakers - target_female

    selected = []

    if male_candidates:
        n_male = min(target_male, len(male_candidates))
        selected.extend(np.random.choice(male_candidates, size=n_male, replace=False).tolist())

    if female_candidates:
        n_female = min(target_female, len(female_candidates))
        selected.extend(np.random.choice(female_candidates, size=n_female, replace=False).tolist())

    remaining_needed = n_speakers - len(selected)
    if remaining_needed > 0:
        remaining_pool = [
            s for s in (male_candidates + female_candidates + unknown_candidates)
            if s not in selected
        ]
        if remaining_pool:
            n_remaining = min(remaining_needed, len(remaining_pool))
            selected.extend(np.random.choice(remaining_pool, size=n_remaining, replace=False).tolist())

    np.random.shuffle(selected)
    return selected


def extract_speaker_dvectors(librispeech_path, n_speakers, n_utterances_per_speaker,
                            sample_rate=16000, device='cuda', subsets=None, duration_per_sample=None,
                            balance_by_sex=False, speakers_txt_path=None):
    """Extract d-vectors from random speakers
    
    Args:
        librispeech_path: Path to LibriSpeech dataset
        n_speakers: Number of speakers to sample
        n_utterances_per_speaker: Number of utterances per speaker
        sample_rate: Audio sample rate
        device: Device for d-vector extraction ('cuda' or 'cpu')
        subsets: List of LibriSpeech subsets to use, or None for all
        duration_per_sample: Duration in seconds to extract from each audio file (None = use full utterance)
        balance_by_sex: If True, sample speakers with approximately balanced male/female ratio
        speakers_txt_path: Path to LibriSpeech SPEAKERS.TXT (required when balance_by_sex=True)
    """
    print(f"\n🎤 Extracting SPEAKER d-vectors...")
    print(f"  Number of speakers: {n_speakers}")
    print(f"  Utterances per speaker: {n_utterances_per_speaker}")
    if duration_per_sample is not None:
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
    
    # Discover all speakers
    librispeech_path = Path(librispeech_path)
    
    # Use provided subsets or default to all
    if subsets is None:
        subsets = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 
                   'train-clean-100', 'train-clean-360', 'train-other-500']
    
    print(f"  LibriSpeech subsets to use: {', '.join(subsets)}")
    
    all_speakers = set()
    for subset in subsets:
        subset_path = librispeech_path / subset
        if subset_path.exists():
            speaker_dirs = [d.name for d in subset_path.iterdir() if d.is_dir() and d.name.isdigit()]
            all_speakers.update(speaker_dirs)
    
    all_speakers = sorted(list(all_speakers))
    print(f"  Found {len(all_speakers)} total speakers")
    
    np.random.seed(RANDOM_SEED)

    # Sample speakers
    n_to_sample = min(n_speakers, len(all_speakers))
    if balance_by_sex:
        if not speakers_txt_path:
            print("  ⚠️  balance_by_sex=True but speakers_txt_path not provided; falling back to random sampling")
            sampled_speakers = np.random.choice(all_speakers, size=n_to_sample, replace=False)
        else:
            sex_map = load_speaker_sex_map(Path(speakers_txt_path))
            if not sex_map:
                print("  ⚠️  Could not load speaker sex map; falling back to random sampling")
                sampled_speakers = np.random.choice(all_speakers, size=n_to_sample, replace=False)
            else:
                sampled_speakers = select_balanced_speakers_by_sex(all_speakers, n_to_sample, sex_map)
                sampled_male = sum(1 for s in sampled_speakers if sex_map.get(s) == 'M')
                sampled_female = sum(1 for s in sampled_speakers if sex_map.get(s) == 'F')
                sampled_unknown = len(sampled_speakers) - sampled_male - sampled_female
                print(f"  Speaker sampling mode: balanced by sex")
                print(f"    Male: {sampled_male}, Female: {sampled_female}, Unknown: {sampled_unknown}")
    else:
        sampled_speakers = np.random.choice(all_speakers, size=n_to_sample, replace=False)
    
    results = {}
    
    for speaker_id in sampled_speakers:
        print(f"\n  Processing Speaker {speaker_id}...")
        
        # Find audio files
        audio_files = []
        for subset in subsets:
            pattern = str(librispeech_path / subset / speaker_id / '**' / '*.flac')
            audio_files.extend(glob.glob(pattern, recursive=True))
        
        if not audio_files:
            print(f"    ⚠️  No audio files found for speaker {speaker_id}")
            continue
        
        print(f"    Found {len(audio_files)} audio files")
        
        # Sample files
        n_samples = min(n_utterances_per_speaker, len(audio_files))
        sampled_files = np.random.choice(audio_files, size=n_samples, replace=False)
        
        # Extract d-vectors
        for audio_file in sampled_files:
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=sample_rate)
                
                if len(audio) < sample_rate * 0.3:  # Skip very short utterances
                    continue
                
                # Apply duration constraint if specified
                if duration_per_sample is not None:
                    target_samples = int(duration_per_sample * sample_rate)
                    
                    # Slice entire audio into segments of target duration
                    n_segments = max(1, len(audio) // target_samples)
                    
                    for seg_idx in range(n_segments):
                        start_idx = seg_idx * target_samples
                        end_idx = start_idx + target_samples
                        
                        # Extract segment
                        if end_idx <= len(audio):
                            audio_segment = audio[start_idx:end_idx]
                        else:
                            # Last segment: pad if needed
                            audio_segment = audio[start_idx:]
                            if len(audio_segment) < target_samples:
                                audio_segment = np.pad(audio_segment, (0, target_samples - len(audio_segment)), mode='constant')
                        
                        # Extract d-vector from segment
                        with torch.no_grad():
                            dvector = encoder.embed_utterance(audio_segment)
                        
                        # Store result
                        file_name = Path(audio_file).stem
                        key = f"speaker_{speaker_id}_{file_name}_seg{seg_idx:03d}"
                        results[key] = {
                            'dvector': dvector,
                            'type': 'speaker',
                            'speaker_id': speaker_id,
                            'audio_file': audio_file,
                            'segment_idx': seg_idx
                        }
                else:
                    # No duration constraint: use full utterance
                    with torch.no_grad():
                        dvector = encoder.embed_utterance(audio)
                    
                    # Store result
                    file_name = Path(audio_file).stem
                    key = f"speaker_{speaker_id}_{file_name}"
                    results[key] = {
                        'dvector': dvector,
                        'type': 'speaker',
                        'speaker_id': speaker_id,
                        'audio_file': audio_file
                    }
                
            except Exception as e:
                print(f"    ⚠️  Error processing {audio_file}: {e}")
                continue
        
        speaker_count = len([k for k in results if results[k].get('speaker_id') == speaker_id])
        print(f"    ✓ Extracted {speaker_count} d-vectors")
    
    print(f"\n✓ Total speaker d-vectors: {len(results)}")
    
    return results


def extract_overlap_dvectors(
    overlap_dirs,
    n_samples=-1,
    sample_rate=16000,
    device='cuda',
    include_main_speaker_samples=True,
    include_other_mixtures=True,
):
    """Extract d-vectors from overlap/mixed samples.

    Supports both metadata formats:
    - Legacy overlap generator: uses `overlap_file`
    - generate_full_overlap_utterances.py: uses `output_name` (+ .wav),
      optionally with `has_main_speaker` flag for mixed datasets.
    """
    # Normalize to list
    if isinstance(overlap_dirs, str):
        overlap_dirs = [overlap_dirs]

    print(f"\n🎤 Extracting OVERLAP d-vectors...")
    print(f"  Number of directories: {len(overlap_dirs)}")

    # Initialize encoder
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
            print(f"    ⚠️  metadata.json not found in {overlap_dir}")
            continue

        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)

        print(f"    Total samples available: {len(metadata_list)}")

        # Sample if requested
        if n_samples > 0 and len(metadata_list) > n_samples:
            np.random.seed(RANDOM_SEED)
            sampled_idx = np.random.choice(len(metadata_list), size=n_samples, replace=False)
            metadata_list = [metadata_list[i] for i in sampled_idx]
            print(f"    Sampled {n_samples} samples")

        audio_dir = Path(overlap_dir) / 'audio'
        n_included = 0
        n_skipped_by_filter = 0

        for metadata in metadata_list:
            try:
                has_main_speaker = metadata.get('has_main_speaker', True)

                # Filter by mixture type (main-speaker overlap vs other-only mixtures)
                if has_main_speaker and not include_main_speaker_samples:
                    n_skipped_by_filter += 1
                    continue
                if (has_main_speaker is False) and not include_other_mixtures:
                    n_skipped_by_filter += 1
                    continue

                # Backward-compatible audio file resolution
                overlap_file = metadata.get('overlap_file')
                if overlap_file:
                    audio_file = audio_dir / overlap_file
                else:
                    output_name = metadata.get('output_name', '')
                    if output_name:
                        if output_name.endswith('.wav'):
                            overlap_file = output_name
                        else:
                            overlap_file = f"{output_name}.wav"
                        audio_file = audio_dir / overlap_file
                    else:
                        sample_id = metadata.get('sample_id', 'unknown')
                        print(f"    ⚠️  Missing overlap_file/output_name in metadata sample: {sample_id}")
                        continue

                if not audio_file.exists():
                    print(f"    ⚠️  Audio file not found: {audio_file}")
                    continue

                # Load audio
                audio, _ = librosa.load(str(audio_file), sr=sample_rate)

                # Extract d-vector
                with torch.no_grad():
                    dvector = encoder.embed_utterance(audio)

                # Store result
                key = f"overlap_{Path(overlap_dir).name}_{Path(audio_file).stem}"
                all_results[key] = {
                    'dvector': dvector,
                    'type': 'overlap_main' if has_main_speaker else 'overlap_other_mixture',
                    'metadata': metadata,
                    'source_dir': str(overlap_dir),
                }
                n_included += 1

            except Exception as e:
                print(f"    ⚠️  Error processing overlap: {e}")
                continue

        dir_count = len([k for k in all_results if all_results[k].get('source_dir') == str(overlap_dir)])
        print(f"    ✓ Extracted {dir_count} d-vectors from this directory")
        if n_skipped_by_filter > 0:
            print(f"    ↪ Skipped {n_skipped_by_filter} samples due to overlap mixture filters")
        if n_included == 0:
            print(f"    ⚠️  No usable overlap samples were included from this directory")

    print(f"\n✓ Total overlap d-vectors: {len(all_results)}")

    return all_results


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir):
    """Train the autoencoder"""
    print(f"\n🏋️  Training autoencoder...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(batch_input)
            loss = criterion(reconstructed, batch_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                
                reconstructed = model(batch_input)
                loss = criterion(reconstructed, batch_target)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            print(f"  💾 Saved best model (val_loss: {val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    
    return history, best_val_loss


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    print(f"\n📊 Evaluating model on test set...")
    
    model.eval()
    model = model.to(device)
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    all_input = []
    all_reconstructed = []
    
    with torch.no_grad():
        for batch_input, batch_target in test_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            reconstructed = model(batch_input)
            loss = criterion(reconstructed, batch_target)
            
            total_loss += loss.item()
            
            all_input.append(batch_input.cpu().numpy())
            all_reconstructed.append(reconstructed.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    # Concatenate all batches
    all_input = np.concatenate(all_input, axis=0)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    
    # Compute cosine similarity
    cosine_sims = []
    for i in range(len(all_input)):
        cos_sim = np.dot(all_input[i], all_reconstructed[i]) / (
            np.linalg.norm(all_input[i]) * np.linalg.norm(all_reconstructed[i])
        )
        cosine_sims.append(cos_sim)
    
    cosine_sims = np.array(cosine_sims)
    
    print(f"\n  Test Results:")
    print(f"    MSE Loss: {avg_loss:.6f}")
    print(f"    Cosine Similarity: {cosine_sims.mean():.4f} ± {cosine_sims.std():.4f}")
    print(f"    Min Cosine Sim: {cosine_sims.min():.4f}")
    print(f"    Max Cosine Sim: {cosine_sims.max():.4f}")
    
    return {
        'mse_loss': avg_loss,
        'cosine_similarity': cosine_sims,
        'input': all_input,
        'reconstructed': all_reconstructed
    }


def plot_training_history(history, save_path):
    """Plot training and validation loss"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
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
    
    # Scatter: input vs reconstructed magnitude
    input_norms = np.linalg.norm(eval_results['input'], axis=1)
    recon_norms = np.linalg.norm(eval_results['reconstructed'], axis=1)
    
    axes[1].scatter(input_norms, recon_norms, alpha=0.6, s=30)
    axes[1].plot([input_norms.min(), input_norms.max()], 
                [input_norms.min(), input_norms.max()], 
                'r--', linewidth=2, label='Perfect reconstruction')
    axes[1].set_xlabel('Input Magnitude', fontsize=12)
    axes[1].set_ylabel('Reconstructed Magnitude', fontsize=12)
    axes[1].set_title('Magnitude Preservation', fontsize=14, fontweight='bold')
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
        f.write("D-VECTOR AUTOENCODER CONFIGURATION (IDENTITY MAPPING)\n")
        f.write("=" * 70 + "\n\n")
        
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✓ Saved human-readable config summary to: {config_txt_path}")


def _can_use_stratified_split(labels, split_fraction):
    """Check whether stratified split is feasible for the given labels and split fraction."""
    if not labels or split_fraction <= 0.0 or split_fraction >= 1.0:
        return False

    counts = Counter(labels)
    if len(counts) < 2:
        return False

    # Each class must have at least one sample on both sides after split.
    for n in counts.values():
        if n < 2:
            return False
        if (n * split_fraction) < 1 or (n * (1.0 - split_fraction)) < 1:
            return False

    return True


def split_dataset_keys_by_type(all_dvectors, validation_split, test_split, random_seed):
    """
    Split dataset keys into train/val/test while preserving type distribution when possible.

    Stratification is applied by `data['type']` if feasible; otherwise falls back to random split.
    Supports test_split = 0.0.
    """
    if validation_split < 0 or test_split < 0:
        raise ValueError("VALIDATION_SPLIT and TEST_SPLIT must be >= 0")
    if validation_split + test_split >= 1.0:
        raise ValueError("VALIDATION_SPLIT + TEST_SPLIT must be < 1")

    keys = list(all_dvectors.keys())
    labels = [all_dvectors[k].get('type', 'unknown') for k in keys]

    # First split: train+val vs test
    if test_split > 0:
        stratify_labels = labels if _can_use_stratified_split(labels, test_split) else None
        if stratify_labels is None:
            print("  ⚠️  Stratified test split not feasible; using random split")

        train_val_keys, test_keys = train_test_split(
            keys,
            test_size=test_split,
            random_state=random_seed,
            stratify=stratify_labels,
        )
    else:
        train_val_keys = keys
        test_keys = []

    # Second split: train vs val
    if validation_split > 0:
        if test_split > 0:
            val_size_adjusted = validation_split / (1 - test_split)
        else:
            val_size_adjusted = validation_split

        train_val_labels = [all_dvectors[k].get('type', 'unknown') for k in train_val_keys]
        stratify_labels = train_val_labels if _can_use_stratified_split(train_val_labels, val_size_adjusted) else None
        if stratify_labels is None:
            print("  ⚠️  Stratified validation split not feasible; using random split")

        train_keys, val_keys = train_test_split(
            train_val_keys,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=stratify_labels,
        )
    else:
        train_keys = train_val_keys
        val_keys = []

    return train_keys, val_keys, test_keys


def summarize_split_type_counts(split_name, split_keys, all_dvectors):
    """Print per-type counts for a given split."""
    split_type_counts = Counter(all_dvectors[k].get('type', 'unknown') for k in split_keys)
    print(f"  {split_name} type breakdown:")
    if not split_type_counts:
        print("    (empty)")
    else:
        for type_name, count in sorted(split_type_counts.items()):
            print(f"    {type_name}: {count}")


def main():
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # Resolve paths
    script_dir = Path(__file__).parent
    librispeech_path = Path(LIBRISPEECH_PATH)
    if not librispeech_path.is_absolute():
        librispeech_path = script_dir / librispeech_path

    speakers_txt_path = Path(LIBRISPEECH_SPEAKERS_TXT)
    if not speakers_txt_path.is_absolute():
        speakers_txt_path = script_dir / speakers_txt_path
    
    overlap_dirs = OVERLAP_SAMPLES_DIRS if isinstance(OVERLAP_SAMPLES_DIRS, list) else [OVERLAP_SAMPLES_DIRS]
    overlap_dirs_resolved = []
    for overlap_dir in overlap_dirs:
        od_path = Path(overlap_dir)
        if not od_path.is_absolute():
            od_path = script_dir / od_path
        overlap_dirs_resolved.append(od_path)
    
    save_dir = Path(MODEL_SAVE_DIR)
    if not save_dir.is_absolute():
        save_dir = script_dir / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("D-VECTOR AUTOENCODER TRAINING - IDENTITY MAPPING")
    print("Input -> Same Input (Reconstruction)")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  LibriSpeech: {librispeech_path}")
    print(f"  Number of speakers: {N_SPEAKERS}")
    print(f"  Utterances per speaker: {N_UTTERANCES_PER_SPEAKER}")
    print(f"  Balance speaker sampling by sex: {BALANCE_SPEAKERS_BY_SEX}")
    if BALANCE_SPEAKERS_BY_SEX:
        print(f"    SPEAKERS.TXT: {speakers_txt_path}")
    print(f"  Include overlap identity: {INCLUDE_OVERLAP_IDENTITY}")
    if INCLUDE_OVERLAP_IDENTITY:
        print(f"    Overlap directories: {len(overlap_dirs_resolved)}")
        print(f"    Include main-speaker overlap samples: {OVERLAP_INCLUDE_MAIN_SPEAKER_SAMPLES}")
        print(f"    Include other-only mixtures: {OVERLAP_INCLUDE_OTHER_MIXTURES}")
    print(f"  Include silence identity: {INCLUDE_SILENCE_IDENTITY}")
    if INCLUDE_SILENCE_IDENTITY:
        print(f"    Silence samples: {N_SILENCE_SAMPLES}")
    print(f"  Include noise identity: {INCLUDE_NOISE_IDENTITY}")
    if INCLUDE_NOISE_IDENTITY:
        print(f"    MUSAN root: {MUSAN_ROOT}")
        print(f"    RIRS root: {RIRS_ROOT}")
        print(f"    Noise samples per type: {N_NOISE_SAMPLES_PER_TYPE}")
        print(f"    RIR samples: {N_RIR_SAMPLES}")
        print(f"    Noise sample duration: {NOISE_SAMPLE_DURATION}s")
    print(f"  Model save dir: {save_dir}")
    print(f"\n  Training parameters:")
    print(f"    Architecture: {HIDDEN_DIMS}")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Learning rate: {LEARNING_RATE}")
    print(f"    Epochs: {NUM_EPOCHS}")
    print(f"    Val split: {VALIDATION_SPLIT:.0%}")
    print(f"    Test split: {TEST_SPLIT:.0%}")
    print(f"    Device: {DEVICE}")
    
    # Collect all d-vectors
    all_dvectors = {}
    
    # 1. Extract speaker d-vectors
    speaker_dvectors = extract_speaker_dvectors(
        str(librispeech_path),
        N_SPEAKERS,
        N_UTTERANCES_PER_SPEAKER,
        SAMPLE_RATE,
        DEVICE,
        LIBRISPEECH_SUBSETS,
        duration_per_sample=DURATION_PER_SAMPLE,
        balance_by_sex=BALANCE_SPEAKERS_BY_SEX,
        speakers_txt_path=str(speakers_txt_path)
    )
    all_dvectors.update(speaker_dvectors)
    
    # 2. Extract overlap d-vectors if enabled
    if INCLUDE_OVERLAP_IDENTITY:
        overlap_dvectors = extract_overlap_dvectors(
            overlap_dirs_resolved,
            N_OVERLAP_SAMPLES,
            SAMPLE_RATE,
            DEVICE,
            include_main_speaker_samples=OVERLAP_INCLUDE_MAIN_SPEAKER_SAMPLES,
            include_other_mixtures=OVERLAP_INCLUDE_OTHER_MIXTURES
        )
        all_dvectors.update(overlap_dvectors)
    
    # 3. Extract silence d-vectors if enabled
    if INCLUDE_SILENCE_IDENTITY:
        silence_dvectors = extract_silence_dvectors(
            N_SILENCE_SAMPLES,
            duration_per_sample=1.6,
            sample_rate=SAMPLE_RATE,
            device=DEVICE
        )
        all_dvectors.update(silence_dvectors)
    
    # 4. Extract noise d-vectors if enabled
    if INCLUDE_NOISE_IDENTITY:
        noise_dvectors = extract_noise_dvectors(
            MUSAN_ROOT,
            RIRS_ROOT,
            n_noise_samples_per_type=N_NOISE_SAMPLES_PER_TYPE,
            n_rir_samples=N_RIR_SAMPLES,
            noise_sample_duration=NOISE_SAMPLE_DURATION,
            sample_rate=SAMPLE_RATE,
            device=DEVICE
        )
        all_dvectors.update(noise_dvectors)
    
    print(f"\n✓ Total d-vectors collected: {len(all_dvectors)}")
    
    # Count by type
    type_counts = defaultdict(int)
    for data in all_dvectors.values():
        type_counts[data['type']] += 1
    
    print(f"  Breakdown by type:")
    for type_name, count in sorted(type_counts.items()):
        print(f"    {type_name}: {count}")
    
    # Split into train/val/test (type-aware stratified when feasible)
    train_keys, val_keys, test_keys = split_dataset_keys_by_type(
        all_dvectors,
        VALIDATION_SPLIT,
        TEST_SPLIT,
        RANDOM_SEED,
    )
    
    print(f"\n📊 Dataset split (train:{1-VALIDATION_SPLIT-TEST_SPLIT:.0%} / val:{VALIDATION_SPLIT:.0%} / test:{TEST_SPLIT:.0%}):")
    print(f"  Train: {len(train_keys)}")
    print(f"  Val: {len(val_keys)}")
    print(f"  Test: {len(test_keys)}")
    summarize_split_type_counts('Train', train_keys, all_dvectors)
    summarize_split_type_counts('Val', val_keys, all_dvectors)
    summarize_split_type_counts('Test', test_keys, all_dvectors)
    
    # Create datasets
    train_dvectors = {k: all_dvectors[k] for k in train_keys}
    val_dvectors = {k: all_dvectors[k] for k in val_keys}
    test_dvectors = {k: all_dvectors[k] for k in test_keys}
    
    train_dataset = DvectorIdentityDataset(train_dvectors)
    val_dataset = DvectorIdentityDataset(val_dvectors)
    test_dataset = DvectorIdentityDataset(test_dvectors)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get d-vector dimension
    sample_dvec = list(all_dvectors.values())[0]['dvector']
    dvector_dim = sample_dvec.shape[0]
    print(f"\n  D-vector dimension: {dvector_dim}")
    
    # Create model
    model = DvectorAutoencoder(
        input_dim=dvector_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE
    )
    
    print(f"\n🏗️  Model architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
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
    
    # Evaluate on test set (if available)
    eval_results = None
    if len(test_dataset) > 0:
        eval_results = evaluate_model(model, test_loader, DEVICE)
    else:
        print("\n⚠️  TEST_SPLIT is 0 or no test samples available; skipping test evaluation")

    # Plot results
    plot_training_history(history, save_dir / 'training_history.png')
    if eval_results is not None:
        plot_reconstruction_quality(eval_results, save_dir / 'reconstruction_quality.png')
    
    # Save final model and config
    config = {
        # Model
        'input_dim': dvector_dim,
        'hidden_dims': HIDDEN_DIMS,
        'dropout_rate': DROPOUT_RATE,
        # Training
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'validation_split': VALIDATION_SPLIT,
        'test_split': TEST_SPLIT,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        # Data
        'librispeech_path': str(librispeech_path),
        'balance_speakers_by_sex': BALANCE_SPEAKERS_BY_SEX,
        'librispeech_speakers_txt': str(speakers_txt_path),
        'n_speakers': N_SPEAKERS,
        'n_utterances_per_speaker': N_UTTERANCES_PER_SPEAKER,
        'include_overlap_identity': INCLUDE_OVERLAP_IDENTITY,
        'overlap_samples_dirs': [str(d) for d in overlap_dirs_resolved] if INCLUDE_OVERLAP_IDENTITY else [],
        'n_overlap_samples': N_OVERLAP_SAMPLES if INCLUDE_OVERLAP_IDENTITY else 0,
        'overlap_include_main_speaker_samples': OVERLAP_INCLUDE_MAIN_SPEAKER_SAMPLES if INCLUDE_OVERLAP_IDENTITY else False,
        'overlap_include_other_mixtures': OVERLAP_INCLUDE_OTHER_MIXTURES if INCLUDE_OVERLAP_IDENTITY else False,
        'include_silence_identity': INCLUDE_SILENCE_IDENTITY,
        'n_silence_samples': N_SILENCE_SAMPLES if INCLUDE_SILENCE_IDENTITY else 0,
        'include_noise_identity': INCLUDE_NOISE_IDENTITY,
        'musan_root': MUSAN_ROOT if INCLUDE_NOISE_IDENTITY else '',
        'rirs_root': RIRS_ROOT if INCLUDE_NOISE_IDENTITY else '',
        'n_noise_samples_per_type': N_NOISE_SAMPLES_PER_TYPE if INCLUDE_NOISE_IDENTITY else 0,
        'n_rir_samples': N_RIR_SAMPLES if INCLUDE_NOISE_IDENTITY else 0,
        'noise_sample_duration': NOISE_SAMPLE_DURATION if INCLUDE_NOISE_IDENTITY else 0,
        'sample_rate': SAMPLE_RATE,
        # Dataset stats
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'type_counts': dict(type_counts),
        # Results
        'best_val_loss': best_val_loss,
        'test_mse_loss': eval_results['mse_loss'] if eval_results is not None else None,
        'test_cosine_similarity_mean': eval_results['cosine_similarity'].mean() if eval_results is not None else None,
        'test_cosine_similarity_std': eval_results['cosine_similarity'].std() if eval_results is not None else None,
        # Other
        'random_seed': RANDOM_SEED,
        'device': DEVICE,
    }
    
    save_model_and_config(model, config, save_dir)
    
    # Save evaluation results (if available)
    if eval_results is not None:
        eval_save_path = save_dir / 'evaluation_results.pkl'
        with open(eval_save_path, 'wb') as f:
            pickle.dump(eval_results, f)
        print(f"✓ Saved evaluation results to: {eval_save_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    if eval_results is not None:
        print(f"  Test MSE Loss: {eval_results['mse_loss']:.6f}")
        print(f"  Test Cosine Similarity: {eval_results['cosine_similarity'].mean():.4f} ± {eval_results['cosine_similarity'].std():.4f}")
    else:
        print("  Test evaluation: skipped (no test split)")
    print(f"\n💡 Interpretation:")
    print(f"  Cosine similarity > 0.95 = Excellent reconstruction")
    print(f"  Cosine similarity > 0.90 = Good reconstruction")
    print(f"  Cosine similarity > 0.80 = Moderate reconstruction")
    print(f"\n📁 Model saved to: {save_dir}")
    print(f"\n🎓 Model was trained on IDENTITY MAPPING:")
    print(f"  - Speaker d-vectors: input -> same input")
    if INCLUDE_OVERLAP_IDENTITY:
        print(f"  - Overlap d-vectors: input -> same input")
    if INCLUDE_SILENCE_IDENTITY:
        print(f"  - Silence d-vectors: input -> same input")
    print(f"\n🚀 Use the model for d-vector reconstruction:")
    print(f"  1. Load model: model.load_state_dict(torch.load('{save_dir / 'final_model.pth'}'))")
    print(f"  2. Load config: config = pickle.load(open('{save_dir / 'config.pkl'}', 'rb'))")
    print(f"  3. Reconstruct: reconstructed = model(input_dvector_tensor)")


if __name__ == "__main__":
    main()
