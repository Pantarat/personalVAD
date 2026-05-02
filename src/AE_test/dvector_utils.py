#!/usr/bin/env python3
"""
Shared utilities for d-vector extraction and autoencoder operations
Reduces code duplication across test scripts
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import librosa
import glob
import sys
from pathlib import Path
from collections import defaultdict
from autoencoder_utils import DvectorAutoencoder as SharedDvectorAutoencoder

# Add parent directory to path for resemblyzer_mod
sys.path.insert(0, str(Path(__file__).parent.parent))
from resemblyzer_mod import VoiceEncoder
import pickle
import json


class DvectorAutoencoder(nn.Module):
    """
    Autoencoder to map overlap d-vectors to clean d-vectors
    
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
        n_encoder_layers = len(hidden_dims) // 2
        
        for i in range(n_encoder_layers):
            encoder_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            encoder_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        bottleneck_dim = hidden_dims[n_encoder_layers]
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        encoder_layers.append(nn.BatchNorm1d(bottleneck_dim))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = bottleneck_dim
        
        for i in range(n_encoder_layers + 1, len(hidden_dims)):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dims[i]))
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[i]
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Only encode to bottleneck"""
        return self.encoder(x)
    
    def decode(self, z):
        """Only decode from bottleneck"""
        return self.decoder(z)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_trained_model(model_dir, device='cuda'):
    """Load trained autoencoder model"""
    print(f"\n🔧 Loading trained model...")
    
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / 'config.pkl'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    print(f"  Config loaded:")
    print(f"    Input dim: {config['input_dim']}")
    print(f"    Hidden dims: {config['hidden_dims']}")
    if 'hidden_dims' in config:
        bottleneck_idx = len(config['hidden_dims']) // 2
        print(f"    Bottleneck dim: {config['hidden_dims'][bottleneck_idx]}")
    
    # Create model
    model = SharedDvectorAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config.get('dropout_rate', 0.2),
        norm_type=config.get('norm_type', 'batchnorm'),
        use_residual=config.get('use_residual', False),
        residual_scale_init=config.get('residual_scale_init', 0.5),
    )
    
    # Load weights
    model_path = model_dir / 'final_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    print(f"✓ Model loaded from: {model_path}")
    
    return model, config


def initialize_encoder(device='cuda'):
    """Initialize VoiceEncoder with proper device handling"""
    if torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cuda')
    else:
        device_obj = torch.device('cpu')
    
    encoder = VoiceEncoder(device=device_obj)
    return encoder, device_obj


def extract_dvector_from_audio(audio_path, encoder, device_obj, sample_rate=16000):
    """Extract d-vector from a single audio file"""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    fbanks = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=40
    ).astype('float32').T
    
    with torch.no_grad():
        fbanks_tensor = torch.from_numpy(fbanks).unsqueeze(0).to(device_obj).float()
        dvector = encoder.forward(fbanks_tensor).cpu().numpy().squeeze()
    
    # Normalize
    dvector = dvector / np.linalg.norm(dvector)
    
    return dvector


def extract_clean_speaker_dvectors(librispeech_path, speakers, n_samples, 
                                   sample_rate=16000, device='cuda', random_seed=42, 
                                   splits=None):
    """Extract clean d-vectors for all specified speakers"""
    print(f"\n🎤 Extracting CLEAN d-vectors for all speakers...")
    print(f"  Speakers: {speakers}")
    print(f"  Number of samples per speaker: {n_samples}")
    
    if splits is None:
        splits = ['dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360']
    
    encoder, device_obj = initialize_encoder(device)
    
    results = {}
    librispeech_path = Path(librispeech_path)
    
    # Extract for each speaker
    for speaker in speakers:
        print(f"\n  Processing speaker {speaker}...")
        
        # Try each split in order until we find audio files
        audio_files = []
        for split in splits:
            speaker_pattern = str(librispeech_path / split / speaker / "*" / "*.flac")
            split_files = glob.glob(speaker_pattern)
            if split_files:
                audio_files = split_files
                print(f"    Found {len(audio_files)} audio files in {split}")
                break
        
        if not audio_files:
            print(f"    ⚠️  No audio files found for speaker {speaker} in any split: {', '.join(splits)}")
            continue
        
        # Sample files
        n_samples_speaker = min(n_samples, len(audio_files))
        np.random.seed(random_seed)
        sampled_files = np.random.choice(audio_files, size=n_samples_speaker, replace=False)
        
        # Extract d-vectors
        speaker_count = 0
        for audio_file in sampled_files:
            try:
                dvector = extract_dvector_from_audio(audio_file, encoder, device_obj, sample_rate)
                
                utterance_key = f"clean_{speaker}_{Path(audio_file).stem}"
                
                results[utterance_key] = {
                    'dvector': dvector,
                    'speaker': speaker,
                    'type': 'clean'
                }
                speaker_count += 1
                
            except Exception as e:
                continue
        
        print(f"    ✓ Extracted {speaker_count} clean d-vectors for speaker {speaker}")
    
    print(f"\n✓ Total: Extracted {len(results)} clean d-vectors across all speakers")
    return results


def extract_overlap_dvectors(overlap_dir, n_per_speaker=-1, sample_rate=16000, 
                            device='cuda', dvector_type='overlap_extracted'):
    """Extract d-vectors directly from overlapped audio files"""
    print(f"\n🎤 Extracting d-vectors from OVERLAPPED AUDIO...")
    
    # Load metadata
    metadata_path = Path(overlap_dir) / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"  Total samples available: {len(metadata_list)}")
    
    # Debug: Check metadata format
    if len(metadata_list) > 0:
        sample_keys = list(metadata_list[0].keys())
        print(f"  Metadata format: {sample_keys}")
    
    # Filter by number per speaker if specified
    if n_per_speaker > 0:
        # Check metadata format
        has_other_speakers = len(metadata_list) > 0 and 'other_speakers' in metadata_list[0]
        has_overlapping_speakers = len(metadata_list) > 0 and 'overlapping_speakers' in metadata_list[0]
        has_target_present = len(metadata_list) > 0 and 'is_target_present' in metadata_list[0]
        
        # Non-target overlap: has overlapping_speakers and is_target_present=False
        is_non_target = has_overlapping_speakers and (
            not has_target_present or 
            (len(metadata_list) > 0 and not metadata_list[0].get('is_target_present', True))
        )
        
        print(f"  Detected format: {'non-target overlap' if is_non_target else 'full overlap'}")
        
        if is_non_target:
            # For non-target overlaps, just take n_per_speaker samples total
            import random
            random.seed(42)  # For reproducibility
            if len(metadata_list) > n_per_speaker:
                metadata_list = random.sample(metadata_list, n_per_speaker)
            print(f"  Using {len(metadata_list)} non-target overlap samples (total)")
        else:
            # For full overlaps, filter by other speaker
            filtered_metadata = []
            speaker_counts = defaultdict(int)
            
            for metadata in metadata_list:
                if 'other_speakers' in metadata:
                    other_speaker = metadata['other_speakers'][0]
                else:
                    continue
                    
                if speaker_counts[other_speaker] < n_per_speaker:
                    filtered_metadata.append(metadata)
                    speaker_counts[other_speaker] += 1
            
            metadata_list = filtered_metadata
            print(f"  Using {len(metadata_list)} samples ({n_per_speaker} per speaker)")
    
    encoder, device_obj = initialize_encoder(device)
    
    results = {}
    audio_dir = Path(overlap_dir) / 'audio'
    
    print(f"  Audio directory: {audio_dir}")
    print(f"  Processing {len(metadata_list)} samples...")
    
    files_not_found = 0
    for metadata in metadata_list:
        try:
            output_name = metadata['output_name']
            # Add .wav extension if not present
            if not output_name.endswith('.wav'):
                output_name = output_name + '.wav'
            audio_path = audio_dir / output_name
            
            if not audio_path.exists():
                files_not_found += 1
                if files_not_found <= 3:  # Show first 3 missing files
                    print(f"    ⚠️  File not found: {audio_path}")
                continue
            
            dvector = extract_dvector_from_audio(audio_path, encoder, device_obj, sample_rate)
            
            key_prefix = dvector_type.replace('_extracted', '')
            
            # Handle different metadata formats
            if 'other_speakers' in metadata:
                # Full overlap format (target + other speaker)
                speaker = metadata['other_speakers'][0]
                main_speaker = metadata['main_speaker']
            elif 'overlapping_speakers' in metadata:
                # Non-target overlap format with overlapping_speakers list
                overlapping = metadata['overlapping_speakers']
                if isinstance(overlapping, list):
                    speaker = ','.join(str(s) for s in overlapping)
                else:
                    speaker = str(overlapping)
                main_speaker = metadata.get('target_speaker', None)
            elif 'speaker1' in metadata and 'speaker2' in metadata:
                # Alternative non-target overlap format (2 other speakers, no target)
                speaker = f"{metadata['speaker1']},{metadata['speaker2']}"
                main_speaker = None
            else:
                continue
            
            results[f"{key_prefix}_{metadata['sample_id']}"] = {
                'dvector': dvector,
                'speaker': speaker,
                'type': dvector_type,
                'main_speaker': main_speaker
            }
            
        except Exception as e:
            continue
    
    if files_not_found > 0:
        print(f"  ⚠️  {files_not_found} files not found (skipped)")
    
    print(f"✓ Extracted {len(results)} d-vectors from overlapped audio")
    
    # Show distribution
    speaker_counts = defaultdict(int)
    for data in results.values():
        speaker_counts[data['speaker']] += 1
    
    print(f"  Distribution by other speaker:")
    for speaker in sorted(speaker_counts.keys()):
        print(f"    Speaker {speaker}: {speaker_counts[speaker]} samples")
    
    return results


def encode_to_bottleneck(model, dvector_data, device='cuda'):
    """Use encoder to transform d-vectors to bottleneck representation"""
    bottleneck_dim = model.hidden_dims[len(model.hidden_dims)//2]
    print(f"\n🔮 Encoding d-vectors to {bottleneck_dim}-dim bottleneck...")
    
    model.eval()
    results = {}
    
    with torch.no_grad():
        for key, data in dvector_data.items():
            dvector = data['dvector']
            dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device)
            
            # Use only encoder
            bottleneck = model.encode(dvector_tensor).cpu().numpy().squeeze()
            
            # Normalize
            bottleneck = bottleneck / np.linalg.norm(bottleneck)
            
            results[key] = {
                'dvector': bottleneck,  # Now bottleneck representation
                'speaker': data['speaker'],
                'type': data['type'],
                'main_speaker': data.get('main_speaker', data['speaker'])
            }
    
    print(f"✓ Encoded {len(results)} d-vectors to bottleneck space")
    return results


def predict_clean_dvectors(model, overlap_dvectors, device='cuda'):
    """Use model to predict clean d-vectors from overlap"""
    print(f"\n🔮 Predicting clean d-vectors from overlap...")
    
    model.eval()
    results = {}
    
    with torch.no_grad():
        for key, data in overlap_dvectors.items():
            overlap_dvec = data['dvector']
            overlap_tensor = torch.FloatTensor(overlap_dvec).unsqueeze(0).to(device)
            
            predicted = model(overlap_tensor).cpu().numpy().squeeze()
            
            # Normalize
            predicted = predicted / np.linalg.norm(predicted)
            
            results[key] = {
                'dvector': predicted,
                'speaker': data['speaker'],  # Other speaker in overlap
                'type': 'predicted',
                'main_speaker': data['main_speaker']
            }
    
    print(f"✓ Generated {len(results)} predictions")
    return results
