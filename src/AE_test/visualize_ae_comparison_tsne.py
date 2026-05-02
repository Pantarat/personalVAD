#!/usr/bin/env python3
"""
Autoencoder Comparison t-SNE Visualization
Compare raw d-vectors vs autoencoder-processed d-vectors in t-SNE space

This script visualizes how the autoencoder transforms the d-vector space
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import glob
import librosa
import traceback
from pathlib import Path
from collections import defaultdict

from dvector_utils import initialize_encoder
from tsne_utils import compute_tsne
from autoencoder_utils import load_autoencoder

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
LIBRISPEECH_PATH = '../../data/LibriSpeech'
OVERLAP_SAMPLES_DIR = './test_outputs/300Dev_5s_trainOther500_100pctAmp_40000'

# LibriSpeech splits to search (in priority order)
LIBRISPEECH_SPLITS = ['test-clean']

# Speakers to visualize (all treated equally, no main speaker)
SPEAKERS = [
    "61","121","237","260","672","908"
]

# Number of samples
N_SINGLE_SAMPLES_PER_SPEAKER = 10

# Overlap samples (optional)
INCLUDE_OVERLAP_SAMPLES = False  # Set to False to visualize only single-speaker LibriSpeech utterances
N_OVERLAP_SAMPLES = 100

# Autoencoder (REQUIRED)
AUTOENCODER_MODEL_PATH = './test_outputs/dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000/best_model.pth'
AUTOENCODER_CONFIG_PATH = './test_outputs/dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000/config.pkl'

# Comparison mode
# 'side_by_side': Two separate plots (raw vs AE)
# 'combined': Single plot with both raw and AE samples
COMPARISON_MODE = 'combined'  # or 'combined'

# t-SNE parameters
TSNE_DIMENSIONS = 2
TSNE_PERPLEXITY = 30
RANDOM_SEED = 42

# Output settings
OUTPUT_PATH = './test_outputs/ae_comparison_tsne.png'

# Device
DEVICE = 'cuda'

# Audio settings
SAMPLE_RATE = 16000

# ============================================================================


def extract_single_speaker_dvectors(librispeech_path, speaker_ids, n_samples_per_speaker, 
                                   sample_rate=16000, device='cuda', random_seed=42, splits=None):
    """Extract d-vectors from single-speaker LibriSpeech files"""
    print(f"\n🎤 Extracting d-vectors from single-speaker audio...")
    print(f"  Speakers: {', '.join(speaker_ids)}")
    print(f"  Samples per speaker: {n_samples_per_speaker}")
    
    if splits is None:
        splits = ['dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360']
    
    encoder, device_obj = initialize_encoder(device)
    
    dvectors = {}
    librispeech_path = Path(librispeech_path)
    
    for speaker_id in speaker_ids:
        print(f"\n  Processing Speaker {speaker_id}...")
        
        # Try each split in order until we find audio files
        audio_files = []
        for split in splits:
            speaker_pattern = str(librispeech_path / split / speaker_id / "*" / "*.flac")
            split_files = glob.glob(speaker_pattern)
            if split_files:
                audio_files.extend(split_files)
                print(f"    Found {len(split_files)} files in {split}")
        
        if not audio_files:
            print(f"    ⚠️  No audio files found in any split: {', '.join(splits)}")
            continue
        
        n_samples = min(n_samples_per_speaker, len(audio_files))
        sampled_files = np.random.RandomState(random_seed).choice(audio_files, size=n_samples, replace=False)
        
        for audio_file in sampled_files:
            try:
                wav, sr = librosa.load(audio_file, sr=sample_rate)
                
                if len(wav) < sample_rate * 0.5:
                    continue
                
                with torch.no_grad():
                    dvector = encoder.embed_utterance(wav)
                
                file_id = Path(audio_file).stem
                key = f"single_{speaker_id}_{file_id}"
                dvectors[key] = {
                    'dvector': dvector,
                    'type': 'single',
                    'speaker': speaker_id,
                    'source': audio_file
                }
                
            except Exception as e:
                print(f"    ⚠️  Error processing {audio_file}: {e}")
                continue
        
        print(f"    ✓ Extracted {len([k for k in dvectors.keys() if f'single_{speaker_id}' in k])} d-vectors")
    
    print(f"\n✓ Total single-speaker d-vectors: {len(dvectors)}")
    return dvectors


def load_metadata(overlap_dir):
    """Load metadata from overlap samples"""
    metadata_path = Path(overlap_dir) / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded metadata for {len(metadata)} overlap samples")
    return metadata


def extract_overlap_dvectors(overlap_dir, metadata_list, n_samples, speakers, 
                            sample_rate=16000, device='cuda', random_seed=42):
    """Extract d-vectors from overlap samples (filtered by speakers list)"""
    print(f"\n🎤 Extracting d-vectors from overlap samples...")
    print(f"  Target samples: {n_samples}")
    print(f"  Filtering for speakers: {', '.join(speakers)}")
    
    encoder, device_obj = initialize_encoder(device)
    
    # Filter metadata to only include samples with specified speakers
    filtered_metadata = []
    for metadata in metadata_list:
        all_spks = [metadata.get('main_speaker', '')] + metadata.get('other_speakers', [])
        if any(spk in speakers for spk in all_spks):
            filtered_metadata.append(metadata)
    
    print(f"  Filtered: {len(metadata_list)} → {len(filtered_metadata)} samples")
    
    # Sample from filtered metadata
    if len(filtered_metadata) > n_samples:
        np.random.seed(random_seed)
        filtered_metadata = np.random.choice(filtered_metadata, size=n_samples, replace=False).tolist()
    elif len(filtered_metadata) < n_samples:
        print(f"  ⚠️  Only {len(filtered_metadata)} samples available (requested {n_samples})")
    
    metadata_list = filtered_metadata
    dvectors = {}
    audio_dir = Path(overlap_dir) / 'audio'
    
    for i, metadata in enumerate(metadata_list):
        try:
            output_name = metadata['output_name']
            if not output_name.endswith('.wav'):
                output_name = f"{output_name}.wav"
            audio_path = audio_dir / output_name
            
            if not audio_path.exists():
                print(f"    ⚠️  Audio file not found: {audio_path}")
                continue
            
            wav, sr = librosa.load(str(audio_path), sr=sample_rate)
            
            if len(wav) < sample_rate * 0.5:
                continue
            
            with torch.no_grad():
                dvector = encoder.embed_utterance(wav)
            
            sample_id = metadata['sample_id']
            main_speaker = metadata['main_speaker']
            other_speakers_list = metadata['other_speakers']
            
            key = f"overlap_{sample_id}"
            dvectors[key] = {
                'dvector': dvector,
                'type': 'overlap',
                'speaker': other_speakers_list[0] if len(other_speakers_list) == 1 else main_speaker,
                'main_speaker': main_speaker,
                'other_speaker': other_speakers_list[0] if len(other_speakers_list) == 1 else ','.join(other_speakers_list)
            }
            
            if (i + 1) % 10 == 0 or (i + 1) == len(metadata_list):
                print(f"    Processed {i + 1}/{len(metadata_list)} overlap samples")
            
        except Exception as e:
            print(f"    ⚠️  Error processing sample {metadata.get('sample_id', 'unknown')}: {e}")
            continue
    
    print(f"✓ Total overlap d-vectors: {len(dvectors)}")
    return dvectors


def apply_autoencoder_to_data(data_dict, autoencoder, device='cuda'):
    """Apply autoencoder to all d-vectors in data_dict"""
    print(f"\n🤖 Applying autoencoder to d-vectors...")
    
    autoencoder.eval()
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    autoencoder = autoencoder.to(device_obj)
    
    processed_data = {}
    
    for key, data in data_dict.items():
        dvector = data['dvector']
        dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device_obj)
        
        with torch.no_grad():
            # Full reconstruction through encoder-decoder
            reconstructed = autoencoder(dvector_tensor)
            reconstructed_np = reconstructed.cpu().numpy().squeeze()
        
        # Create new entry with processed d-vector
        processed_data[key] = data.copy()
        processed_data[key]['dvector'] = reconstructed_np
        processed_data[key]['original_dvector'] = dvector
    
    print(f"✓ Processed {len(processed_data)} d-vectors through autoencoder")
    return processed_data


def plot_comparison_side_by_side(raw_embedding, ae_embedding, data_dict, keys, output_path=None):
    """Create side-by-side comparison of raw vs AE d-vectors"""
    print("\nCreating side-by-side comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Separate by type
    single_mask = np.array([data_dict[k]['type'] == 'single' for k in keys])
    overlap_mask = np.array([data_dict[k]['type'] == 'overlap' for k in keys])
    
    # Get speakers
    speakers = [data_dict[k]['speaker'] for k in keys]
    unique_speakers = sorted(set(s for s in speakers if s is not None))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    speaker_to_color = {spk: colors[i] for i, spk in enumerate(unique_speakers)}
    
    # Plot raw d-vectors (left) - all speakers with circles
    for speaker in unique_speakers:
        mask = single_mask & np.array([data_dict[k]['speaker'] == speaker for k in keys])
        if mask.sum() > 0:
            ax1.scatter(
                raw_embedding[mask, 0], raw_embedding[mask, 1],
                c=[speaker_to_color[speaker]], label=f'Single {speaker}',
                alpha=0.7, s=80, marker='o', edgecolors='black', linewidth=0.5
            )
    
    for speaker in unique_speakers:
        mask = overlap_mask & np.array([data_dict[k].get('other_speaker', '') == speaker for k in keys])
        if mask.sum() > 0:
            ax1.scatter(
                raw_embedding[mask, 0], raw_embedding[mask, 1],
                c=[speaker_to_color[speaker]], label=f'Overlap+{speaker}',
                alpha=0.7, s=120, marker='s', edgecolors='black', linewidth=0.5
            )
    
    # Plot AE d-vectors (right) - all speakers with circles
    for speaker in unique_speakers:
        mask = single_mask & np.array([data_dict[k]['speaker'] == speaker for k in keys])
        if mask.sum() > 0:
            ax2.scatter(
                ae_embedding[mask, 0], ae_embedding[mask, 1],
                c=[speaker_to_color[speaker]], label=f'Single {speaker}',
                alpha=0.7, s=80, marker='o', edgecolors='black', linewidth=0.5
            )
    
    for speaker in unique_speakers:
        mask = overlap_mask & np.array([data_dict[k].get('other_speaker', '') == speaker for k in keys])
        if mask.sum() > 0:
            ax2.scatter(
                ae_embedding[mask, 0], ae_embedding[mask, 1],
                c=[speaker_to_color[speaker]], label=f'Overlap+{speaker}',
                alpha=0.7, s=120, marker='s', edgecolors='black', linewidth=0.5
            )
    
    # Configure left plot
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_title('Raw D-Vectors', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Configure right plot
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.set_title('Autoencoder-Processed D-Vectors', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")
    
    plt.show()
    return fig


def plot_comparison_combined(embedding, data_dict, keys, output_path=None):
    """Create combined plot showing raw vs AE d-vectors together"""
    print("\nCreating combined comparison plot...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Split keys into raw and AE
    raw_keys = [k for k in keys if not k.endswith('_ae')]
    ae_keys = [k for k in keys if k.endswith('_ae')]
    
    raw_indices = [i for i, k in enumerate(keys) if not k.endswith('_ae')]
    ae_indices = [i for i, k in enumerate(keys) if k.endswith('_ae')]
    
    # Get unique speakers (from raw keys)
    speakers = [data_dict[k]['speaker'] for k in raw_keys]
    unique_speakers = sorted(set(s for s in speakers if s is not None))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    speaker_to_color = {spk: colors[i] for i, spk in enumerate(unique_speakers)}
    
    # Plot raw d-vectors (filled markers) - all speakers with circles
    for speaker in unique_speakers:
        raw_single_mask = np.array([
            i in raw_indices and data_dict[keys[i]]['type'] == 'single' and data_dict[keys[i]]['speaker'] == speaker
            for i in range(len(keys))
        ])
        if raw_single_mask.sum() > 0:
            ax.scatter(
                embedding[raw_single_mask, 0], embedding[raw_single_mask, 1],
                c=[speaker_to_color[speaker]], label=f'Raw Single {speaker}',
                alpha=0.7, s=80, marker='o', edgecolors='black', linewidth=0.5
            )
    
    # Plot AE d-vectors (hollow markers) - all speakers with circles
    for speaker in unique_speakers:
        ae_single_mask = np.array([
            i in ae_indices and data_dict[keys[i]]['type'] == 'single' and data_dict[keys[i]]['speaker'] == speaker
            for i in range(len(keys))
        ])
        if ae_single_mask.sum() > 0:
            ax.scatter(
                embedding[ae_single_mask, 0], embedding[ae_single_mask, 1],
                c='none', label=f'AE Single {speaker}',
                alpha=0.7, s=80, marker='o', edgecolors=speaker_to_color[speaker], linewidth=2
            )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Raw vs Autoencoder D-Vectors Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")
    
    plt.show()
    return fig


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    script_dir = Path(__file__).parent
    librispeech_path = script_dir / LIBRISPEECH_PATH
    overlap_dir = script_dir / OVERLAP_SAMPLES_DIR
    output_path = script_dir / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AUTOENCODER COMPARISON t-SNE: RAW vs AE D-VECTORS")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  Speakers: {', '.join(SPEAKERS)}")
    print(f"  Single samples per speaker: {N_SINGLE_SAMPLES_PER_SPEAKER}")
    print(f"  Include overlap samples: {INCLUDE_OVERLAP_SAMPLES}")
    if INCLUDE_OVERLAP_SAMPLES:
        print(f"    Overlap samples: {N_OVERLAP_SAMPLES}")
    print(f"  Autoencoder model: {AUTOENCODER_MODEL_PATH}")
    print(f"  Comparison mode: {COMPARISON_MODE}")
    print(f"  t-SNE dimensions: {TSNE_DIMENSIONS}D")
    print(f"  Output: {output_path}")
    
    # Extract single-speaker d-vectors
    single_dvectors = extract_single_speaker_dvectors(
        librispeech_path, SPEAKERS, N_SINGLE_SAMPLES_PER_SPEAKER,
        SAMPLE_RATE, DEVICE, RANDOM_SEED, LIBRISPEECH_SPLITS
    )
    
    # Extract overlap d-vectors (optional)
    overlap_dvectors = {}
    if INCLUDE_OVERLAP_SAMPLES:
        try:
            metadata_list = load_metadata(overlap_dir)
        except FileNotFoundError as e:
            print(f"\n⚠️  Warning: {e}")
            print(f"  Continuing with single-speaker samples only...")
        else:
            overlap_dvectors = extract_overlap_dvectors(
                overlap_dir, metadata_list, N_OVERLAP_SAMPLES, SPEAKERS,
                SAMPLE_RATE, DEVICE, RANDOM_SEED
            )
    else:
        print(f"\n⏭️  Skipping overlap samples (INCLUDE_OVERLAP_SAMPLES=False)")
    
    # Combine raw d-vectors
    raw_data = {}
    raw_data.update(single_dvectors)
    raw_data.update(overlap_dvectors)
    
    if len(raw_data) < 2:
        print(f"\n❌ Error: Need at least 2 samples, found {len(raw_data)}")
        return
    
    print(f"\n📊 Extracted data:")
    print(f"  Single-speaker: {len(single_dvectors)}")
    if INCLUDE_OVERLAP_SAMPLES:
        print(f"  Overlap: {len(overlap_dvectors)}")
    print(f"  Total: {len(raw_data)}")
    
    # Load autoencoder
    print(f"\n🤖 Loading autoencoder...")
    try:
        from autoencoder_utils import load_autoencoder_with_config
        autoencoder, config = load_autoencoder_with_config(
            script_dir / AUTOENCODER_MODEL_PATH,
            script_dir / AUTOENCODER_CONFIG_PATH,
            device=DEVICE
        )
        print(f"✓ Autoencoder loaded")
        print(f"  Architecture: {config.get('hidden_dims', 'unknown')}")
    except Exception as e:
        print(f"❌ Error loading autoencoder: {e}")
        traceback.print_exc()
        return
    
    # Apply autoencoder to create processed data
    ae_data = apply_autoencoder_to_data(raw_data, autoencoder, DEVICE)
    
    # Adjust perplexity
    n_samples = len(raw_data)
    perplexity = min(TSNE_PERPLEXITY, (n_samples - 1) // 3)
    if perplexity < 5:
        perplexity = 5
    
    if perplexity != TSNE_PERPLEXITY:
        print(f"\n⚠️  Adjusted perplexity: {TSNE_PERPLEXITY} → {perplexity}")
    
    if COMPARISON_MODE == 'side_by_side':
        # Compute t-SNE separately for raw and AE
        print(f"\n{'='*70}")
        print("COMPUTING t-SNE FOR RAW D-VECTORS")
        print(f"{'='*70}")
        raw_embedding, raw_keys, _, _ = compute_tsne(
            raw_data, perplexity, TSNE_DIMENSIONS, RANDOM_SEED
        )
        
        print(f"\n{'='*70}")
        print("COMPUTING t-SNE FOR AUTOENCODER D-VECTORS")
        print(f"{'='*70}")
        ae_embedding, ae_keys, _, _ = compute_tsne(
            ae_data, perplexity, TSNE_DIMENSIONS, RANDOM_SEED
        )
        
        # Plot side by side
        if TSNE_DIMENSIONS == 2:
            plot_comparison_side_by_side(
                raw_embedding, ae_embedding, raw_data, raw_keys, str(output_path)
            )
        else:
            print("⚠️  3D side-by-side comparison not yet implemented")
    
    elif COMPARISON_MODE == 'combined':
        # Combine both datasets with suffix for AE
        combined_data = {}
        for key, data in raw_data.items():
            combined_data[key] = data
        for key, data in ae_data.items():
            combined_data[f"{key}_ae"] = data
        
        print(f"\n{'='*70}")
        print("COMPUTING t-SNE FOR COMBINED DATA")
        print(f"{'='*70}")
        combined_embedding, combined_keys, _, _ = compute_tsne(
            combined_data, perplexity, TSNE_DIMENSIONS, RANDOM_SEED
        )
        
        # Plot combined
        if TSNE_DIMENSIONS == 2:
            plot_comparison_combined(
                combined_embedding, combined_data, combined_keys, str(output_path)
            )
        else:
            print("⚠️  3D combined comparison not yet implemented")
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
