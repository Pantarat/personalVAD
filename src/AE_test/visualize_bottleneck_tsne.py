#!/usr/bin/env python3
"""
Visualize 64-dim Bottleneck Representations with t-SNE
Compare single-speaker vs overlap samples using autoencoder bottleneck features

This script:
1. Extracts 256-dim d-vectors from single-speaker and overlap samples
2. Passes them through trained autoencoder encoder to get 64-dim bottleneck
3. Visualizes 64-dim representations with t-SNE
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import librosa
import json
import glob
import os
from pathlib import Path
from collections import defaultdict

# Import shared utilities
from dvector_utils import initialize_encoder
from autoencoder_utils import load_autoencoder, extract_bottleneck_features
from tsne_utils import compute_tsne

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
LIBRISPEECH_PATH = '../../data/LibriSpeech'
OVERLAP_SAMPLES_DIR = './test_outputs/8288+30TestSpk_300'
AUTOENCODER_MODEL_PATH = './test_outputs/dvector_ae_84_w_other_singles/best_model.pth'
AUTOENCODER_CONFIG_PATH = './test_outputs/dvector_ae_84_w_other_singles/config.pkl'
OUTPUT_PATH = './test_outputs/8288Test_overlap_ae_w_other_singles.png'

# LibriSpeech splits to search (in priority order)
LIBRISPEECH_SPLITS = ['dev-clean', 'dev-other', 'test-clean', 'train-clean-100', 'train-clean-360']

# Main speaker
MAIN_SPEAKER = '8288'

# Other speakers
# OTHER_SPEAKERS = ['174', '251', '422', '652', '777'] #Train
OTHER_SPEAKERS = ['61', '121', '237', '260', '672'] #Test

# Number of samples
N_SINGLE_SAMPLES_PER_SPEAKER = 10
N_OVERLAP_SAMPLES = 100

# Arbitrary speaker IDs (optional)
# Set to None or empty list to disable, or provide a list of speaker IDs to extract from LibriSpeech
# Example: ['1234', '5678', '9012']
# These will be extracted from LibriSpeech but plotted separately from main/other speakers
ARBITRARY_SPEAKERS = None  # List of speaker IDs or None
N_ARBITRARY_SAMPLES_PER_SPEAKER = 10  # Samples to extract per arbitrary speaker

# t-SNE parameters
TSNE_DIMENSIONS = 2
TSNE_PERPLEXITY = 30
RANDOM_SEED = 42

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                audio_files = split_files
                print(f"    Found {len(audio_files)} audio files in {split}")
                break
        
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
                
                wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
                if device == 'cuda' and torch.cuda.is_available():
                    wav_tensor = wav_tensor.cuda()
                
                with torch.no_grad():
                    dvector = encoder.embed_utterance(wav_tensor.squeeze().cpu().numpy())
                
                filename = os.path.basename(audio_file)
                key = f"single_{speaker_id}_{filename}"
                dvectors[key] = {
                    'dvector': dvector,
                    'type': 'single',
                    'speaker': speaker_id
                }
                
            except Exception as e:
                print(f"      ⚠️  Error processing {os.path.basename(audio_file)}: {e}")
                if len(dvectors) == 0:  # Print traceback for first error only
                    import traceback
                    traceback.print_exc()
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


def extract_overlap_dvectors(overlap_dir, metadata_list, n_samples, other_speakers, 
                            sample_rate=16000, device='cuda', random_seed=42):
    """Extract d-vectors from overlap samples (filtered by other_speakers list)"""
    print(f"\n🎤 Extracting d-vectors from overlap samples...")
    print(f"  Target samples: {n_samples}")
    print(f"  Filtering for speakers: {', '.join(other_speakers)}")
    
    encoder, device_obj = initialize_encoder(device)
    
    # Filter metadata to only include samples with other_speakers
    filtered_metadata = []
    for metadata in metadata_list:
        other_spks = metadata.get('other_speakers', [])
        # Check if any of the other speakers in this sample are in our OTHER_SPEAKERS list
        if any(spk in other_speakers for spk in other_spks):
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
            # Add .wav extension if not present
            if not output_name.endswith('.wav'):
                output_name = output_name + '.wav'
            audio_path = audio_dir / output_name
            
            if not audio_path.exists():
                print(f"    [{i+1}/{len(metadata_list)}] ⚠️  Audio not found: {audio_path}")
                continue
            
            wav, sr = librosa.load(str(audio_path), sr=sample_rate)
            
            if len(wav) < sample_rate * 0.5:
                continue
            
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
            if device == 'cuda' and torch.cuda.is_available():
                wav_tensor = wav_tensor.cuda()
            
            with torch.no_grad():
                dvector = encoder.embed_utterance(wav_tensor.squeeze().cpu().numpy())
            
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
                print(f"    Progress: {i + 1}/{len(metadata_list)} ({len(dvectors)} extracted)")
            
        except Exception as e:
            print(f"    ⚠️  Error processing sample {metadata.get('sample_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✓ Total overlap d-vectors: {len(dvectors)}")
    return dvectors


def compute_separation_metrics(bottleneck_matrix, labels):
    """Compute separation metrics between single and overlap samples"""
    print(f"\n📊 Computing separation metrics...")
    
    single_mask = np.array([label == 'single' for label in labels])
    overlap_mask = np.array([label == 'overlap' for label in labels])
    
    single_bottlenecks = bottleneck_matrix[single_mask]
    overlap_bottlenecks = bottleneck_matrix[overlap_mask]
    
    # Within-group distances
    single_dists = []
    for i in range(len(single_bottlenecks)):
        for j in range(i + 1, len(single_bottlenecks)):
            dist = np.linalg.norm(single_bottlenecks[i] - single_bottlenecks[j])
            single_dists.append(dist)
    
    overlap_dists = []
    for i in range(len(overlap_bottlenecks)):
        for j in range(i + 1, len(overlap_bottlenecks)):
            dist = np.linalg.norm(overlap_bottlenecks[i] - overlap_bottlenecks[j])
            overlap_dists.append(dist)
    
    # Between-group distances
    between_dists = []
    for single_vec in single_bottlenecks:
        for overlap_vec in overlap_bottlenecks:
            dist = np.linalg.norm(single_vec - overlap_vec)
            between_dists.append(dist)
    
    metrics = {
        'within_single_mean': np.mean(single_dists) if single_dists else 0,
        'within_single_std': np.std(single_dists) if single_dists else 0,
        'within_overlap_mean': np.mean(overlap_dists) if overlap_dists else 0,
        'within_overlap_std': np.std(overlap_dists) if overlap_dists else 0,
        'between_mean': np.mean(between_dists) if between_dists else 0,
        'between_std': np.std(between_dists) if between_dists else 0
    }
    
    # Separation score: ratio of between-group to within-group distance
    within_mean = (metrics['within_single_mean'] + metrics['within_overlap_mean']) / 2
    if within_mean > 0:
        metrics['separation_score'] = metrics['between_mean'] / within_mean
    else:
        metrics['separation_score'] = 0
    
    print(f"  Within-single distance: {metrics['within_single_mean']:.4f} ± {metrics['within_single_std']:.4f}")
    print(f"  Within-overlap distance: {metrics['within_overlap_mean']:.4f} ± {metrics['within_overlap_std']:.4f}")
    print(f"  Between-group distance: {metrics['between_mean']:.4f} ± {metrics['between_std']:.4f}")
    print(f"  Separation score: {metrics['separation_score']:.4f}")
    
    return metrics


def plot_bottleneck_tsne_2d(embedding, data_dict, keys, main_speaker, output_path=None):
    """Create 2D t-SNE plot of 64-dim bottleneck features"""
    print("\nCreating 2D bottleneck t-SNE plot...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Separate by type
    single_mask = np.array([data_dict[k]['type'] == 'single' for k in keys])
    overlap_mask = np.array([data_dict[k]['type'] == 'overlap' for k in keys])
    arbitrary_mask = np.array([data_dict[k]['type'] == 'arbitrary' for k in keys])
    
    # Get speakers
    speakers = [data_dict[k]['speaker'] for k in keys]
    unique_speakers = sorted(set(speakers))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    speaker_to_color = {spk: colors[i] for i, spk in enumerate(unique_speakers)}
    
    # Plot single-speaker samples (main speaker as star, others as circles)
    for speaker in unique_speakers:
        mask = single_mask & np.array([data_dict[k]['speaker'] == speaker for k in keys])
        if mask.sum() > 0:
            marker = '*' if speaker == main_speaker else 'o'
            size = 150 if speaker == main_speaker else 80
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[speaker_to_color[speaker]],
                label=f'Single Speaker {speaker} (n={mask.sum()})',
                alpha=0.7,
                s=size,
                marker=marker,
                edgecolors='black',
                linewidth=0.5
            )
    
    # Plot overlap samples (squares)
    for speaker in unique_speakers:
        mask = overlap_mask & np.array([data_dict[k].get('other_speaker', '') == speaker for k in keys])
        if mask.sum() > 0:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[speaker_to_color[speaker]],
                label=f'Overlap (Main + {speaker}) (n={mask.sum()})',
                alpha=0.7,
                s=120,
                marker='s',
                edgecolors='black',
                linewidth=0.5
            )
    
    # Plot arbitrary speakers (diamonds)
    for speaker in unique_speakers:
        mask = arbitrary_mask & np.array([data_dict[k]['speaker'] == speaker for k in keys])
        if mask.sum() > 0:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[speaker_to_color[speaker]],
                label=f'Arbitrary Speaker {speaker} (n={mask.sum()})',
                alpha=0.7,
                s=100,
                marker='D',
                edgecolors='black',
                linewidth=0.5
            )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    title = f'64-dim Bottleneck t-SNE: Single-Speaker vs Overlap (Main: {main_speaker})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Info box
    n_single = single_mask.sum()
    n_overlap = overlap_mask.sum()
    n_arbitrary = arbitrary_mask.sum()
    info_text = f'○ Single-speaker: {n_single}\n□ Overlap (Main {main_speaker} + Other): {n_overlap}'
    if n_arbitrary > 0:
        info_text += f'\n◇ Arbitrary speakers: {n_arbitrary}'
    ax.text(0.02, 0.98, 
            info_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")
    
    plt.show()
    return fig


def plot_bottleneck_tsne_3d(embedding, data_dict, keys, main_speaker, output_path=None):
    """Create 3D t-SNE plot of 64-dim bottleneck features"""
    print("\nCreating 3D bottleneck t-SNE plot...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    single_mask = np.array([data_dict[k]['type'] == 'single' for k in keys])
    overlap_mask = np.array([data_dict[k]['type'] == 'overlap' for k in keys])
    arbitrary_mask = np.array([data_dict[k]['type'] == 'arbitrary' for k in keys])
    
    speakers = [data_dict[k]['speaker'] for k in keys]
    unique_speakers = sorted(set(speakers))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    speaker_to_color = {spk: colors[i] for i, spk in enumerate(unique_speakers)}
    
    # Plot single-speaker (main speaker as star, others as circles)
    for speaker in unique_speakers:
        mask = single_mask & np.array([data_dict[k]['speaker'] == speaker for k in keys])
        if mask.sum() > 0:
            marker = '*' if speaker == main_speaker else 'o'
            size = 150 if speaker == main_speaker else 80
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=[speaker_to_color[speaker]],
                label=f'Single Speaker {speaker} (n={mask.sum()})',
                alpha=0.7,
                s=size,
                marker=marker,
                edgecolors='black',
                linewidth=0.5
            )
    
    # Plot overlap
    for speaker in unique_speakers:
        mask = overlap_mask & np.array([data_dict[k].get('other_speaker', '') == speaker for k in keys])
        if mask.sum() > 0:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=[speaker_to_color[speaker]],
                label=f'Overlap (Main + {speaker}) (n={mask.sum()})',
                alpha=0.7,
                s=120,
                marker='s',
                edgecolors='black',
                linewidth=0.5
            )
    
    # Plot arbitrary speakers (diamonds)
    for speaker in unique_speakers:
        mask = arbitrary_mask & np.array([data_dict[k]['speaker'] == speaker for k in keys])
        if mask.sum() > 0:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=[speaker_to_color[speaker]],
                label=f'Arbitrary Speaker {speaker} (n={mask.sum()})',
                alpha=0.7,
                s=100,
                marker='D',
                edgecolors='black',
                linewidth=0.5
            )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_zlabel('t-SNE Dimension 3', fontsize=11)
    
    title = f'64-dim Bottleneck t-SNE: Single-Speaker vs Overlap (Main: {main_speaker})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved 3D plot to: {output_path}")
    
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
    print("64-DIM BOTTLENECK t-SNE: SINGLE-SPEAKER vs OVERLAP")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  Main speaker: {MAIN_SPEAKER}")
    print(f"  Other speakers: {', '.join(OTHER_SPEAKERS)}")
    print(f"  Single samples per speaker: {N_SINGLE_SAMPLES_PER_SPEAKER}")
    print(f"  Overlap samples: {N_OVERLAP_SAMPLES}")
    print(f"  Autoencoder model: {AUTOENCODER_MODEL_PATH}")
    print(f"  t-SNE dimensions: {TSNE_DIMENSIONS}D")
    print(f"  Output: {output_path}")
    
    # Extract single-speaker d-vectors
    all_speakers = [MAIN_SPEAKER] + OTHER_SPEAKERS
    single_dvectors = extract_single_speaker_dvectors(
        librispeech_path, all_speakers, N_SINGLE_SAMPLES_PER_SPEAKER,
        SAMPLE_RATE, DEVICE, RANDOM_SEED, LIBRISPEECH_SPLITS
    )
    
    # Load overlap metadata
    try:
        metadata_list = load_metadata(overlap_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Extract overlap d-vectors
    overlap_dvectors = extract_overlap_dvectors(
        overlap_dir, metadata_list, N_OVERLAP_SAMPLES, OTHER_SPEAKERS,
        SAMPLE_RATE, DEVICE, RANDOM_SEED
    )
    
    # Extract arbitrary speaker d-vectors if configured
    arbitrary_dvectors = {}
    if ARBITRARY_SPEAKERS:
        print(f"\n{'='*70}")
        print("EXTRACTING ARBITRARY SPEAKER D-VECTORS")
        print(f"{'='*70}")
        arbitrary_dvectors_raw = extract_single_speaker_dvectors(
            librispeech_path, ARBITRARY_SPEAKERS, N_ARBITRARY_SAMPLES_PER_SPEAKER,
            SAMPLE_RATE, DEVICE, RANDOM_SEED, LIBRISPEECH_SPLITS
        )
        # Change type from 'single' to 'arbitrary' to distinguish in plots
        for key, data in arbitrary_dvectors_raw.items():
            arbitrary_dvectors[key.replace('single_', 'arbitrary_')] = {
                'dvector': data['dvector'],
                'type': 'arbitrary',
                'speaker': data['speaker']
            }
    
    # Combine
    combined_data = {}
    combined_data.update(single_dvectors)
    combined_data.update(overlap_dvectors)
    combined_data.update(arbitrary_dvectors)
    
    if len(combined_data) < 2:
        print(f"\n❌ Error: Need at least 2 samples, found {len(combined_data)}")
        return
    
    # Load autoencoder and extract bottleneck features (64-dim)
    try:
        model_path = script_dir / AUTOENCODER_MODEL_PATH
        config_path = script_dir / AUTOENCODER_CONFIG_PATH
        
        if not model_path.exists():
            print(f"\n❌ Error: Autoencoder model not found: {model_path}")
            return
        if not config_path.exists():
            print(f"\n❌ Error: Autoencoder config not found: {config_path}")
            return
        
        print(f"\n{'='*70}")
        print("EXTRACTING 64-DIM BOTTLENECK FEATURES")
        print(f"{'='*70}")
        
        autoencoder = load_autoencoder(str(model_path), str(config_path), DEVICE)
        combined_data = extract_bottleneck_features(combined_data, autoencoder, DEVICE)
        
    except Exception as e:
        print(f"\n❌ Error loading/applying autoencoder: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n📊 Combined bottleneck data:")
    print(f"  Single-speaker: {len(single_dvectors)}")
    print(f"  Overlap: {len(overlap_dvectors)}")
    if arbitrary_dvectors:
        print(f"  Arbitrary speakers: {len(arbitrary_dvectors)}")
    print(f"  Total: {len(combined_data)}")
    
    # Adjust perplexity
    n_samples = len(combined_data)
    perplexity = min(TSNE_PERPLEXITY, (n_samples - 1) // 3)
    if perplexity < 5:
        perplexity = 5
    
    if perplexity != TSNE_PERPLEXITY:
        print(f"\n⚠️  Adjusted perplexity: {TSNE_PERPLEXITY} → {perplexity}")
    
    # Compute t-SNE on 64-dim bottleneck
    print(f"\n{'='*70}")
    print("COMPUTING t-SNE ON 64-DIM BOTTLENECK")
    print(f"{'='*70}")
    
    embedding, keys, types, speakers = compute_tsne(
        combined_data, perplexity, TSNE_DIMENSIONS, RANDOM_SEED
    )
    
    # Plot
    if TSNE_DIMENSIONS == 2:
        plot_bottleneck_tsne_2d(embedding, combined_data, keys, MAIN_SPEAKER, str(output_path))
    else:
        plot_bottleneck_tsne_3d(embedding, combined_data, keys, MAIN_SPEAKER, str(output_path))
    
    print("\n" + "=" * 70)
    print("✅ BOTTLENECK VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Plot saved to: {output_path}")
    print(f"\n💡 This visualization uses 64-dim bottleneck features")
    print(f"   Compare with full 256-dim d-vector visualization to see")
    print(f"   if compression helps distinguish single-speaker from overlap")


if __name__ == "__main__":
    main()
