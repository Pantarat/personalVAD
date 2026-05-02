#!/usr/bin/env python3
"""
Overlap t-SNE Visualization (Refactored)
Analyze speaker embeddings from overlap samples grouped by non-target speaker

Uses shared utilities from dvector_utils and tsne_utils
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import json
from pathlib import Path
from collections import defaultdict

from dvector_utils import initialize_encoder
from tsne_utils import compute_tsne

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to overlap samples directory
OVERLAP_SAMPLES_DIR = './test_outputs/full_overlap_samples'

# Clean main speaker samples
INCLUDE_CLEAN_MAIN_SPEAKER = True  # Include clean main speaker utterances
LIBRISPEECH_PATH = '../../data/LibriSpeech/dev-clean'
N_CLEAN_UTTERANCES = 50  # Number of clean main speaker utterances to include (-1 = all)

# t-SNE parameters
TSNE_DIMENSIONS = 2
TSNE_PERPLEXITY = 40
RANDOM_SEED = 42

# Output settings
OUTPUT_PATH = './test_outputs/84_overlap.png'

# Device
DEVICE = 'cuda'

# Audio settings
SAMPLE_RATE = 16000

# ============================================================================


def load_metadata(overlap_dir):
    """Load metadata from overlap samples directory"""
    metadata_path = Path(overlap_dir) / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded metadata for {len(metadata)} samples")
    return metadata


def extract_dvectors_from_overlap_samples(overlap_dir, metadata_list, sample_rate=16000, 
                                         device='cuda'):
    """Extract d-vectors from overlap audio samples"""
    print(f"\n🎤 Extracting d-vectors from overlap samples...")
    print(f"  Overlap samples directory: {overlap_dir}")
    print(f"  Number of samples: {len(metadata_list)}")
    
    encoder, device_obj = initialize_encoder(device)
    
    results = {}
    audio_dir = Path(overlap_dir) / 'audio'
    
    for i, metadata in enumerate(metadata_list):
        try:
            output_name = metadata['output_name']
            # Add .wav extension if not present
            if not output_name.endswith('.wav'):
                output_name = output_name + '.wav'
            audio_path = audio_dir / output_name
            
            if not audio_path.exists():
                print(f"    ⚠️  Audio not found: {audio_path}")
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
            other_speakers = metadata['other_speakers']
            other_speaker = other_speakers[0] if len(other_speakers) == 1 else ','.join(other_speakers)
            
            results[sample_id] = {
                'dvector': dvector,
                'type': 'overlap',
                'speaker': other_speaker,
                'main_speaker': main_speaker,
                'other_speaker': other_speaker
            }
            
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{len(metadata_list)}")
                
        except Exception as e:
            print(f"    ⚠️  Error processing sample {metadata.get('sample_id', i)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Total d-vectors extracted: {len(results)}")
    if len(results) > 0:
        sample_dvector = list(results.values())[0]['dvector']
        print(f"  D-vector dimension: {sample_dvector.shape[0]}")
    
    return results


def extract_clean_main_speaker_dvectors(librispeech_path, main_speaker_id, 
                                          n_utterances=-1, sample_rate=16000, device='cuda'):
    """Extract d-vectors from clean main speaker utterances"""
    print(f"\n🎤 Extracting d-vectors from clean main speaker utterances...")
    print(f"  LibriSpeech path: {librispeech_path}")
    print(f"  Main speaker: {main_speaker_id}")
    
    encoder, device_obj = initialize_encoder(device)
    
    speaker_dir = Path(librispeech_path) / main_speaker_id
    
    if not speaker_dir.exists():
        print(f"  ⚠️  Speaker directory not found: {speaker_dir}")
        return {}
    
    results = {}
    audio_files = list(speaker_dir.rglob('*.flac'))
    
    if n_utterances > 0:
        audio_files = audio_files[:n_utterances]
    
    print(f"  Found {len(audio_files)} audio files")
    
    for i, audio_path in enumerate(audio_files):
        try:
            wav, sr = librosa.load(str(audio_path), sr=sample_rate)
            
            if len(wav) < sample_rate * 0.5:
                continue
            
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
            if device == 'cuda' and torch.cuda.is_available():
                wav_tensor = wav_tensor.cuda()
            
            with torch.no_grad():
                dvector = encoder.embed_utterance(wav_tensor.squeeze().cpu().numpy())
            
            sample_id = f"clean_{main_speaker_id}_{i:04d}"
            
            results[sample_id] = {
                'dvector': dvector,
                'type': 'clean',
                'speaker': main_speaker_id,
                'main_speaker': main_speaker_id,
                'other_speaker': None
            }
            
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{len(audio_files)}")
                
        except Exception as e:
            print(f"    ⚠️  Error processing {audio_path.name}: {e}")
            continue
    
    print(f"\n✓ Total clean main speaker d-vectors extracted: {len(results)}")
    return results


def plot_tsne_2d(embedding, other_speakers, main_speaker, output_path=None):
    """Create 2D t-SNE plot colored by non-target speaker"""
    print("\nCreating 2D t-SNE plot (colored by non-target speaker)...")
    
    # Separate clean and overlap samples
    clean_mask = np.array([spk is None for spk in other_speakers])
    overlap_mask = ~clean_mask
    
    unique_speakers = sorted(set([spk for spk in other_speakers if spk is not None]))
    n_speakers = len(unique_speakers)
    
    print(f"  Main speaker: {main_speaker}")
    print(f"  Clean samples: {clean_mask.sum()}")
    print(f"  Overlap samples: {overlap_mask.sum()}")
    print(f"  Non-target speakers: {n_speakers}")
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_speakers)))
    if n_speakers > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_speakers))
    
    speaker_to_color = {spk: colors[i % len(colors)] for i, spk in enumerate(unique_speakers)}
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot clean main speaker samples first (with distinct marker)
    if clean_mask.sum() > 0:
        ax.scatter(
            embedding[clean_mask, 0],
            embedding[clean_mask, 1],
            c='gold',
            label=f'Clean: Main Speaker {main_speaker} (n={clean_mask.sum()})',
            alpha=0.9,
            s=120,
            marker='o',
            edgecolors='black',
            linewidth=1.0,
            zorder=10
        )
    
    # Plot overlap samples
    for other_speaker in unique_speakers:
        mask = np.array([spk == other_speaker for spk in other_speakers])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[speaker_to_color[other_speaker]],
            label=f'Overlap: Main + Other {other_speaker} (n={mask.sum()})',
            alpha=0.7,
            s=80,
            marker='^',
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    title_text = f't-SNE: Main Speaker {main_speaker}'
    if clean_mask.sum() > 0:
        title_text += f' (Clean + Overlap)'
    else:
        title_text += f' (Overlap Only)'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    info_text = f'Main Speaker {main_speaker}'
    if clean_mask.sum() > 0:
        info_text += f'\nClean: {clean_mask.sum()} | Overlap: {overlap_mask.sum()}'
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")
    
    plt.show()
    return fig


def plot_tsne_3d(embedding, other_speakers, main_speaker, output_path=None):
    """Create 3D t-SNE plot colored by non-target speaker"""
    print("\nCreating 3D t-SNE plot (colored by non-target speaker)...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Separate clean and overlap samples
    clean_mask = np.array([spk is None for spk in other_speakers])
    overlap_mask = ~clean_mask
    
    unique_speakers = sorted(set([spk for spk in other_speakers if spk is not None]))
    n_speakers = len(unique_speakers)
    
    print(f"  Clean samples: {clean_mask.sum()}")
    print(f"  Overlap samples: {overlap_mask.sum()}")
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_speakers)))
    if n_speakers > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_speakers))
    
    speaker_to_color = {spk: colors[i % len(colors)] for i, spk in enumerate(unique_speakers)}
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot clean main speaker samples first
    if clean_mask.sum() > 0:
        ax.scatter(
            embedding[clean_mask, 0],
            embedding[clean_mask, 1],
            embedding[clean_mask, 2],
            c='gold',
            label=f'Clean: Main Speaker {main_speaker} (n={clean_mask.sum()})',
            alpha=0.9,
            s=120,
            marker='*',
            edgecolors='black',
            linewidth=1.0
        )
    
    # Plot overlap samples
    for other_speaker in unique_speakers:
        mask = np.array([spk == other_speaker for spk in other_speakers])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=[speaker_to_color[other_speaker]],
            label=f'Overlap: Main + Other {other_speaker} (n={mask.sum()})',
            alpha=0.7,
            s=80,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_zlabel('t-SNE Dimension 3', fontsize=11)
    
    title_text = f'3D t-SNE: Main Speaker {main_speaker}'
    if clean_mask.sum() > 0:
        title_text += f' (Clean + Overlap)'
    else:
        title_text += f' (Overlap Only)'
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved 3D plot to: {output_path}")
    
    plt.show()
    return fig


def compute_separation_metrics(dvector_matrix, other_speakers, main_speaker):
    """Compute metrics to quantify separation by non-target speaker"""
    print("\nComputing speaker separation metrics...")
    print(f"  Main speaker: {main_speaker} (constant)")
    print(f"  Grouping by: Non-target (other) speaker")
    
    intra_distances = []
    inter_distances = []
    
    for i, other_i in enumerate(other_speakers):
        for j in range(i + 1, len(other_speakers)):
            other_j = other_speakers[j]
            distance = np.linalg.norm(dvector_matrix[i] - dvector_matrix[j])
            
            if other_i == other_j:
                intra_distances.append(distance)
            else:
                inter_distances.append(distance)
    
    metrics = {
        'intra_group_mean': np.mean(intra_distances) if intra_distances else 0,
        'intra_group_std': np.std(intra_distances) if intra_distances else 0,
        'inter_group_mean': np.mean(inter_distances) if inter_distances else 0,
        'inter_group_std': np.std(inter_distances) if inter_distances else 0
    }
    
    if metrics['intra_group_mean'] > 0:
        metrics['separation_ratio'] = metrics['inter_group_mean'] / metrics['intra_group_mean']
    else:
        metrics['separation_ratio'] = float('inf')
    
    print(f"\n📊 Separation Metrics (by non-target speaker):")
    print(f"  Intra-group distance: {metrics['intra_group_mean']:.4f} ± {metrics['intra_group_std']:.4f}")
    print(f"  Inter-group distance: {metrics['inter_group_mean']:.4f} ± {metrics['inter_group_std']:.4f}")
    print(f"  Separation ratio: {metrics['separation_ratio']:.2f}")
    
    return metrics


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    script_dir = Path(__file__).parent
    overlap_dir = script_dir / OVERLAP_SAMPLES_DIR
    output_path = script_dir / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("OVERLAP SAMPLES t-SNE VISUALIZATION")
    print("(Grouped by Non-Target Speaker)")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  Overlap samples directory: {overlap_dir}")
    if INCLUDE_CLEAN_MAIN_SPEAKER:
        print(f"  Include clean main speaker: Yes")
        print(f"  LibriSpeech path: {script_dir / LIBRISPEECH_PATH}")
        print(f"  N clean utterances: {N_CLEAN_UTTERANCES if N_CLEAN_UTTERANCES > 0 else 'all'}")
    print(f"  t-SNE dimensions: {TSNE_DIMENSIONS}D")
    print(f"  t-SNE perplexity: {TSNE_PERPLEXITY}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {output_path}")
    
    if not overlap_dir.exists():
        print(f"\n❌ Error: Overlap samples directory not found: {overlap_dir}")
        print("\nPlease run generate_full_overlap_utterances.py first")
        return
    
    # Load metadata
    try:
        metadata_list = load_metadata(overlap_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    if len(metadata_list) == 0:
        print(f"\n❌ Error: No samples found")
        return
    
    main_speaker = metadata_list[0]['main_speaker']
    
    # Extract d-vectors from overlap samples
    dvector_data = extract_dvectors_from_overlap_samples(
        overlap_dir, metadata_list, SAMPLE_RATE, DEVICE
    )
    
    # Extract clean main speaker d-vectors if requested
    if INCLUDE_CLEAN_MAIN_SPEAKER:
        librispeech_path = script_dir / LIBRISPEECH_PATH
        clean_dvectors = extract_clean_main_speaker_dvectors(
            librispeech_path, main_speaker, N_CLEAN_UTTERANCES, SAMPLE_RATE, DEVICE
        )
        
        if len(clean_dvectors) > 0:
            print(f"\n✓ Merging {len(clean_dvectors)} clean main speaker samples...")
            dvector_data.update(clean_dvectors)
        else:
            print(f"\n⚠️  No clean main speaker samples found")
    
    if len(dvector_data) < 2:
        print(f"\n❌ Error: Need at least 2 d-vectors, found {len(dvector_data)}")
        return
    
    # Count samples per other speaker
    other_speaker_counts = defaultdict(int)
    for data in dvector_data.values():
        other_speaker_counts[data['other_speaker']] += 1
    
    print(f"\n  Samples breakdown:")
    # Print clean samples first (None key)
    if None in other_speaker_counts:
        print(f"    Clean main speaker: {other_speaker_counts[None]} samples")
    
    # Print overlap samples (non-None keys)
    print(f"  Overlap samples by non-target speaker:")
    for spk, count in sorted([(s, c) for s, c in other_speaker_counts.items() if s is not None]):
        print(f"    Speaker {spk}: {count} samples")
    
    # Adjust perplexity
    n_samples = len(dvector_data)
    perplexity = min(TSNE_PERPLEXITY, (n_samples - 1) // 3)
    if perplexity < 5:
        perplexity = 5
    
    if perplexity != TSNE_PERPLEXITY:
        print(f"\n⚠️  Adjusted perplexity: {TSNE_PERPLEXITY} → {perplexity}")
    
    # Compute t-SNE
    embedding, sample_ids, types, speakers = compute_tsne(
        dvector_data, perplexity, TSNE_DIMENSIONS, RANDOM_SEED
    )
    
    # Get other speakers list
    other_speakers_list = [dvector_data[sid]['other_speaker'] for sid in sample_ids]
    dvector_matrix = np.array([dvector_data[sid]['dvector'] for sid in sample_ids])
    
    # Compute metrics
    metrics = compute_separation_metrics(dvector_matrix, other_speakers_list, main_speaker)
    
    # Plot
    if TSNE_DIMENSIONS == 2:
        plot_tsne_2d(embedding, other_speakers_list, main_speaker, str(output_path))
    else:
        plot_tsne_3d(embedding, other_speakers_list, main_speaker, str(output_path))
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nKey findings:")
    n_clean = len([spk for spk in other_speakers_list if spk is None])
    n_overlap = len([spk for spk in other_speakers_list if spk is not None])
    print(f"  • Main speaker {main_speaker}:")
    if n_clean > 0:
        print(f"    - Clean samples: {n_clean}")
    print(f"    - Overlap samples: {n_overlap}")
    print(f"  • {len(other_speaker_counts)} non-target speakers")
    print(f"  • Separation ratio: {metrics['separation_ratio']:.2f}")
    print(f"\n📁 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
