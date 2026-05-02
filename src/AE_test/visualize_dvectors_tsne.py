#!/usr/bin/env python3
"""
D-Vector t-SNE Visualization (Refactored)
Analyze and visualize speaker embeddings using t-SNE dimensionality reduction

Uses shared utilities from dvector_utils and tsne_utils
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import glob
import os
from pathlib import Path
from collections import defaultdict

from dvector_utils import initialize_encoder
from tsne_utils import compute_tsne, create_figure

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to LibriSpeech directory
LIBRISPEECH_PATH = '../../data/LibriSpeech'

# Selected speakers
SELECTED_SPEAKERS = ['84', '174', '251', '422', '652']

# Number of utterances per speaker
N_UTTERANCES_PER_SPEAKER = 50

# t-SNE parameters
TSNE_DIMENSIONS = 2  # 2 or 3
TSNE_PERPLEXITY = 40
RANDOM_SEED = 42

# Output settings
OUTPUT_PATH = './test_outputs/dvector_tsne.png'

# Device
DEVICE = 'cuda'

# Audio settings
SAMPLE_RATE = 16000

# ============================================================================


def extract_dvectors_from_audio(librispeech_path, speaker_ids, n_utterances_per_speaker=20, 
                                sample_rate=16000, device='cuda', random_seed=42):
    """Extract d-vectors from LibriSpeech audio files"""
    print(f"\n🎤 Extracting d-vectors from LibriSpeech audio files...")
    print(f"  LibriSpeech path: {librispeech_path}")
    print(f"  Target speakers: {', '.join(speaker_ids)}")
    print(f"  Utterances per speaker: {n_utterances_per_speaker}")
    
    encoder, device_obj = initialize_encoder(device)
    
    dvectors = {}
    librispeech_path = Path(librispeech_path)
    
    for speaker_id in speaker_ids:
        print(f"\n  Processing Speaker {speaker_id}...")
        
        speaker_pattern = str(librispeech_path / "dev-clean" / speaker_id / "*" / "*.flac")
        audio_files = glob.glob(speaker_pattern)
        
        if not audio_files:
            print(f"    ⚠️  No audio files found for speaker {speaker_id}")
            continue
        
        print(f"    Found {len(audio_files)} audio files")
        
        n_samples = min(n_utterances_per_speaker, len(audio_files))
        sampled_files = np.random.RandomState(random_seed).choice(audio_files, size=n_samples, replace=False)
        
        for i, audio_file in enumerate(sampled_files):
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
                key = f"{speaker_id}-{filename}"
                dvectors[key] = {
                    'dvector': dvector,
                    'type': 'clean',
                    'speaker': speaker_id
                }
                
            except Exception as e:
                print(f"      ⚠️  Error processing {os.path.basename(audio_file)}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"    ✓ Extracted {len([k for k in dvectors.keys() if k.startswith(speaker_id)])} d-vectors")
    
    print(f"\n✓ Total d-vectors extracted: {len(dvectors)}")
    if len(dvectors) > 0:
        print(f"  D-vector dimension: {list(dvectors.values())[0]['dvector'].shape[0]}")
    
    return dvectors


def extract_speaker_ids(dvector_keys):
    """Extract unique speaker IDs from d-vector keys"""
    speaker_map = defaultdict(list)
    
    for key in dvector_keys:
        speaker_id = key.split('-')[0]
        speaker_map[speaker_id].append(key)
    
    return speaker_map


def plot_tsne_2d(embedding, keys, speaker_map, output_path=None, 
                title="D-Vector t-SNE Visualization"):
    """Create 2D t-SNE plot colored by speaker"""
    print("\nCreating 2D t-SNE plot...")
    
    speaker_ids = [key.split('-')[0] for key in keys]
    unique_speakers = sorted(set(speaker_ids))
    n_speakers = len(unique_speakers)
    
    print(f"  Found {n_speakers} unique speakers")
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_speakers)))
    if n_speakers > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_speakers))
    
    speaker_to_color = {spk: colors[i % len(colors)] for i, spk in enumerate(unique_speakers)}
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for speaker_id in unique_speakers:
        mask = np.array([spk == speaker_id for spk in speaker_ids])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[speaker_to_color[speaker_id]],
            label=f'Speaker {speaker_id} (n={mask.sum()})',
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if n_speakers <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        speaker_counts = {spk: len(speaker_map[spk]) for spk in unique_speakers}
        top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        handles, labels = ax.get_legend_handles_labels()
        top_indices = [unique_speakers.index(spk) for spk, _ in top_speakers]
        ax.legend(
            [handles[i] for i in top_indices],
            [labels[i] for i in top_indices],
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=8,
            title=f"Top 15 of {n_speakers} speakers"
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")
    
    plt.show()
    return fig


def plot_tsne_3d(embedding, keys, speaker_map, output_path=None,
                title="D-Vector t-SNE 3D Visualization"):
    """Create 3D t-SNE plot colored by speaker"""
    print("\nCreating 3D t-SNE plot...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    speaker_ids = [key.split('-')[0] for key in keys]
    unique_speakers = sorted(set(speaker_ids))
    n_speakers = len(unique_speakers)
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_speakers)))
    if n_speakers > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_speakers))
    
    speaker_to_color = {spk: colors[i % len(colors)] for i, spk in enumerate(unique_speakers)}
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for speaker_id in unique_speakers:
        mask = np.array([spk == speaker_id for spk in speaker_ids])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=[speaker_to_color[speaker_id]],
            label=f'Speaker {speaker_id} (n={mask.sum()})',
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_zlabel('t-SNE Dimension 3', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if n_speakers <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        speaker_counts = {spk: len(speaker_map[spk]) for spk in unique_speakers}
        top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        handles, labels = ax.get_legend_handles_labels()
        top_indices = [unique_speakers.index(spk) for spk, _ in top_speakers]
        ax.legend(
            [handles[i] for i in top_indices],
            [labels[i] for i in top_indices],
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=8,
            title=f"Top 15 of {n_speakers} speakers"
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved 3D plot to: {output_path}")
    
    plt.show()
    return fig


def compute_separation_metrics(dvector_matrix, speaker_ids):
    """Compute metrics to quantify speaker separation"""
    print("\nComputing speaker separation metrics...")
    
    unique_speakers = sorted(set(speaker_ids))
    
    intra_distances = []
    inter_distances = []
    
    for i, spk_i in enumerate(speaker_ids):
        for j in range(i + 1, len(speaker_ids)):
            spk_j = speaker_ids[j]
            distance = np.linalg.norm(dvector_matrix[i] - dvector_matrix[j])
            
            if spk_i == spk_j:
                intra_distances.append(distance)
            else:
                inter_distances.append(distance)
    
    metrics = {
        'intra_speaker_mean': np.mean(intra_distances) if intra_distances else 0,
        'intra_speaker_std': np.std(intra_distances) if intra_distances else 0,
        'inter_speaker_mean': np.mean(inter_distances) if inter_distances else 0,
        'inter_speaker_std': np.std(inter_distances) if inter_distances else 0
    }
    
    if metrics['intra_speaker_mean'] > 0:
        metrics['separation_ratio'] = metrics['inter_speaker_mean'] / metrics['intra_speaker_mean']
    else:
        metrics['separation_ratio'] = float('inf')
    
    print(f"\n📊 Speaker Separation Metrics:")
    print(f"  Intra-speaker distance: {metrics['intra_speaker_mean']:.4f} ± {metrics['intra_speaker_std']:.4f}")
    print(f"  Inter-speaker distance: {metrics['inter_speaker_mean']:.4f} ± {metrics['inter_speaker_std']:.4f}")
    print(f"  Separation ratio: {metrics['separation_ratio']:.2f} (higher is better)")
    
    return metrics


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    script_dir = Path(__file__).parent
    librispeech_path = script_dir / LIBRISPEECH_PATH
    output_path = script_dir / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("D-VECTOR t-SNE VISUALIZATION")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  LibriSpeech path: {librispeech_path}")
    print(f"  Selected speakers: {', '.join(SELECTED_SPEAKERS)}")
    print(f"  Utterances per speaker: {N_UTTERANCES_PER_SPEAKER}")
    print(f"  t-SNE dimensions: {TSNE_DIMENSIONS}D")
    print(f"  t-SNE perplexity: {TSNE_PERPLEXITY}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {output_path}")
    
    if not librispeech_path.exists():
        print(f"\n❌ Error: LibriSpeech path not found: {librispeech_path}")
        return
    
    # Extract d-vectors
    dvectors = extract_dvectors_from_audio(
        str(librispeech_path),
        SELECTED_SPEAKERS,
        n_utterances_per_speaker=N_UTTERANCES_PER_SPEAKER,
        sample_rate=SAMPLE_RATE,
        device=DEVICE,
        random_seed=RANDOM_SEED
    )
    
    if len(dvectors) < 2:
        print(f"\n❌ Error: Need at least 2 d-vectors, found {len(dvectors)}")
        return
    
    # Extract speaker info
    speaker_map = extract_speaker_ids(list(dvectors.keys()))
    print(f"\nFound {len(speaker_map)} unique speakers:")
    for spk, keys in sorted(speaker_map.items()):
        print(f"  Speaker {spk}: {len(keys)} utterances")
    
    # Adjust perplexity
    n_samples = len(dvectors)
    perplexity = min(TSNE_PERPLEXITY, (n_samples - 1) // 3)
    if perplexity < 5:
        perplexity = 5
    
    if perplexity != TSNE_PERPLEXITY:
        print(f"\n⚠️  Adjusted perplexity: {TSNE_PERPLEXITY} → {perplexity}")
    
    # Compute t-SNE
    embedding, keys, types, speakers = compute_tsne(
        dvectors, perplexity, TSNE_DIMENSIONS, RANDOM_SEED
    )
    
    # Get dvector matrix for metrics
    dvector_matrix = np.array([dvectors[k]['dvector'] for k in keys])
    speaker_ids = [key.split('-')[0] for key in keys]
    
    # Compute metrics
    metrics = compute_separation_metrics(dvector_matrix, speaker_ids)
    
    # Plot
    if TSNE_DIMENSIONS == 2:
        plot_tsne_2d(embedding, keys, speaker_map, str(output_path))
    else:
        plot_tsne_3d(embedding, keys, speaker_map, str(output_path))
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  • {len(speaker_map)} speakers with {n_samples} total utterances")
    print(f"  • Separation ratio: {metrics['separation_ratio']:.2f}")
    print(f"  • {'Good' if metrics['separation_ratio'] > 1.5 else 'Moderate' if metrics['separation_ratio'] > 1.0 else 'Poor'} speaker separation")
    print(f"\n📁 Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
