#!/usr/bin/env python3
"""
Test Autoencoder: Compare Clean, Overlap, and Predicted on t-SNE (Refactored)
Uses shared utilities from dvector_utils and tsne_utils
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Import shared utilities
from dvector_utils import (
    load_trained_model,
    extract_clean_speaker_dvectors,
    extract_overlap_dvectors,
    predict_clean_dvectors
)
from tsne_utils import (
    compute_tsne,
    create_figure,
    get_marker_config,
    plot_scatter,
    create_legend_elements,
    add_info_box
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
LIBRISPEECH_PATH = '../../data/LibriSpeech'
OVERLAP_SAMPLES_DIR = './test_outputs/full_overlap_samples'
MODEL_DIR = './test_outputs/dvector_autoencoder'
OUTPUT_PATH = './test_outputs/autoencoder_comparison_tsne.png'

# Main speaker
MAIN_SPEAKER = '84'

# Other speakers
OTHER_SPEAKERS = ['174', '251', '422', '652']

# Number of samples to test
N_TEST_SAMPLES_PER_SPEAKER = 50
N_CLEAN_SAMPLES_PER_SPEAKER = 50

# t-SNE parameters
TSNE_PERPLEXITY = 40
TSNE_DIMENSIONS = 2
RANDOM_SEED = 42

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Audio settings
SAMPLE_RATE = 16000

# ============================================================================


def compute_reconstruction_metrics(clean_dvectors, overlap_dvectors, predicted_dvectors, main_speaker):
    """Compute metrics comparing clean, overlap, and predicted"""
    print(f"\n📊 Computing reconstruction metrics...")
    
    overlap_keys = list(overlap_dvectors.keys())
    predicted_keys = list(predicted_dvectors.keys())
    clean_values = list(clean_dvectors.values())
    
    overlap_to_clean = []
    predicted_to_clean = []
    improvements = []
    
    for i, (overlap_key, pred_key) in enumerate(zip(overlap_keys, predicted_keys)):
        if i >= len(clean_values):
            break
        
        overlap_dvec = overlap_dvectors[overlap_key]['dvector']
        predicted_dvec = predicted_dvectors[pred_key]['dvector']
        clean_dvec = clean_values[i % len(clean_values)]['dvector']
        
        # Normalize
        overlap_norm = overlap_dvec / np.linalg.norm(overlap_dvec)
        predicted_norm = predicted_dvec / np.linalg.norm(predicted_dvec)
        clean_norm = clean_dvec / np.linalg.norm(clean_dvec)
        
        # Cosine similarities
        sim_overlap = np.dot(overlap_norm, clean_norm)
        sim_predicted = np.dot(predicted_norm, clean_norm)
        
        overlap_to_clean.append(sim_overlap)
        predicted_to_clean.append(sim_predicted)
        improvements.append(sim_predicted - sim_overlap)
    
    overlap_to_clean = np.array(overlap_to_clean)
    predicted_to_clean = np.array(predicted_to_clean)
    improvements = np.array(improvements)
    
    print(f"\n  Cosine Similarity to Clean Main Speaker:")
    print(f"    Overlap → Clean:    {overlap_to_clean.mean():.4f} ± {overlap_to_clean.std():.4f}")
    print(f"    Predicted → Clean:  {predicted_to_clean.mean():.4f} ± {predicted_to_clean.std():.4f}")
    print(f"\n  Improvement:")
    print(f"    Average: {improvements.mean():.4f} ± {improvements.std():.4f}")
    print(f"    Median:  {np.median(improvements):.4f}")
    
    improved_count = (improvements > 0).sum()
    print(f"\n  Samples improved: {improved_count}/{len(improvements)} ({100*improved_count/len(improvements):.1f}%)")
    
    return {
        'overlap_to_clean': overlap_to_clean,
        'predicted_to_clean': predicted_to_clean,
        'improvements': improvements
    }


def plot_comparison_tsne(embedding, types, speakers, main_speaker, output_path):
    """Plot t-SNE with clean, overlap, and predicted"""
    print(f"\n📈 Creating comparison t-SNE plot...")
    
    is_3d = embedding.shape[1] == 3
    fig, ax = create_figure(is_3d)
    
    type_markers, type_sizes, type_alphas = get_marker_config()
    
    # Colors for speakers
    unique_speakers = sorted(set(speakers))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    speaker_to_color = {spk: colors[i] for i, spk in enumerate(unique_speakers)}
    
    # Plot each combination
    plot_types = ['clean', 'overlap_extracted', 'overlap', 'predicted']
    for type_name in plot_types:
        if type_name not in type_markers:
            continue
            
        for speaker in unique_speakers:
            mask = np.array([(t == type_name and s == speaker) 
                           for t, s in zip(types, speakers)])
            
            if mask.sum() == 0:
                continue
            
            # Label format
            if type_name == 'clean':
                if speaker == main_speaker:
                    label = f'Clean Main {speaker}'
                else:
                    label = f'Clean Other {speaker}'
            elif type_name == 'overlap_extracted':
                label = f'Overlap Extracted (Main + {speaker})'
            elif type_name == 'overlap':
                label = f'Overlap Pairs (Main + {speaker})'
            else:  # predicted
                label = f'Predicted Main (from {speaker})'
            
            plot_scatter(ax, embedding, mask, speaker_to_color[speaker], 
                        type_name, type_markers, type_sizes, type_alphas, label, is_3d)
    
    # Labels and title
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    if is_3d:
        ax.set_zlabel('t-SNE Dimension 3', fontsize=14, fontweight='bold')
        ax.set_title(f'3D Autoencoder Comparison: Clean (All) vs Overlap vs Predicted\nMain Speaker: {main_speaker}', 
                     fontsize=16, fontweight='bold', pad=20)
    else:
        ax.set_title(f'Autoencoder Comparison: Clean (All) vs Overlap vs Predicted\nMain Speaker: {main_speaker}', 
                     fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = create_legend_elements(type_markers, unique_speakers, 
                                            speaker_to_color, main_speaker, include_predicted=True)
    
    if is_3d:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95, edgecolor='black')
    else:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95, edgecolor='black')
    
    # Info box
    n_clean = types.count('clean')
    n_overlap = types.count('overlap') + types.count('overlap_extracted')
    n_predicted = types.count('predicted')
    
    info_text = (
        f'Sample counts:\n'
        f'  Clean: {n_clean}\n'
        f'  Overlap: {n_overlap}\n'
        f'  Predicted: {n_predicted}\n\n'
        f'Interpretation:\n'
        f'  If △ (predicted) clusters\n'
        f'  near ○ (clean) → Good!\n'
        f'  If △ stays near □/◇ (overlap)\n'
        f'  → Model needs improvement'
    )
    
    add_info_box(ax, types, is_3d, info_text)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    
    if is_3d:
        print(f"  💡 Tip: The plot is interactive - you can rotate it with your mouse!")
    
    plt.show()
    
    return fig


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    script_dir = Path(__file__).parent
    librispeech_path = script_dir / LIBRISPEECH_PATH
    overlap_dir = script_dir / OVERLAP_SAMPLES_DIR
    model_dir = script_dir / MODEL_DIR
    output_path = script_dir / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AUTOENCODER TEST: CLEAN vs OVERLAP vs PREDICTED")
    print("=" * 70)
    
    # Load model
    model, config = load_trained_model(model_dir, DEVICE)
    
    # Extract clean d-vectors for all speakers
    all_speakers = [MAIN_SPEAKER] + OTHER_SPEAKERS
    clean_dvectors = extract_clean_speaker_dvectors(
        librispeech_path, all_speakers, N_CLEAN_SAMPLES_PER_SPEAKER,
        SAMPLE_RATE, DEVICE, RANDOM_SEED
    )
    
    # Extract overlap d-vectors (extracted and pairs)
    overlap_extracted_dvectors = extract_overlap_dvectors(
        overlap_dir, N_TEST_SAMPLES_PER_SPEAKER, SAMPLE_RATE, DEVICE, 'overlap_extracted'
    )
    
    overlap_dvectors = extract_overlap_dvectors(
        overlap_dir, N_TEST_SAMPLES_PER_SPEAKER, SAMPLE_RATE, DEVICE, 'overlap'
    )
    
    # Predict clean from overlap
    predicted_dvectors = predict_clean_dvectors(model, overlap_dvectors, DEVICE)
    
    # Compute metrics
    clean_main_only = {k: v for k, v in clean_dvectors.items() if v['speaker'] == MAIN_SPEAKER}
    metrics = compute_reconstruction_metrics(clean_main_only, overlap_dvectors, predicted_dvectors, MAIN_SPEAKER)
    
    # Combine all data
    combined_data = {}
    combined_data.update(clean_dvectors)
    combined_data.update(overlap_extracted_dvectors)
    combined_data.update(overlap_dvectors)
    combined_data.update(predicted_dvectors)
    
    # Compute t-SNE
    embedding, keys, types, speakers = compute_tsne(
        combined_data, TSNE_PERPLEXITY, TSNE_DIMENSIONS, RANDOM_SEED
    )
    
    # Plot
    plot_comparison_tsne(embedding, types, speakers, MAIN_SPEAKER, output_path)
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
