#!/usr/bin/env python3
"""
Shared t-SNE visualization utilities
Reduces code duplication across test scripts
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D


def compute_tsne(combined_data, perplexity=30, n_components=2, random_state=42):
    """Compute t-SNE on combined data"""
    print(f"\n📊 Computing t-SNE...")
    
    keys = list(combined_data.keys())
    dvector_matrix = np.array([combined_data[k]['dvector'] for k in keys])
    types = [combined_data[k]['type'] for k in keys]
    speakers = [combined_data[k]['speaker'] for k in keys]
    
    print(f"  Input shape: {dvector_matrix.shape}")
    print(f"  Clean samples: {types.count('clean')}")
    print(f"  Overlap samples: {types.count('overlap') + types.count('overlap_extracted')}")
    if 'predicted' in types:
        print(f"  Predicted samples: {types.count('predicted')}")
    # Filter out None values before sorting
    unique_speakers = set(speakers)
    unique_speakers.discard(None)
    print(f"  Unique speakers: {sorted(unique_speakers)}")
    
    # Adjust perplexity
    n_samples = len(keys)
    perplexity = min(perplexity, (n_samples - 1) // 3)
    if perplexity < 5:
        perplexity = 5
    
    print(f"  Perplexity: {perplexity}")
    
    # PCA preprocessing
    max_pca_components = min(50, dvector_matrix.shape[0] - 1, dvector_matrix.shape[1])
    
    if dvector_matrix.shape[1] > max_pca_components and max_pca_components >= n_components:
        print(f"  Applying PCA: {dvector_matrix.shape[1]} → {max_pca_components} dims")
        pca = PCA(n_components=max_pca_components, random_state=random_state)
        dvector_matrix = pca.fit_transform(dvector_matrix)
        print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=1
    )
    
    embedding = tsne.fit_transform(dvector_matrix)
    
    print(f"✓ t-SNE complete! Output shape: {embedding.shape}")
    
    return embedding, keys, types, speakers


def create_figure(is_3d):
    """Create matplotlib figure for 2D or 3D plot"""
    if is_3d:
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(16, 12))
    
    return fig, ax


def get_marker_config():
    """Get marker configuration for different sample types"""
    type_markers = {
        'clean': 'o',              # Circle for clean
        'overlap_extracted': 'D',  # Diamond for extracted from overlap audio
        'overlap': 's',            # Square for overlap (training pairs)
        'predicted': '^'           # Triangle for predicted
    }
    
    type_sizes = {
        'clean': 120,
        'overlap_extracted': 110,
        'overlap': 100,
        'predicted': 120
    }
    
    type_alphas = {
        'clean': 0.9,
        'overlap_extracted': 0.7,
        'overlap': 0.6,
        'predicted': 0.8
    }
    
    return type_markers, type_sizes, type_alphas


def plot_scatter(ax, embedding, mask, speaker_color, type_name, type_markers, 
                type_sizes, type_alphas, label, is_3d):
    """Plot scatter points for a specific type and speaker"""
    if is_3d:
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=[speaker_color],
            marker=type_markers[type_name],
            s=type_sizes[type_name],
            alpha=type_alphas[type_name],
            label=label,
            edgecolors='black',
            linewidth=1.5 if type_name == 'predicted' else 1.0
        )
    else:
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[speaker_color],
            marker=type_markers[type_name],
            s=type_sizes[type_name],
            alpha=type_alphas[type_name],
            label=label,
            edgecolors='black',
            linewidth=1.5 if type_name == 'predicted' else 1.0
        )


def create_legend_elements(type_markers_dict, unique_speakers, speaker_to_color, 
                          main_speaker, include_predicted=True):
    """Create legend elements for the plot"""
    legend_elements = []
    
    # Add marker types
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                 markersize=10, label='○ Clean speaker', markeredgecolor='black'))
    
    if 'overlap_extracted' in type_markers_dict:
        legend_elements.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
                                     markersize=9, label='◇ Extracted from overlap audio', markeredgecolor='black'))
    
    if 'overlap' in type_markers_dict:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                                     markersize=9, label='□ Overlap pairs (training)', markeredgecolor='black'))
    
    if include_predicted and 'predicted' in type_markers_dict:
        legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                                     markersize=10, label='△ Predicted main (by model)', markeredgecolor='black'))
    
    # Add speaker colors
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                                 markersize=0, label=''))  # Spacer
    
    for speaker in unique_speakers:
        if speaker == main_speaker:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=speaker_to_color[speaker], 
                                        markersize=10, label=f'Main speaker {speaker}', 
                                        markeredgecolor='black'))
        else:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=speaker_to_color[speaker], 
                                        markersize=10, label=f'Other speaker {speaker}', 
                                        markeredgecolor='black'))
    
    return legend_elements


def add_info_box(ax, types, is_3d, info_text):
    """Add information box to the plot"""
    if not is_3d:
        ax.text(0.98, 0.02, info_text,
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
    else:
        # For 3D, add text in a different way
        ax.text2D(0.98, 0.02, info_text,
                  transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                  horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))
