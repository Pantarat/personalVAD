"""
AE_test Package
Autoencoder testing and training utilities for PersonalVAD
"""

from .dvector_utils import (
    DvectorAutoencoder,
    load_trained_model,
    initialize_encoder,
    extract_dvector_from_audio,
    extract_clean_speaker_dvectors,
    extract_overlap_dvectors,
    encode_to_bottleneck,
    predict_clean_dvectors
)

from .tsne_utils import (
    compute_tsne,
    create_figure,
    get_marker_config,
    plot_scatter,
    create_legend_elements,
    add_info_box
)

__all__ = [
    # D-vector utilities
    'DvectorAutoencoder',
    'load_trained_model',
    'initialize_encoder',
    'extract_dvector_from_audio',
    'extract_clean_speaker_dvectors',
    'extract_overlap_dvectors',
    'encode_to_bottleneck',
    'predict_clean_dvectors',
    
    # t-SNE utilities
    'compute_tsne',
    'create_figure',
    'get_marker_config',
    'plot_scatter',
    'create_legend_elements',
    'add_info_box'
]
