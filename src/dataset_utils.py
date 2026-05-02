"""
Dataset utilities for PersonalVAD training.

Provides common functions for managing datasets, especially for handling
augmented data with balanced sampling.
"""

import numpy as np


def get_base_utt_id(utt_id):
    """
    Get base utterance ID by removing augmentation suffixes.
    
    Examples:
        84-121550-0000_OV_174-50561-0029-reverb -> 84-121550-0000_OV_174-50561-0029
        84-121550-0000_OV_174-50561-0029-babble -> 84-121550-0000_OV_174-50561-0029
        84-121550-0000_OV_174-50561-0029 -> 84-121550-0000_OV_174-50561-0029 (no change)
    """
    suffixes = ['-reverb', '-noise', '-music', '-babble']
    for suffix in suffixes:
        if utt_id.endswith(suffix):
            return utt_id[:-len(suffix)]
    return utt_id


def sample_evenly_across_augmentations(keys, max_utterances):
    """
    Sample utterances by limiting unique base utterances, keeping all augmentations.
    
    When limiting to N utterances, this selects N unique base utterances and includes
    ALL augmentations of those base utterances. This ensures:
    - The limit represents unique speech content (not total samples)
    - All augmentation variants of selected utterances are included
    - Better diversity in actual speech content
    
    Example:
        max_utterances=100, augmentation types=5 (original+reverb+noise+music+babble)
        Result: 100 unique base utterances × 5 augmentations = 500 total samples
    
    Args:
        keys: Array of utterance IDs (potentially including augmented versions)
        max_utterances: Maximum number of unique base utterances to keep
        
    Returns:
        Array of selected utterance IDs (includes all augmentations of selected bases)
    """
    # Group utterances by base ID and augmentation type
    base_to_variants = {}  # base_id -> list of all augmented variants
    
    for key in keys:
        base_id = get_base_utt_id(key)
        if base_id not in base_to_variants:
            base_to_variants[base_id] = []
        base_to_variants[base_id].append(key)
    
    total_base_utts = len(base_to_variants)
    
    # If we have fewer unique utterances than the limit, return all
    if total_base_utts <= max_utterances:
        print(f"  Dataset has {total_base_utts} unique utterances (limit: {max_utterances})")
        return keys
    
    # Select first N unique base utterances
    selected_bases = sorted(base_to_variants.keys())[:max_utterances]
    
    # Collect all augmentations of selected base utterances
    selected = []
    aug_type_counts = {}
    
    for base_id in selected_bases:
        variants = base_to_variants[base_id]
        selected.extend(variants)
        
        # Count augmentation types for reporting
        for variant in variants:
            if variant.endswith('-reverb'):
                aug_type = 'reverb'
            elif variant.endswith('-noise'):
                aug_type = 'noise'
            elif variant.endswith('-music'):
                aug_type = 'music'
            elif variant.endswith('-babble'):
                aug_type = 'babble'
            else:
                aug_type = 'original'
            aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1
    
    print(f"  Selected {len(selected_bases)} unique base utterances from {total_base_utts} total")
    print(f"  Total samples (including all augmentations): {len(selected)}")
    print(f"  Augmentation distribution:")
    for aug_type in sorted(aug_type_counts.keys()):
        print(f"    {aug_type}: {aug_type_counts[aug_type]}")
    
    return np.array(selected)
