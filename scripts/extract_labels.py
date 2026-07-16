#!/usr/bin/env python3
"""
Extract and visualize labels from Kaldi format label files.
Works with both legacy .scp files and new single-speaker datasets.

Usage:
    # Extract specific utterance
    python extract_labels.py --labels_scp data/speaker_1089/train/labels.scp \
        --output_dir data/extracted_labels --utt_id 1089_chunk_00000
    
    # Extract all utterances
    python extract_labels.py --labels_scp data/speaker_1089/train/labels.scp \
        --output_dir data/extracted_labels --all
"""

import argparse
import os
from pathlib import Path
import numpy as np

# ================ CONFIGURATION - EDIT HERE =========================

# Default paths
LABELS_SCP = "data/61_ov_test_noOther_main61_500_31-5-2026/labels.scp"
OUTPUT_DIR = "data/extracted_labels/"
TARGET_UTT = "61_chunk_00001_OV_none"  # Set to specific utt_id or None to extract all
FRAME_SHIFT = 0.01  # seconds (10ms frames)
LABEL_TYPE = "multiclass"  # Options: 'binary' (0/1), 'multiclass' (0/1/2), 'auto' (detect from data)

# ====================================================================


def load_labels_from_scp(scp_path: str, utt_id: str = None):
    """
    Load labels from labels.scp file.
    
    Args:
        scp_path: Path to labels.scp file
        utt_id: Specific utterance ID to load (None = load all)
    
    Returns:
        dict: {utt_id: labels_array} or {utt_id: labels_array} for single utterance
    """
    labels_dict = {}
    
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            current_utt_id = parts[0]
            label_path = parts[1]
            
            # Skip if we're looking for a specific utterance and this isn't it
            if utt_id is not None and current_utt_id != utt_id:
                continue
            
            # Load labels based on file extension
            if label_path.endswith('.npy'):
                # Numpy format (from extract_single_speaker.py)
                labels = np.load(label_path)
            else:
                # Try kaldiio for other formats
                try:
                    from kaldiio import load_scp
                    all_labels = load_scp(scp_path)
                    labels = all_labels[current_utt_id]
                except ImportError:
                    print("ERROR: kaldiio not installed. Install with: pip install kaldiio")
                    return None
                except Exception as e:
                    print(f"ERROR loading {current_utt_id}: {e}")
                    continue
            
            labels_dict[current_utt_id] = labels
            
            # If we found the specific utterance we were looking for, stop
            if utt_id is not None:
                break
    
    return labels_dict


def save_labels_to_text(labels: np.ndarray, output_path: str, frame_shift: float = 0.01):
    """
    Save labels to text file with timestamps.
    
    Format: start_time end_time label
    """
    with open(output_path, 'w') as f:
        for i, lab in enumerate(labels):
            start = i * frame_shift
            end = (i + 1) * frame_shift
            f.write(f"{start:.3f}\t{end:.3f}\t{int(lab)}\n")


def print_label_statistics(labels: np.ndarray, utt_id: str, label_type: str = 'auto'):
    """Print statistics about the labels.
    
    Args:
        labels: Label array
        utt_id: Utterance ID for display
        label_type: 'binary', 'multiclass', or 'auto' to detect from data
    """
    total_frames = len(labels)
    duration = total_frames * FRAME_SHIFT
    
    # Determine label type
    if label_type == 'auto':
        # Auto-detect based on unique values
        unique_labels = np.unique(labels)
        is_binary = set(unique_labels).issubset({0, 1})
    elif label_type == 'binary':
        is_binary = True
    elif label_type == 'multiclass':
        is_binary = False
    else:
        raise ValueError(f"Invalid label_type: {label_type}. Must be 'binary', 'multiclass', or 'auto'")
    
    # Print statistics based on label type
    if is_binary:
        # Binary labels: 0=non-speech, 1=speech
        speech_frames = np.sum(labels == 1)
        non_speech_frames = np.sum(labels == 0)
        
        print(f"\n{utt_id}:")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Speech frames: {speech_frames} ({speech_frames/total_frames*100:.1f}%)")
        print(f"  Non-speech frames: {non_speech_frames} ({non_speech_frames/total_frames*100:.1f}%)")
    else:
        # Multi-class labels: 0=silence, 1=non-target speech, 2=target speech
        silence_frames = np.sum(labels == 0)
        non_target_frames = np.sum(labels == 1)
        target_frames = np.sum(labels == 2)
        
        print(f"\n{utt_id}:")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Silence/Non-speech (0): {silence_frames} ({silence_frames/total_frames*100:.1f}%)")
        print(f"  Non-target speech (1): {non_target_frames} ({non_target_frames/total_frames*100:.1f}%)")
        print(f"  Target speech (2): {target_frames} ({target_frames/total_frames*100:.1f}%)")
        print(f"  Total speech: {non_target_frames + target_frames} ({(non_target_frames + target_frames)/total_frames*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract labels from Kaldi format files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--labels_scp', type=str, default=LABELS_SCP,
                        help='Path to labels.scp file')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for extracted labels')
    parser.add_argument('--utt_id', type=str, default=TARGET_UTT,
                        help='Specific utterance ID to extract (None = extract all)')
    parser.add_argument('--all', action='store_true',
                        help='Extract all utterances')
    parser.add_argument('--frame_shift', type=float, default=FRAME_SHIFT,
                        help='Frame shift in seconds')
    parser.add_argument('--label_type', type=str, default=LABEL_TYPE,
                        choices=['binary', 'multiclass', 'auto'],
                        help='Label type: binary (0/1), multiclass (0/1/2), or auto (detect from data)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which utterances to extract
    utt_to_extract = None if args.all else args.utt_id
    
    # Load labels
    print(f"Loading labels from {args.labels_scp}...")
    labels_dict = load_labels_from_scp(args.labels_scp, utt_to_extract)
    
    if not labels_dict:
        print("ERROR: No labels found or failed to load")
        return 1
    
    print(f"Loaded {len(labels_dict)} utterance(s)")
    
    # Extract and save each utterance
    for utt_id, labels in labels_dict.items():
        output_path = os.path.join(args.output_dir, f"{utt_id}.txt")
        save_labels_to_text(labels, output_path, args.frame_shift)
        print_label_statistics(labels, utt_id, args.label_type)
        print(f"  Saved to: {output_path}")
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())

