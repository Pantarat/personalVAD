#!/usr/bin/env python3
"""
Diagnose augmentation label issues.

This script checks if:
1. Augmented utterance IDs in wav.scp have corresponding entries in text file
2. Labels are being correctly preserved after augmentation
3. No speech segments are incorrectly labeled as target speaker

Usage:
    python src/diagnose_augmentation_labels.py --data_dir kaldi/egs/pvad/data/augmented
"""

import argparse
import os
import numpy as np

def load_text_labels(text_file):
    """Load labels from text file"""
    labels_dict = {}
    with open(text_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 2)
            if len(parts) < 3:
                continue
            utt_id = parts[0]
            labels_str = parts[1]
            timestamps_str = parts[2]
            
            labels = labels_str.split(',')
            timestamps = [float(t) for t in timestamps_str.split()]
            
            labels_dict[utt_id] = {
                'labels': labels,
                'timestamps': timestamps
            }
    return labels_dict

def load_wav_scp(wav_scp):
    """Load utterance IDs from wav.scp"""
    utt_ids = []
    with open(wav_scp, 'r') as f:
        for line in f:
            utt_id = line.strip().split()[0]
            utt_ids.append(utt_id)
    return utt_ids

def analyze_labels(labels_dict):
    """Analyze labels to find potential issues"""
    issues = []
    stats = {
        'total_utterances': len(labels_dict),
        'has_silence': 0,
        'has_target': 0,
        'has_nontarget': 0,
        'only_silence': 0,
        'only_target': 0,
        'problematic': []
    }
    
    for utt_id, data in labels_dict.items():
        labels = data['labels']
        timestamps = data['timestamps']
        
        # Count label types
        has_empty = '' in labels or ' ' in labels
        has_T = 'T' in labels
        has_N = 'N' in labels
        
        if has_empty:
            stats['has_silence'] += 1
        if has_T:
            stats['has_target'] += 1
        if has_N:
            stats['has_nontarget'] += 1
        
        # Check for problematic patterns
        if not has_empty and has_T:
            # No silence labels but has target - potentially problematic
            stats['only_target'] += 1
            stats['problematic'].append(utt_id)
        
        if not has_empty and not has_T and not has_N:
            # No labels at all
            issues.append(f"{utt_id}: No valid labels found")
        
        # Check if timestamps match labels
        expected_timestamps = len(labels) + 1  # New format: N labels, N+1 timestamps
        legacy_timestamps = len(labels)  # Old format: N labels, N timestamps
        
        if len(timestamps) != expected_timestamps and len(timestamps) != legacy_timestamps:
            issues.append(f"{utt_id}: Mismatch - {len(labels)} labels but {len(timestamps)} timestamps")
    
    return stats, issues

def main():
    parser = argparse.ArgumentParser(description="Diagnose augmentation label issues")
    parser.add_argument('--data_dir', type=str, required=True,
                       help="Path to augmented data directory (e.g., kaldi/egs/pvad/data/augmented)")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    print("=" * 80)
    print("AUGMENTATION LABEL DIAGNOSIS")
    print("=" * 80)
    print(f"\nData directory: {data_dir}\n")
    
    # Check if files exist
    text_file = os.path.join(data_dir, 'text')
    wav_scp = os.path.join(data_dir, 'wav.scp')
    
    if not os.path.exists(text_file):
        print(f"❌ Error: text file not found: {text_file}")
        return
    
    if not os.path.exists(wav_scp):
        print(f"❌ Error: wav.scp not found: {wav_scp}")
        return
    
    # Load data
    print("Loading data...")
    labels_dict = load_text_labels(text_file)
    utt_ids_wav = load_wav_scp(wav_scp)
    
    print(f"  Utterances in wav.scp: {len(utt_ids_wav)}")
    print(f"  Utterances in text: {len(labels_dict)}")
    
    # Check for missing labels
    print("\n" + "=" * 80)
    print("CHECKING FOR MISSING LABELS")
    print("=" * 80)
    
    missing = []
    for utt_id in utt_ids_wav[:100]:  # Check first 100
        if utt_id not in labels_dict:
            missing.append(utt_id)
    
    if missing:
        print(f"\n⚠️  Found {len(missing)} utterances in wav.scp without labels in text file:")
        for utt_id in missing[:10]:
            print(f"  - {utt_id}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("\n✓ All checked utterances have labels")
    
    # Analyze labels
    print("\n" + "=" * 80)
    print("ANALYZING LABEL PATTERNS")
    print("=" * 80)
    
    stats, issues = analyze_labels(labels_dict)
    
    print(f"\nLabel Statistics:")
    print(f"  Total utterances: {stats['total_utterances']}")
    print(f"  Has silence ('') labels: {stats['has_silence']} ({stats['has_silence']/stats['total_utterances']*100:.1f}%)")
    print(f"  Has target ('T') labels: {stats['has_target']} ({stats['has_target']/stats['total_utterances']*100:.1f}%)")
    print(f"  Has non-target ('N') labels: {stats['has_nontarget']} ({stats['has_nontarget']/stats['total_utterances']*100:.1f}%)")
    print(f"  Only target labels (no silence): {stats['only_target']}")
    
    if stats['only_target'] > 0:
        print(f"\n⚠️  WARNING: Found {stats['only_target']} utterances with ONLY target labels (no silence)")
        print("  This likely indicates a labeling problem!")
        print("  First 5 problematic utterances:")
        for utt_id in stats['problematic'][:5]:
            data = labels_dict[utt_id]
            print(f"    {utt_id}: labels={data['labels'][:10]}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} label issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    
    # Sample some utterances
    print("\n" + "=" * 80)
    print("SAMPLE UTTERANCES")
    print("=" * 80)
    
    sample_utts = list(labels_dict.keys())[:5]
    for utt_id in sample_utts:
        data = labels_dict[utt_id]
        labels = data['labels']
        timestamps = data['timestamps']
        
        print(f"\n{utt_id}:")
        print(f"  Labels: {labels[:20]}{'...' if len(labels) > 20 else ''}")
        print(f"  Timestamps: {timestamps[:10]}{'...' if len(timestamps) > 10 else ''}")
        
        # Count label types
        label_counts = {}
        for label in labels:
            label_counts[label if label else 'SILENCE'] = label_counts.get(label if label else 'SILENCE', 0) + 1
        print(f"  Label counts: {label_counts}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
