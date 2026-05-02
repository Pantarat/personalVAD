#!/usr/bin/env python3
"""
Diagnose utterance count issues after augmentation.

This script checks:
1. How many base utterances were generated
2. How many utterances exist after each augmentation type
3. Which base utterances failed to augment
4. Why augmentation might have failed

Usage:
    python src/diagnose_utterance_count.py --base_dir kaldi/egs/pvad/data/overlap \\
                                           --augmented_dir kaldi/egs/pvad/data/augmented
"""

import argparse
import os
import sys

def load_utt_ids_from_scp(scp_file):
    """Load utterance IDs from .scp file"""
    utt_ids = []
    if not os.path.exists(scp_file):
        return utt_ids
    
    with open(scp_file, 'r') as f:
        for line in f:
            utt_id = line.strip().split()[0]
            utt_ids.append(utt_id)
    return utt_ids

def get_base_utt_id(utt_id):
    """Remove augmentation suffix"""
    suffixes = ['-reverb', '-noise', '-music', '-babble']
    for suffix in suffixes:
        if utt_id.endswith(suffix):
            return utt_id[:-len(suffix)]
    return utt_id

def main():
    parser = argparse.ArgumentParser(description="Diagnose utterance count issues")
    parser.add_argument('--base_dir', type=str, required=True,
                       help="Base data directory (before augmentation)")
    parser.add_argument('--augmented_dir', type=str, default=None,
                       help="Augmented data directory (after augmentation)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("UTTERANCE COUNT DIAGNOSIS")
    print("=" * 80)
    
    # Check base directory
    base_wav_scp = os.path.join(args.base_dir, 'wav.scp')
    if not os.path.exists(base_wav_scp):
        print(f"\n❌ Error: Base wav.scp not found: {base_wav_scp}")
        sys.exit(1)
    
    base_utts = load_utt_ids_from_scp(base_wav_scp)
    print(f"\n📊 BASE DATASET: {args.base_dir}")
    print(f"   Total utterances: {len(base_utts)}")
    
    if len(base_utts) > 0:
        print(f"   Sample IDs: {base_utts[:3]}")
    
    # Check augmented directory if provided
    if args.augmented_dir and os.path.exists(args.augmented_dir):
        print(f"\n📊 AUGMENTED DATASET: {args.augmented_dir}")
        
        # Check individual augmentation directories
        aug_types = {
            'original': args.base_dir,
            'reverb': os.path.join(os.path.dirname(args.augmented_dir), 'reverb'),
            'noise': os.path.join(os.path.dirname(args.augmented_dir), 'noise'),
            'music': os.path.join(os.path.dirname(args.augmented_dir), 'music'),
            'babble': os.path.join(os.path.dirname(args.augmented_dir), 'babble'),
        }
        
        aug_utts = {}
        for aug_type, aug_dir in aug_types.items():
            wav_scp = os.path.join(aug_dir, 'wav.scp')
            if os.path.exists(wav_scp):
                utts = load_utt_ids_from_scp(wav_scp)
                aug_utts[aug_type] = utts
                print(f"\n   {aug_type}:")
                print(f"      Utterances: {len(utts)}")
                if len(utts) > 0:
                    print(f"      Sample IDs: {utts[:2]}")
        
        # Check combined augmented directory
        combined_wav_scp = os.path.join(args.augmented_dir, 'wav.scp')
        if os.path.exists(combined_wav_scp):
            combined_utts = load_utt_ids_from_scp(combined_wav_scp)
            print(f"\n   Combined augmented:")
            print(f"      Total utterances: {len(combined_utts)}")
            
            # Count by augmentation type
            type_counts = {}
            for utt_id in combined_utts:
                if utt_id.endswith('-reverb'):
                    aug_type = 'reverb'
                elif utt_id.endswith('-noise'):
                    aug_type = 'noise'
                elif utt_id.endswith('-music'):
                    aug_type = 'music'
                elif utt_id.endswith('-babble'):
                    aug_type = 'babble'
                else:
                    aug_type = 'original'
                type_counts[aug_type] = type_counts.get(aug_type, 0) + 1
            
            print(f"\n   Augmentation distribution:")
            for aug_type in sorted(type_counts.keys()):
                print(f"      {aug_type}: {type_counts[aug_type]}")
        
        # Check for missing augmentations
        print(f"\n📋 CHECKING FOR MISSING AUGMENTATIONS:")
        
        base_ids_set = set(base_utts)
        
        for aug_type, utts in aug_utts.items():
            if aug_type == 'original':
                continue
            
            # Get base IDs from augmented utterances
            aug_base_ids = set([get_base_utt_id(u) for u in utts])
            
            # Find which base utterances didn't get augmented
            missing = base_ids_set - aug_base_ids
            
            if missing:
                print(f"\n   {aug_type}: {len(missing)} base utterances not augmented")
                print(f"      Expected: {len(base_utts)}, Got: {len(utts)}")
                if len(missing) <= 10:
                    print(f"      Missing base IDs: {sorted(list(missing))[:10]}")
                else:
                    print(f"      Sample missing IDs: {sorted(list(missing))[:10]}")
            else:
                print(f"\n   {aug_type}: ✓ All base utterances augmented")
        
        # Check text file (labels)
        print(f"\n📝 CHECKING TEXT FILE (LABELS):")
        
        base_text = os.path.join(args.base_dir, 'text')
        if os.path.exists(base_text):
            with open(base_text, 'r') as f:
                base_text_lines = len(f.readlines())
            print(f"   Base text entries: {base_text_lines}")
        
        aug_text = os.path.join(args.augmented_dir, 'text')
        if os.path.exists(aug_text):
            with open(aug_text, 'r') as f:
                aug_text_lines = len(f.readlines())
            print(f"   Augmented text entries: {aug_text_lines}")
            
            if aug_text_lines < len(combined_utts):
                print(f"   ⚠️  WARNING: Text file has fewer entries than wav.scp!")
                print(f"   Missing {len(combined_utts) - aug_text_lines} label entries")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    print("\n💡 COMMON ISSUES:")
    print("   1. Augmentation scripts failing for some utterances (check Kaldi logs)")
    print("   2. Utterances too short for augmentation requirements")
    print("   3. MUSAN dataset not fully downloaded")
    print("   4. Text file not properly synchronized after augmentation")
    print("\n   Run: bash prepare_overlap_dataset.sh 2  (to re-run augmentation)")

if __name__ == '__main__':
    main()
