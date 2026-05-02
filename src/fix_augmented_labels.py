#!/usr/bin/env python3
"""
Fix augmented data labels by copying text entries for suffixed utterances.

When Kaldi augmentation adds suffixes like -reverb, -noise, -babble to utterance IDs,
the text file (containing labels) needs to have matching suffixed entries.

This script fixes the issue by:
1. Reading the original text file
2. For each augmented utterance in wav.scp, creating a corresponding text entry

Usage:
    python src/fix_augmented_labels.py --data_dir kaldi/egs/pvad/data/augmented \\
                                       --original_text kaldi/egs/pvad/data/overlap/text
"""

import argparse
import os
import sys

def load_text_dict(text_file):
    """Load text file into dictionary"""
    text_dict = {}
    with open(text_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            rest = parts[1]
            text_dict[utt_id] = rest
    return text_dict

def load_wav_ids(wav_scp):
    """Load utterance IDs from wav.scp"""
    utt_ids = []
    with open(wav_scp, 'r') as f:
        for line in f:
            utt_id = line.strip().split()[0]
            utt_ids.append(utt_id)
    return utt_ids

def get_base_utt_id(utt_id):
    """
    Get base utterance ID by removing augmentation suffix.
    
    Examples:
        84-121550-0000_OV_174-50561-0029-reverb -> 84-121550-0000_OV_174-50561-0029
        84-121550-0000_OV_174-50561-0029-babble -> 84-121550-0000_OV_174-50561-0029
        84-121550-0000_OV_174-50561-0029-noise -> 84-121550-0000_OV_174-50561-0029
        84-121550-0000_OV_174-50561-0029 -> 84-121550-0000_OV_174-50561-0029 (no change)
    """
    # Check for known suffixes
    suffixes = ['-reverb', '-noise', '-music', '-babble']
    for suffix in suffixes:
        if utt_id.endswith(suffix):
            return utt_id[:-len(suffix)]
    return utt_id

def main():
    parser = argparse.ArgumentParser(description="Fix augmented data labels")
    parser.add_argument('--data_dir', type=str, required=True,
                       help="Path to augmented data directory")
    parser.add_argument('--original_text', type=str, required=True,
                       help="Path to original text file (before augmentation)")
    parser.add_argument('--output_text', type=str, default=None,
                       help="Output text file path (default: data_dir/text)")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    original_text = args.original_text
    output_text = args.output_text or os.path.join(data_dir, 'text')
    
    print("=" * 80)
    print("FIX AUGMENTED LABELS")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Original text: {original_text}")
    print(f"Output text: {output_text}\n")
    
    # Check if files exist
    wav_scp = os.path.join(data_dir, 'wav.scp')
    
    if not os.path.exists(original_text):
        print(f"❌ Error: Original text file not found: {original_text}")
        sys.exit(1)
    
    if not os.path.exists(wav_scp):
        print(f"❌ Error: wav.scp not found: {wav_scp}")
        sys.exit(1)
    
    # Load original text
    print("Loading original text file...")
    original_dict = load_text_dict(original_text)
    print(f"  Loaded {len(original_dict)} entries")
    
    # Load utterance IDs from wav.scp
    print("\nLoading wav.scp...")
    wav_utt_ids = load_wav_ids(wav_scp)
    print(f"  Found {len(wav_utt_ids)} utterances")
    
    # Create new text entries
    print("\nCreating augmented text entries...")
    new_text_dict = {}
    found = 0
    not_found = 0
    not_found_list = []
    
    for utt_id in wav_utt_ids:
        base_id = get_base_utt_id(utt_id)
        
        if base_id in original_dict:
            new_text_dict[utt_id] = original_dict[base_id]
            found += 1
        else:
            not_found += 1
            if not_found <= 10:
                not_found_list.append((utt_id, base_id))
    
    print(f"  Created entries: {found}")
    print(f"  Not found: {not_found}")
    
    if not_found_list:
        print(f"\n⚠️  Sample of not found utterances:")
        for utt_id, base_id in not_found_list:
            print(f"    {utt_id} (base: {base_id})")
    
    # Write new text file
    print(f"\nWriting new text file to {output_text}...")
    with open(output_text, 'w') as f:
        for utt_id in wav_utt_ids:  # Maintain wav.scp order
            if utt_id in new_text_dict:
                f.write(f"{utt_id} {new_text_dict[utt_id]}\n")
    
    print(f"✓ Wrote {len(new_text_dict)} entries")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Verify the new text file: head {output_text}")
    print(f"  2. Check for consistency: wc -l {wav_scp} {output_text}")
    print(f"  3. Run diagnosis: python src/diagnose_augmentation_labels.py --data_dir {data_dir}")

if __name__ == '__main__':
    main()
