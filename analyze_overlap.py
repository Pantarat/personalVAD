#!/usr/bin/env python
"""Analyze the actual overlap percentage in generated files by reading the text file."""

import sys

def parse_text_line(line):
    """Parse a line from the text file."""
    parts = line.strip().split()
    if len(parts) < 2:
        return None, None, None
    
    utt_id = parts[0]
    labels_str = parts[1]
    times_str = parts[2:]
    
    labels = labels_str.split(',')
    times = [float(t) for t in times_str]
    
    return utt_id, labels, times

def calculate_overlap_from_alignments(libri_root, utt_id, labels, times):
    """Calculate actual overlap by reading original alignment files."""
    import os
    
    # Parse utterance ID to find alignment file
    # Format: 84-121123-0000_OV_7850
    main_part = utt_id.split('_OV_')[0]  # 84-121123-0000
    speaker_id = main_part.split('-')[0]  # 84
    chapter_id = main_part.split('-')[1]  # 121123
    
    # This is complex - let's use a simpler approach
    # Just analyze the label transitions in our generated data
    return None

def analyze_labels_simple(labels, times):
    """Simple analysis of labels to estimate overlap.
    
    Since T means target speaking (with or without overlap) and N means only non-target,
    we can't directly measure overlap from labels alone.
    We need to reconstruct from the generation process.
    """
    if len(labels) == 0 or len(times) == 0:
        return 0, 0, 0
    
    prev_time = 0
    target_duration = 0
    non_target_duration = 0
    silence_duration = 0
    
    for i in range(len(times)):
        duration = times[i] - prev_time
        label = labels[i] if i < len(labels) else ''
        
        if label == 'T':
            target_duration += duration
        elif label == 'N':
            non_target_duration += duration
        else:  # '' or empty
            silence_duration += duration
        
        prev_time = times[i]
    
    return target_duration, non_target_duration, silence_duration

def analyze_overlap(text_file):
    """Analyze overlap percentage from text file."""
    
    with open(text_file, 'r') as f:
        lines = f.readlines()
    
    total_files = len(lines)
    total_target_speech = 0
    total_non_target_speech = 0
    total_silence = 0
    
    print("Per-file analysis:")
    print("=" * 80)
    
    for idx, line in enumerate(lines):
        utt_id, labels, times = parse_text_line(line)
        if not utt_id:
            continue
        
        target_dur, non_target_dur, silence_dur = analyze_labels_simple(labels, times)
        total_target_speech += target_dur
        total_non_target_speech += non_target_dur
        total_silence += silence_dur
        
        total_dur = target_dur + non_target_dur + silence_dur
        
        if idx < 10:  # Show first 10
            print(f"{utt_id}:")
            print(f"  Total: {total_dur:.2f}s | Target: {target_dur:.2f}s ({target_dur/total_dur*100:.1f}%) | "
                  f"Non-target: {non_target_dur:.2f}s ({non_target_dur/total_dur*100:.1f}%) | "
                  f"Silence: {silence_dur:.2f}s ({silence_dur/total_dur*100:.1f}%)")
            
            # Show label pattern
            label_str = ','.join(labels[:40])
            if len(labels) > 40:
                label_str += '...'
            print(f"  Labels: {label_str}\n")
    
    print("=" * 80)
    print(f"\nSummary for {total_files} files:")
    print(f"Total duration: {total_target_speech + total_non_target_speech + total_silence:.2f}s")
    print(f"Target speech (T): {total_target_speech:.2f}s ({total_target_speech/(total_target_speech+total_non_target_speech+total_silence)*100:.1f}%)")
    print(f"Non-target speech (N): {total_non_target_speech:.2f}s ({total_non_target_speech/(total_target_speech+total_non_target_speech+total_silence)*100:.1f}%)")
    print(f"Silence (''): {total_silence:.2f}s ({total_silence/(total_target_speech+total_non_target_speech+total_silence)*100:.1f}%)")
    
    print(f"\nNOTE: In PersonalVAD labeling:")
    print(f"  'T' = Target speaking (includes simultaneous overlap)")
    print(f"  'N' = ONLY non-target speaking")
    print(f"  ''  = Silence")
    print(f"\nTo measure actual speech-to-speech overlap, we need to check")
    print(f"if 'N' segments occur during target speech regions in the source files.")

if __name__ == '__main__':
    text_file = sys.argv[1] if len(sys.argv) > 1 else 'data/test_overlap/text'
    analyze_overlap(text_file)
