#!/usr/bin/python
"""Diagnose NTSS labeling at the end of audio files."""

import os
import sys
import soundfile as sf
import numpy as np

def analyze_labels_file(data_dir):
    """Analyze labels from text file to check end segments."""
    text_file = os.path.join(data_dir, 'text')
    
    if not os.path.exists(text_file):
        print(f"Error: {text_file} not found")
        return
    
    issues = []
    total_samples = 0
    
    with open(text_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            utt_id = parts[0]
            labels = parts[1].split(',')
            timestamps = [float(t) for t in parts[2:]]
            
            if len(timestamps) != len(labels) + 1:
                print(f"Warning: Timestamp/label mismatch in {utt_id}")
                continue
            
            total_samples += 1
            
            # Check the last label
            last_label = labels[-1]
            last_start = timestamps[-2]
            last_end = timestamps[-1]
            last_duration = last_end - last_start
            
            # Load audio to get actual duration
            audio_path = os.path.join(data_dir, 'audio_data', f'{utt_id}.flac')
            if os.path.exists(audio_path):
                audio, sr = sf.read(audio_path)
                actual_duration = audio.size / sr
                
                # Check if there's NTSS audio at the end
                # Sample the last 0.5 seconds and check energy
                end_samples = min(int(0.5 * sr), audio.size)
                end_audio = audio[-end_samples:]
                end_energy = np.mean(end_audio ** 2)
                
                # If last segment is labeled as silence but has significant energy
                if last_label == '' and end_energy > 1e-6:
                    issues.append({
                        'utt_id': utt_id,
                        'line': line_num,
                        'last_label': last_label,
                        'last_duration': last_duration,
                        'end_energy': end_energy,
                        'actual_duration': actual_duration,
                        'label_duration': last_end
                    })
                    print(f"\n❌ Issue in {utt_id}:")
                    print(f"   Last segment: {last_start:.2f}-{last_end:.2f}s (label: '{last_label}')")
                    print(f"   End energy: {end_energy:.6f} (should be ~0 for silence)")
                    print(f"   Actual audio duration: {actual_duration:.2f}s")
                    
                    # Check metadata
                    meta_path = os.path.join(data_dir, 'audio_data', f'{utt_id}.overlap_meta')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as mf:
                            meta_lines = mf.readlines()
                            print(f"   Metadata:")
                            for ml in meta_lines:
                                if 'standalone_ntss' in ml or 'NTSS' in ml:
                                    print(f"     {ml.strip()}")
    
    print(f"\n{'='*60}")
    print(f"Summary: Found {len(issues)} samples with potential end-labeling issues")
    print(f"Total samples analyzed: {total_samples}")
    
    if issues:
        print(f"\nCommon pattern:")
        avg_end_energy = np.mean([i['end_energy'] for i in issues])
        print(f"  Average end energy: {avg_end_energy:.6f}")
        print(f"  (Energy > 1e-6 indicates speech, not silence)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_end_ntss.py <data_dir>")
        print("Example: python diagnose_end_ntss.py data/overlap_50pct_aug_main84_1000")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    analyze_labels_file(data_dir)
