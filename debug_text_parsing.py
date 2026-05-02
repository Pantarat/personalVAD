#!/usr/bin/env python3
"""Debug text file parsing to find the labeling bug."""

import numpy as np
import sys

# Read the text file
text_file = "data/overlap/text"
target_utt = "84_chunk_00001_OV_3575-170457-0034"

print(f"Reading {text_file}...")
with open(text_file, 'r') as f:
    for line in f:
        if line.startswith(target_utt):
            print(f"\nFound line for {target_utt}")
            print(f"Line length: {len(line)} chars")
            print(f"First 100 chars: {line[:100]}")
            
            # Parse like extract_features.py
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                print(f"ERROR: Expected 2 parts after split, got {len(parts)}")
                sys.exit(1)
            
            utt_id, rest = parts
            print(f"\nutt_id: {utt_id}")
            print(f"rest (first 100 chars): {rest[:100]}")
            
            # Partition on first space
            labels_str, _, tstamps_str = rest.partition(' ')
            print(f"\nlabels_str (first 100 chars): {labels_str[:100]!r}")
            print(f"tstamps_str (first 100 chars): {tstamps_str[:100]!r}")
            
            # Split labels
            gtruth = labels_str.split(',')
            tstamps_list = tstamps_str.split(' ')
            tstamps = np.array([int(float(stamp)*1000) for stamp in tstamps_list], dtype='int32')
            
            print(f"\n=== PARSED DATA ===")
            print(f"Number of labels: {len(gtruth)}")
            print(f"Number of timestamps: {len(tstamps)}")
            print(f"\nFirst 10 labels:")
            for i, lbl in enumerate(gtruth[:10]):
                print(f"  [{i}] = {lbl!r} (len={len(lbl)}, type={type(lbl).__name__})")
            
            print(f"\nFirst 11 timestamps (seconds):")
            for i, ts in enumerate(tstamps[:11]):
                print(f"  [{i}] = {ts/1000.0:.2f}s")
            
            # Simulate labeling logic
            print(f"\n=== SIMULATING LABEL CREATION ===")
            # Convert to 10ms frames
            tstamps_frames = tstamps // 10
            n = 3821  # Approximate frame count for ~38s audio
            
            print(f"Format check: len(gtruth)={len(gtruth)} vs len(tstamps)-1={len(tstamps)-1}")
            
            if len(gtruth) == len(tstamps) - 1:
                print("Using NEW format")
                labels = np.zeros(n, dtype=np.float32)
                
                # Process first 5 labels
                for i in range(min(5, len(gtruth))):
                    label = gtruth[i]
                    stamp_start = min(tstamps_frames[i], n)
                    stamp_end = min(tstamps_frames[i + 1], n)
                    
                    print(f"\nLabel[{i}]: {label!r}")
                    print(f"  Time: {tstamps[i]/1000:.2f}s to {tstamps[i+1]/1000:.2f}s")
                    print(f"  Frames: {stamp_start} to {stamp_end}")
                    print(f"  Duration: {(stamp_end-stamp_start)/100:.2f}s")
                    print(f"  Condition checks:")
                    print(f"    label == '': {label == ''}")
                    print(f"    label == 'T': {label == 'T'}")
                    print(f"    label == 'N': {label == 'N'}")
                    
                    if label == '' or label == '$':
                        labels[stamp_start:stamp_end] = 0
                        class_name = "SILENCE (0)"
                    elif label == 'T':
                        labels[stamp_start:stamp_end] = 2
                        class_name = "TARGET (2)"
                    elif label == 'N':
                        labels[stamp_start:stamp_end] = 1
                        class_name = "NTSS (1)"
                    else:
                        labels[stamp_start:stamp_end] = 0
                        class_name = "ELSE->SILENCE (0)"
                    
                    print(f"  -> Assigned: {class_name}")
                
                print(f"\n=== RESULT CHECK ===")
                print(f"labels[0:10] = {labels[0:10]}")
                print(f"labels[1880:1885] = {labels[1880:1885]}")
                
            else:
                print("Would use OLD format")
            
            break
else:
    print(f"ERROR: Could not find {target_utt} in {text_file}")
