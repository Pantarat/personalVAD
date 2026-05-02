#!/usr/bin/env python3
"""Test label parsing for the problematic utterance."""

import numpy as np

# Simulate the text file line
text_line = "84_chunk_00001_OV_3575-170457-0034 ,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,N,N,,N,N,,N,N,N,N,N,N,,N,N,N,,N,N,N,N,,N,N,N,,N,,N,N,N,,N,N,,N,N,N,N,N,N,,N,N,,N 0.00 18.82 18.84 18.85 18.87 18.89 18.91 18.92 18.94 18.96 18.98 19.00 19.01 19.03 19.05 19.07 19.09 19.10 19.12 19.14 19.16 19.17 19.19 19.21 19.23 19.25 19.26 19.28 19.30 19.32 19.34 19.35 19.37 19.39 19.41 19.42 19.44 19.46 19.48 19.50 19.51 19.53 19.55 19.57 19.59 19.60 19.62 19.64 19.66 19.67 19.69 19.71 19.73 20.18 20.19 20.21 20.23 20.25 20.26 20.28 20.30 20.32 20.34 20.35 20.37 20.39 20.41 20.42 20.44 20.46 20.48 20.50 20.51 20.53 20.55 20.57 20.59 20.60 20.62 20.64 20.66 20.67 20.69 20.71 20.73 20.75 20.76 20.78 20.80 20.82 20.84 20.85 20.87 20.89 20.91 20.92 20.94 20.96 20.98 21.02 21.03 21.05 21.07 21.09 21.10 21.12 21.14 21.16 21.17 21.19 21.21 21.23 21.25 21.26 21.28 21.30 21.32 21.34 21.35 21.37 21.39 21.41 21.42 21.44 21.46 21.48 21.50 21.51 21.53 21.55 21.57 21.59 21.60 21.62 21.64 21.66 21.67 21.69 21.71 21.73 21.75 21.76 21.78 21.80 21.82 21.84 21.85 21.87 21.89 21.91 21.92 21.94 21.96 21.98 22.00 22.01 22.03 22.05 22.07 22.09 22.10 22.12 22.14 22.16 22.17 22.19 22.21 22.23 22.25 22.26 22.28 22.30 22.32 22.34 22.35 22.37 22.39 22.41 22.42 22.44 22.46 22.48 22.50 22.51 22.53 22.55 22.57 22.59 22.60 22.62 22.64 22.66 22.67 22.69 22.71 22.73 22.75 22.76 22.78 22.80 22.82 22.84 22.85 22.87 22.89 22.91 22.92 22.94 22.96 22.98 23.00 23.01 23.03 23.05 23.07 23.09 23.10 23.12 23.14 23.16 23.17 23.19 23.21 23.23 23.25 23.26 23.28 23.30 23.32 23.34 23.35 23.37 23.39 23.41 23.42 23.44 23.46 23.48 23.50 23.51 23.53 23.55 23.57 23.59 23.60 23.62 23.64 23.66 23.67 23.69 23.71 23.73 23.75 23.76 23.78 23.80 23.82 24.13 24.56 25.63 25.83 26.28 26.31 26.51 26.96 27.11 27.42 27.48 27.86 27.89 28.23 28.30 28.90 29.32 29.72 29.80 30.17 30.70 30.73 30.87 30.93 31.44 31.47 31.81 31.84 32.42 32.94 33.49 33.89 34.31 34.72 34.97 35.39 35.91 36.05 36.22 36.51 37.07 37.42 37.78 38.18 38.21"

# Parse like extract_features.py does
utt_id, rest = text_line.split(' ', 1)
labels_str, _, tstamps_str = rest.partition(' ')

print(f"utt_id: {utt_id}")
print(f"\nlabels_str (first 50 chars): {labels_str[:50]!r}")
print(f"tstamps_str (first 50 chars): {tstamps_str[:50]!r}")

gtruth = labels_str.split(',')
tstamps = np.array([int(float(stamp)*1000) for stamp in tstamps_str.split(' ')], dtype='int32')

print(f"\n=== PARSED ===")
print(f"gtruth length: {len(gtruth)}")
print(f"tstamps length: {len(tstamps)}")
print(f"First 10 labels: {gtruth[:10]}")
print(f"Label types: {[type(l).__name__ for l in gtruth[:10]]}")
print(f"Label repr: {[repr(l) for l in gtruth[:10]]}")
print(f"First 11 timestamps (ms): {tstamps[:11]}")

# Assume 38.21s audio @ 100fps = 3821 frames
n = 3821
labels = np.zeros(n, dtype=np.float32)

print(f"\n=== LABELING (NEW FORMAT) ===")
print(f"Format check: len(gtruth)={len(gtruth)} vs len(tstamps)-1={len(tstamps)-1}")

# Note: tstamps are in milliseconds, need to convert to frames (10ms each)
# Divide by 10 to get frame index
tstamps_frames = tstamps // 10

if len(gtruth) == len(tstamps) - 1:
    print("Using NEW format (N labels, N+1 timestamps)")
    
    for i, label in enumerate(gtruth[:5]):  # Just first 5
        stamp_start = min(tstamps_frames[i], n)
        stamp_end = min(tstamps_frames[i + 1], n)
        
        print(f"\nLabel {i}: {repr(label)}")
        print(f"  Type: {type(label).__name__}")
        print(f"  label == '': {label == ''}")
        print(f"  bool(label): {bool(label)}")
        print(f"  len(label): {len(label)}")
        print(f"  Frames: {stamp_start} to {stamp_end} ({(stamp_end-stamp_start)/100:.2f}s)")
        
        if label == '' or label == '$':
            labels[stamp_start:stamp_end] = 0
            print(f"  -> Set to class 0 (SILENCE)")
        elif label == 'T':
            labels[stamp_start:stamp_end] = 2
            print(f"  -> Set to class 2 (TARGET)")
        elif label == 'N':
            labels[stamp_start:stamp_end] = 1
            print(f"  -> Set to class 1 (NTSS)")
        else:
            labels[stamp_start:stamp_end] = 0
            print(f"  -> ELSE: Set to class 0 (SILENCE)")

print(f"\n=== VERIFICATION ===")
print(f"First 10 label values: {labels[:10]}")
print(f"Labels at frame 0-10: class {labels[0]}")
print(f"Labels at frame 1000-1010: class {labels[1000]}")
print(f"Labels at frame 1882: class {labels[1882] if 1882 < len(labels) else 'OUT OF RANGE'}")
