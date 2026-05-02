#!/usr/bin/env python3
"""Test label parsing to debug the silence labeling issue."""

import numpy as np

# Simulate what generate_overlapping_utterances.py creates
labels_from_generator = ['', 'T', 'T', 'N', '']
timestamps_from_generator = [0.00, 18.55, 25.32, 32.10, 38.45, 40.00]

# Format as string (what gets written to text file)
labels_str = ','.join(labels_from_generator)
timestamps_str = ' '.join([f'{t:.2f}' for t in timestamps_from_generator])
text_line = f"test_utt {labels_str} {timestamps_str}"

print("=== GENERATOR OUTPUT ===")
print(f"Labels list: {labels_from_generator}")
print(f"Timestamps list: {timestamps_from_generator}")
print(f"Text line: {text_line}")
print()

# Simulate what extract_features.py does
print("=== EXTRACT_FEATURES PARSING ===")
parts = text_line.split(' ', 1)  # Split into utt_id and rest
utt_id = parts[0]
rest = parts[1]

labels, _, tstamps = rest.partition(' ')
print(f"Raw labels string: {labels!r}")
print(f"Raw timestamps string: {tstamps!r}")
print()

# Parse
gtruth = labels.split(',')
tstamps_array = np.array([int(float(stamp)*1000) for stamp in tstamps.split(' ')], dtype='int32')

print(f"Parsed labels (gtruth): {gtruth}")
print(f"Parsed timestamps (ms): {tstamps_array}")
print(f"Number of labels: {len(gtruth)}")
print(f"Number of timestamps: {len(tstamps_array)}")
print()

# Simulate the labeling logic
n = 4000  # Assume 40-second audio = 4000 frames (10ms each)
labels_output = np.zeros(n, dtype=np.float32)
tstamps_frames = tstamps_array // 10

print("=== LABELING LOGIC (NEW FORMAT) ===")
if len(gtruth) == len(tstamps_frames) - 1:
    print("Using NEW format (N labels, N+1 timestamps)")
    print()
    
    for i, label in enumerate(gtruth):
        stamp_start = min(tstamps_frames[i], n)
        stamp_end = min(tstamps_frames[i + 1], n)
        
        if label == '' or label == '$':
            class_val = 0  # Silence
            class_name = "SILENCE"
        elif label == 'T':
            class_val = 2  # Target
            class_name = "TARGET"
        elif label == 'N':
            class_val = 1  # Non-target
            class_name = "NON-TARGET"
        else:
            class_val = 0  # Unknown -> silence
            class_name = "SILENCE (unknown)"
        
        labels_output[stamp_start:stamp_end] = class_val
        
        print(f"Segment {i}: label={label!r:5} -> frames {stamp_start:4}-{stamp_end:4} -> class {class_val} ({class_name})")
else:
    print(f"ERROR: Format mismatch! {len(gtruth)} labels != {len(tstamps_frames)} - 1")

print()
print("=== VERIFICATION ===")
# Check a few frames
check_frames = [0, 100, 1000, 1855, 1856, 2532, 3210, 3845, 3999]
for frame in check_frames:
    if frame < n:
        time_s = frame * 0.01
        class_val = labels_output[frame]
        class_name = ["SILENCE", "NON-TARGET", "TARGET"][int(class_val)]
        print(f"Frame {frame:4} (time={time_s:.2f}s): class={int(class_val)} ({class_name})")
