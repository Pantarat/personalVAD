#!/usr/bin/env python3
"""Check labels.scp content to debug labeling issue."""

import sys
import kaldiio
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python check_labels_scp.py <labels.scp> <utt_id>")
    sys.exit(1)

labels_scp = sys.argv[1]
utt_id = sys.argv[2]

# Load the labels
labels_dict = kaldiio.load_scp(labels_scp)

if utt_id not in labels_dict:
    print(f"Error: utterance '{utt_id}' not found in {labels_scp}")
    print(f"Available utterances (first 5): {list(labels_dict.keys())[:5]}")
    sys.exit(1)

labels = labels_dict[utt_id]

print(f"=== Labels for {utt_id} ===")
print(f"Shape: {labels.shape}")
print(f"Dtype: {labels.dtype}")
print(f"\nFirst 2000 frames (0-20 seconds @ 10ms frames):")
print(f"Labels: {labels[:2000]}")
print(f"\nClass distribution in first 2000 frames:")
print(f"  Class 0 (SILENCE): {np.sum(labels[:2000] == 0)} frames")
print(f"  Class 1 (NTSS): {np.sum(labels[:2000] == 1)} frames")
print(f"  Class 2 (TSS): {np.sum(labels[:2000] == 2)} frames")

# Find transitions
transitions = []
prev_label = labels[0]
transition_start = 0

for i in range(1, min(2000, len(labels))):
    if labels[i] != prev_label:
        label_name = 'SILENCE' if prev_label == 0 else ('NTSS' if prev_label == 1 else 'TSS')
        transitions.append((transition_start/100, i/100, label_name, prev_label))
        transition_start = i
        prev_label = labels[i]

# Final segment
label_name = 'SILENCE' if prev_label == 0 else ('NTSS' if prev_label == 1 else 'TSS')
transitions.append((transition_start/100, min(2000, len(labels))/100, label_name, prev_label))

print(f"\nSegments in first 20 seconds:")
for start, end, name, cls in transitions[:20]:
    print(f"  {start:.2f}-{end:.2f}s: {name} (class {cls})")
