#!/usr/bin/env python
"""Diagnose labeling issues at the end of audio files."""

import numpy as np
import soundfile as sf
import sys
import os
from kaldiio import load_scp

# Configuration
DATASET_DIR = "data/overlap_0pct_main84_500_new2"
AUDIO_DIR = f"data/audio_data/overlap_0pct_main84_500_new2_audio"
SAMPLE_ID = "84-121123-0000_00121"  # Example utterance

# Find first subdirectory
subdirs = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
if subdirs:
    AUDIO_DIR = os.path.join(AUDIO_DIR, subdirs[0])
    print(f"Using audio directory: {AUDIO_DIR}")

# Load audio
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.flac')]
if audio_files:
    audio_path = os.path.join(AUDIO_DIR, audio_files[0])
    print(f"Analyzing: {audio_path}")
    audio, sr = sf.read(audio_path)
else:
    print("No audio files found")
    sys.exit(1)

# Load labels
labels_scp_path = f"{DATASET_DIR}/labels.scp"
if os.path.exists(labels_scp_path):
    labels_dict = load_scp(labels_scp_path)
    utt_id = os.path.splitext(audio_files[0])[0]
    if utt_id in labels_dict:
        labels = labels_dict[utt_id]
    else:
        print(f"Utterance {utt_id} not found in labels")
        print(f"Available utterances: {list(labels_dict.keys())[:5]}")
        # Try first available
        utt_id = list(labels_dict.keys())[0]
        labels = labels_dict[utt_id]
        print(f"Using first utterance: {utt_id}")
        # Reload audio for this utterance
        audio_path = os.path.join(AUDIO_DIR, f"{utt_id}.flac")
        audio, sr = sf.read(audio_path)
else:
    print(f"Labels file not found: {labels_scp_path}")
    sys.exit(1)

# Analyze last 2 seconds
duration = len(audio) / sr
last_n_seconds = 2.0
last_n_samples = int(last_n_seconds * sr)
last_audio = audio[-last_n_samples:]
last_labels = labels[-int(last_n_seconds * 100):]  # 10ms frames

print(f"\n{'='*60}")
print(f"AUDIO ANALYSIS (last {last_n_seconds}s)")
print(f"{'='*60}")
print(f"Total duration: {duration:.2f}s")
print(f"Sample rate: {sr} Hz")
print(f"\nLast {last_n_seconds}s audio statistics:")
print(f"  - Max amplitude: {np.max(np.abs(last_audio)):.6f}")
print(f"  - Mean energy: {np.mean(last_audio**2):.6e}")
print(f"  - RMS: {np.sqrt(np.mean(last_audio**2)):.6f}")

# Frame-level analysis
frame_len = int(0.02 * sr)  # 20ms
hop_len = frame_len // 2

print(f"\nFrame-level analysis (20ms frames):")
for i in range(0, min(len(last_audio) - frame_len, last_n_samples), hop_len):
    frame = last_audio[i:i+frame_len]
    energy = np.mean(frame**2)
    time_in_last_section = i / sr
    abs_time = duration - last_n_seconds + time_in_last_section
    
    # Get label for this time
    label_idx = int((abs_time * 100)) - int((duration - last_n_seconds) * 100)
    if 0 <= label_idx < len(last_labels):
        label = int(last_labels[label_idx])
        label_str = {0: 'NS', 1: 'NTSS', 2: 'TSS'}[label]
    else:
        label_str = 'N/A'
    
    if i % (10 * hop_len) == 0:  # Print every 10 frames (~100ms)
        print(f"  Frame @ {abs_time:.3f}s: energy={energy:.6e}, label={label_str}")

print(f"\n{'='*60}")
print(f"LABEL ANALYSIS (last {last_n_seconds}s)")
print(f"{'='*60}")

# Count labels in last section
label_counts = {0: 0, 1: 0, 2: 0}
for label in last_labels:
    label_counts[int(label)] += 1

total = len(last_labels)
print(f"Label distribution in last {last_n_seconds}s:")
print(f"  - Class 0 (NS/silence): {label_counts[0]}/{total} ({100*label_counts[0]/total:.1f}%)")
print(f"  - Class 1 (NTSS): {label_counts[1]}/{total} ({100*label_counts[1]/total:.1f}%)")
print(f"  - Class 2 (TSS): {label_counts[2]}/{total} ({100*label_counts[2]/total:.1f}%)")

# Check if there's audible audio but labeled as silence
has_energy = np.max(np.abs(last_audio)) > 0.01  # Significant amplitude
has_ntss_label = label_counts[1] > 0
has_silence_label = label_counts[0] > 0

print(f"\n{'='*60}")
print(f"DIAGNOSIS")
print(f"{'='*60}")

if has_energy and has_silence_label and not has_ntss_label:
    print("⚠️  ISSUE DETECTED:")
    print("   - Audio has significant energy (speech)")
    print("   - But it's labeled as silence (class 0)")
    print("   - Should be labeled as NTSS (class 1)")
    print("\nPossible causes:")
    print("   1. Energy threshold too high in detect_speech_regions()")
    print("   2. NTSS segment not being detected/added properly")
    print("   3. Event timeline not covering the end properly")
elif has_energy and has_ntss_label:
    print("✓ Looks OK:")
    print("   - Audio has energy and is labeled as NTSS")
elif not has_energy:
    print("✓ Correctly labeled as silence (no energy)")

print(f"\n{'='*60}")
