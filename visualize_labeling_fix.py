#!/usr/bin/env python
"""
Visual demonstration of the labeling fix.

This script shows how the labeling changes from the old approach to the new approach.
"""

import numpy as np
import matplotlib.pyplot as plt

def create_example_audio(duration=5.0, sr=16000):
    """Create example audio with speech and silence regions."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples)
    
    # Add speech in specific regions
    # Speech from 0.5s to 1.5s
    speech1_start = int(0.5 * sr)
    speech1_end = int(1.5 * sr)
    audio[speech1_start:speech1_end] = np.random.randn(speech1_end - speech1_start) * 0.1
    
    # Speech from 2.5s to 4.0s
    speech2_start = int(2.5 * sr)
    speech2_end = int(4.0 * sr)
    audio[speech2_start:speech2_end] = np.random.randn(speech2_end - speech2_start) * 0.1
    
    return audio, sr

def old_labeling(start_time, end_time, duration):
    """Old approach: label entire region as NTSS."""
    times = np.linspace(0, duration, 1000)
    labels = np.zeros(1000)
    
    # Mark entire region as NTSS (class 1)
    mask = (times >= start_time) & (times <= end_time)
    labels[mask] = 1
    
    return times, labels

def new_labeling(audio, sr, region_start, region_end):
    """New approach: detect actual speech within region."""
    # Extract region
    start_sample = int(region_start * sr)
    end_sample = int(region_end * sr)
    region_audio = audio[start_sample:end_sample]
    
    # Simple energy-based detection
    frame_len = int(0.02 * sr)  # 20ms frames
    hop_len = frame_len // 2
    
    times = []
    labels = []
    
    for i in range(0, len(region_audio) - frame_len, hop_len):
        frame = region_audio[i:i+frame_len]
        energy = np.sum(frame ** 2) / len(frame)
        time = region_start + i / sr
        
        # Threshold for speech detection
        if energy > 1e-5:
            labels.append(1)  # NTSS
        else:
            labels.append(0)  # Silence
        times.append(time)
    
    return np.array(times), np.array(labels)

# Create example
audio, sr = create_example_audio()
duration = len(audio) / sr

# NTSS region: 0 to 5 seconds (includes silence at start, middle, and end)
ntss_start = 0.0
ntss_end = 5.0

# Get labelings
times_old, labels_old = old_labeling(ntss_start, ntss_end, duration)
times_new, labels_new = new_labeling(audio, sr, ntss_start, ntss_end)

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot 1: Audio waveform
time_axis = np.linspace(0, duration, len(audio))
axes[0].plot(time_axis, audio, 'b-', alpha=0.7, linewidth=0.5)
axes[0].axvline(ntss_start, color='red', linestyle='--', label='NTSS region start')
axes[0].axvline(ntss_end, color='red', linestyle='--', label='NTSS region end')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Audio Waveform (NTSS audio placed from 0-5s)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: OLD labeling (entire region as NTSS)
axes[1].fill_between(times_old, 0, labels_old, where=(labels_old==1), 
                      color='orange', alpha=0.6, label='NTSS (class 1)')
axes[1].fill_between(times_old, 0, labels_old, where=(labels_old==0), 
                      color='white', alpha=0.6, label='Silence (class 0)')
axes[1].set_ylabel('Label')
axes[1].set_title('OLD Labeling: Entire region labeled as NTSS (INCORRECT)')
axes[1].set_ylim(-0.1, 1.5)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: NEW labeling (only actual speech as NTSS)
axes[2].fill_between(times_new, 0, labels_new, where=(labels_new==1), 
                      color='green', alpha=0.6, label='NTSS (class 1)')
axes[2].fill_between(times_new, 0, labels_new, where=(labels_new==0), 
                      color='gray', alpha=0.3, label='Silence (class 0)')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Label')
axes[2].set_title('NEW Labeling: Only actual speech labeled as NTSS (CORRECT)')
axes[2].set_ylim(-0.1, 1.5)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/labeling_fix_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to data/labeling_fix_comparison.png")

# Print statistics
print("\n" + "="*60)
print("LABELING COMPARISON STATISTICS")
print("="*60)
print(f"\nAudio duration: {duration:.2f}s")
print(f"NTSS region: {ntss_start:.2f}s to {ntss_end:.2f}s")
print(f"\nActual speech in region:")
print(f"  - Speech 1: 0.5s to 1.5s (1.0s)")
print(f"  - Speech 2: 2.5s to 4.0s (1.5s)")
print(f"  - Total speech: 2.5s / 5.0s (50%)")
print(f"  - Total silence: 2.5s / 5.0s (50%)")

old_ntss_pct = np.sum(labels_old == 1) / len(labels_old) * 100
new_ntss_pct = np.sum(labels_new == 1) / len(labels_new) * 100

print(f"\nOLD labeling (INCORRECT):")
print(f"  - Labeled as NTSS: {old_ntss_pct:.1f}%")
print(f"  - Labeled as silence: {100-old_ntss_pct:.1f}%")
print(f"  - ERROR: {old_ntss_pct - 50:.1f}% over-labeled as NTSS")

print(f"\nNEW labeling (CORRECT):")
print(f"  - Labeled as NTSS: {new_ntss_pct:.1f}%")
print(f"  - Labeled as silence: {100-new_ntss_pct:.1f}%")
print(f"  - ACCURACY: Matches actual speech distribution")

print("\n" + "="*60)
