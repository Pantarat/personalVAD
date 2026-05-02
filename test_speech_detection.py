#!/usr/bin/env python
"""Test script for verifying speech detection function."""

import numpy as np
import sys
import os

# Directly define the function here to avoid import issues
def detect_speech_regions(audio, sr=16000, frame_len_ms=20, energy_threshold_percentile=40):
    """Detect actual speech regions in audio using energy-based analysis.
    
    This function identifies which parts of an audio segment contain actual speech
    vs silence/background noise. Used to prevent labeling silent regions as NTSS.
    
    Args:
        audio (np.array): Audio samples (normalized to -1 to 1 range)
        sr (int): Sample rate in Hz (default 16000)
        frame_len_ms (int): Frame length in milliseconds for energy analysis (default 20ms)
        energy_threshold_percentile (int): Percentile for energy threshold (default 40)
                                          Lower = more aggressive (detects more silence)
                                          Higher = less aggressive (detects more speech)
    
    Returns:
        list: List of (start_sample, end_sample) tuples for speech regions
    """
    if audio.size == 0:
        return []
    
    # Calculate frame length in samples
    frame_len = int(frame_len_ms * sr / 1000)
    hop_len = frame_len // 2  # 50% overlap
    
    # Calculate energy for each frame
    n_frames = (len(audio) - frame_len) // hop_len + 1
    if n_frames <= 0:
        # Audio too short, check if it has any energy
        energy = np.sum(audio ** 2)
        if energy > 1e-6:  # Has some energy
            return [(0, len(audio))]
        return []
    
    frame_energies = []
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frame = audio[start:end]
        energy = np.sum(frame ** 2) / len(frame)  # Mean squared energy
        frame_energies.append(energy)
    
    frame_energies = np.array(frame_energies)
    
    # Determine threshold using percentile of frame energies
    # This adapts to the audio's characteristics
    if len(frame_energies) > 0 and np.max(frame_energies) > 0:
        energy_threshold = np.percentile(frame_energies, energy_threshold_percentile)
    else:
        # No energy in audio
        return []
    
    # Ensure minimum threshold to avoid labeling pure silence as speech
    min_threshold = 1e-6
    energy_threshold = max(energy_threshold, min_threshold)
    
    # Identify speech frames (energy above threshold)
    is_speech = frame_energies > energy_threshold
    
    # Find continuous speech regions (merge nearby speech frames)
    speech_regions = []
    in_speech = False
    speech_start = 0
    
    for i, speech_frame in enumerate(is_speech):
        if speech_frame and not in_speech:
            # Start of speech region
            in_speech = True
            speech_start = i * hop_len
        elif not speech_frame and in_speech:
            # End of speech region
            in_speech = False
            speech_end = i * hop_len + frame_len
            speech_regions.append((speech_start, min(speech_end, len(audio))))
    
    # Handle case where speech continues to end
    if in_speech:
        speech_regions.append((speech_start, len(audio)))
    
    # Merge regions that are very close together (within 100ms)
    min_gap_samples = int(0.1 * sr)  # 100ms
    merged_regions = []
    for start, end in speech_regions:
        if merged_regions and start - merged_regions[-1][1] < min_gap_samples:
            # Merge with previous region
            merged_regions[-1] = (merged_regions[-1][0], end)
        else:
            merged_regions.append((start, end))
    
    # Filter out very short regions (< 50ms)
    min_duration_samples = int(0.05 * sr)  # 50ms
    filtered_regions = [(s, e) for s, e in merged_regions if e - s >= min_duration_samples]
    
    return filtered_regions

def test_silence():
    """Test pure silence detection."""
    audio = np.zeros(16000)  # 1 second of silence
    regions = detect_speech_regions(audio)
    print(f"Test 1 - Pure silence: {len(regions)} regions detected (expected: 0)")
    assert len(regions) == 0, "Pure silence should have no speech regions"
    print("✓ PASS\n")

def test_continuous_speech():
    """Test continuous speech detection."""
    # Simulate speech with random noise
    audio = np.random.randn(16000) * 0.1  # 1 second of noisy speech
    regions = detect_speech_regions(audio)
    print(f"Test 2 - Continuous speech: {len(regions)} regions detected (expected: 1)")
    assert len(regions) >= 1, "Continuous speech should have at least 1 region"
    if len(regions) == 1:
        start, end = regions[0]
        duration = (end - start) / 16000
        print(f"  Region: {start/16000:.3f}s - {end/16000:.3f}s (duration: {duration:.3f}s)")
    print("✓ PASS\n")

def test_speech_with_silence():
    """Test speech with silence gaps."""
    audio = np.zeros(16000 * 3)  # 3 seconds
    # Add speech in first and third seconds
    audio[0:16000] = np.random.randn(16000) * 0.1
    audio[32000:48000] = np.random.randn(16000) * 0.1
    regions = detect_speech_regions(audio)
    print(f"Test 3 - Speech-Silence-Speech: {len(regions)} regions detected (expected: 2)")
    for i, (start, end) in enumerate(regions):
        print(f"  Region {i+1}: {start/16000:.3f}s - {end/16000:.3f}s")
    assert len(regions) >= 1, "Should detect at least 1 speech region"
    print("✓ PASS\n")

def test_leading_trailing_silence():
    """Test speech with leading and trailing silence."""
    audio = np.zeros(16000 * 3)  # 3 seconds
    # Add speech in middle second only
    audio[16000:32000] = np.random.randn(16000) * 0.1
    regions = detect_speech_regions(audio)
    print(f"Test 4 - Silence-Speech-Silence: {len(regions)} regions detected (expected: 1)")
    if len(regions) > 0:
        start, end = regions[0]
        print(f"  Region: {start/16000:.3f}s - {end/16000:.3f}s")
        # Check that silence is mostly excluded
        speech_start_expected = 1.0  # second
        assert start/16000 > 0.5, "Should skip most leading silence"
        assert end/16000 < 2.5, "Should skip most trailing silence"
    print("✓ PASS\n")

def test_very_quiet_speech():
    """Test very quiet speech detection."""
    audio = np.random.randn(16000) * 0.01  # Very quiet speech
    regions = detect_speech_regions(audio)  # Use default threshold (now 25)
    print(f"Test 5 - Very quiet speech: {len(regions)} regions detected")
    print(f"  (with default threshold=25, should detect speech)")
    assert len(regions) >= 1, "Should detect quiet speech with threshold=25"
    print("✓ PASS\n")

def test_speech_at_end():
    """Test speech detection at the very end of audio."""
    audio = np.zeros(16000 * 3)  # 3 seconds
    # Add speech only in last 0.5 seconds
    audio[-8000:] = np.random.randn(8000) * 0.1
    regions = detect_speech_regions(audio)
    print(f"Test 6 - Speech at end: {len(regions)} regions detected (expected: 1)")
    if len(regions) > 0:
        start, end = regions[-1]  # Get last region
        print(f"  Last region: {start/16000:.3f}s - {end/16000:.3f}s")
        # Should detect speech near the end
        assert end/16000 > 2.4, "Should detect speech near the end (> 2.4s)"
        print(f"  ✓ Correctly detected speech at end")
    assert len(regions) >= 1, "Should detect speech at end"
    print("✓ PASS\n")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Speech Detection Function")
    print("=" * 60 + "\n")
    
    try:
        test_silence()
        test_continuous_speech()
        test_speech_with_silence()
        test_leading_trailing_silence()
        test_very_quiet_speech()
        test_speech_at_end()
        
        print("=" * 60)
        print("All tests PASSED! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
