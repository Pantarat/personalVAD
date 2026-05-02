#!/usr/bin/env python
"""Test alignment parsing logic."""

# Test case 1: Simple alignment with speech and silence
aligned_text = "W,W,,W,W"
stamps = ["0.5", "1.0", "1.5", "2.0", "2.5"]

print("Test 1: Simple alignment")
print(f"aligned_text: {aligned_text}")
print(f"stamps: {stamps}")

segments = aligned_text.split(',')
print(f"segments: {segments}")

speech_ranges = []
prev_stamp = 0.0

for i, segment in enumerate(segments):
    is_speech = (segment.strip() in ['W'] or 
                (segment.strip() != '' and segment.strip() != '$'))
    
    segment_end = float(stamps[i])
    segment_start = prev_stamp
    
    print(f"  Segment {i}: '{segment}' -> is_speech={is_speech}, range=[{segment_start:.2f}, {segment_end:.2f}]")
    
    if is_speech:
        speech_ranges.append((segment_start, segment_end))
    
    prev_stamp = segment_end

print(f"\nSpeech ranges: {speech_ranges}")
print(f"Expected: [(0.0, 0.5), (0.5, 1.0), (1.5, 2.0), (2.0, 2.5)]")
print(f"Match: {speech_ranges == [(0.0, 0.5), (0.5, 1.0), (1.5, 2.0), (2.0, 2.5)]}")

# Test case 2: All speech
print("\n" + "="*60)
print("Test 2: All speech (no silence)")
aligned_text2 = "W,W,W"
stamps2 = ["0.5", "1.0", "1.5"]

segments2 = aligned_text2.split(',')
speech_ranges2 = []
prev_stamp = 0.0

for i, segment in enumerate(segments2):
    is_speech = (segment.strip() in ['W'] or 
                (segment.strip() != '' and segment.strip() != '$'))
    
    segment_end = float(stamps2[i])
    segment_start = prev_stamp
    
    if is_speech:
        speech_ranges2.append((segment_start, segment_end))
    
    prev_stamp = segment_end

print(f"Speech ranges: {speech_ranges2}")
print(f"Expected: [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]")
print(f"Match: {speech_ranges2 == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]}")

# Test case 3: Leading and trailing silence
print("\n" + "="*60)
print("Test 3: Leading and trailing silence")
aligned_text3 = ",,W,W,,"
stamps3 = ["0.2", "0.4", "0.9", "1.4", "1.6", "1.8"]

segments3 = aligned_text3.split(',')
speech_ranges3 = []
prev_stamp = 0.0

for i, segment in enumerate(segments3):
    is_speech = (segment.strip() in ['W'] or 
                (segment.strip() != '' and segment.strip() != '$'))
    
    segment_end = float(stamps3[i])
    segment_start = prev_stamp
    
    print(f"  Segment {i}: '{segment}' -> is_speech={is_speech}, range=[{segment_start:.2f}, {segment_end:.2f}]")
    
    if is_speech:
        speech_ranges3.append((segment_start, segment_end))
    
    prev_stamp = segment_end

print(f"Speech ranges: {speech_ranges3}")
print(f"Expected: [(0.4, 0.9), (0.9, 1.4)]  # Only segments 2 and 3")
print(f"Match: {speech_ranges3 == [(0.4, 0.9), (0.9, 1.4)]}")

print("\n" + "="*60)
if all([
    speech_ranges == [(0.0, 0.5), (0.5, 1.0), (1.5, 2.0), (2.0, 2.5)],
    speech_ranges2 == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)],
    speech_ranges3 == [(0.4, 0.9), (0.9, 1.4)]
]):
    print("✓ All tests PASSED!")
else:
    print("❌ Some tests FAILED!")
