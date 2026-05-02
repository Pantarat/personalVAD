# Labeling Fix: NTSS Speech vs Silence Detection

## Problem Description

The PersonalVAD overlap generation was incorrectly labeling regions:
1. **Silent regions labeled as NTSS (class 1)**: When non-target speaker audio contained leading/trailing silence, those silent portions were labeled as non-target speech instead of silence (class 0)
2. **Speech regions labeled as silence (class 0)**: In some cases, actual non-target speaker speech was being labeled as silence

## Root Cause

The original labeling logic assigned labels based on **time ranges** where NTSS audio was placed, without checking whether that audio actually contained speech or silence at each point. This meant:

- When NTSS audio files with leading/trailing silence were added, the **entire region** was labeled as NTSS (class 1), including the silent parts
- The alignment timestamps from LibriSpeech were not consistently accounting for silence padding added during generation

## Solution Implemented

### 1. Energy-Based Speech Detection Function

Added `detect_speech_regions()` function that:
- Analyzes audio using frame-based energy analysis (20ms frames with 50% overlap)
- Uses adaptive thresholding (40th percentile by default) to distinguish speech from silence
- Merges nearby speech regions (within 100ms) to avoid fragmentation
- Filters out very short regions (< 50ms) to reduce false positives
- Returns precise sample-level boundaries of actual speech regions

**Key Parameters:**
- `frame_len_ms=20`: Frame size for energy calculation (20ms is standard for speech)
- `energy_threshold_percentile=40`: Adaptive threshold that adjusts to audio characteristics
- Minimum energy threshold of 1e-6 to avoid false positives on pure noise

### 2. Updated Overlap Speech Range Detection

Changed from alignment-based to energy-based detection:

**Before:**
```python
# Used alignment data which might include silence
for alignment_data in alignment_data_list:
    aligned_text = alignment_data['text']
    stamps = alignment_data['stamps']
    # Mark entire aligned regions as speech
```

**After:**
```python
# Extract actual audio and detect speech regions
overlap_audio = final_audio[ov_start_sample:ov_end_sample]
speech_regions_samples = detect_speech_regions(overlap_audio, sr=sr)
# Only mark regions with actual speech energy
```

### 3. Updated Standalone NTSS Range Detection

Applied the same energy-based detection to standalone NTSS segments:

**Before:**
```python
# Mark entire NTSS segment as non-target speech
ntss_only_ranges.append((ntss['start'], ntss['end']))
```

**After:**
```python
# Extract audio and detect only actual speech
ntss_audio = final_audio[ntss_start_sample:ntss_end_sample]
speech_regions_samples = detect_speech_regions(ntss_audio, sr=sr)
# Add only regions that contain actual speech
```

## Results

The fix ensures:
- ✓ Silent portions (leading/trailing silence, gaps) are correctly labeled as class 0 (NS)
- ✓ Non-target speaker speech is correctly labeled as class 1 (NTSS)
- ✓ Target speaker speech continues to be correctly labeled as class 2 (TSS)
- ✓ More accurate training data for the PersonalVAD model
- ✓ Better distinction between silence and speech in evaluation

## Testing

Created `test_speech_detection.py` with 5 test cases:
1. Pure silence detection (0 regions expected) - ✓ PASS
2. Continuous speech detection (1 region expected) - ✓ PASS
3. Speech-Silence-Speech pattern (2 regions expected) - ✓ PASS
4. Silence-Speech-Silence pattern (1 region in middle) - ✓ PASS
5. Very quiet speech with adjusted threshold - ✓ PASS

## Files Modified

1. **src/generate_overlapping_utterances.py**
   - Added `detect_speech_regions()` function (lines ~56-153)
   - Updated overlap speech range detection (lines ~1010-1021)
   - Updated standalone NTSS range detection (lines ~1047-1072)

2. **test_speech_detection.py** (new)
   - Comprehensive test suite for speech detection function

## Usage

The fix is automatically applied when generating overlapping utterances:
```bash
python src/generate_overlapping_utterances.py \
    --libri_root /path/to/LibriSpeech \
    --concat_dir data/overlap_50pct_main84_1000 \
    --count 1000 \
    --overlap_pct 50 \
    --base-speaker 84
```

No parameter changes needed - the improved labeling is automatic.

## Future Improvements

Potential enhancements:
1. Use VAD (Voice Activity Detection) model instead of energy-based detection
2. Add configurable energy threshold parameter for different acoustic conditions
3. Consider using spectral features (e.g., zero-crossing rate) in addition to energy
4. Log statistics about how much silence was detected in NTSS segments
