# Fix for NTSS at End Being Labeled as Silence

## Issue
Non-target speaker speech (NTSS) at the end of audio files was being labeled as silence (class 0) instead of NTSS (class 1).

## Root Cause
The energy threshold in `detect_speech_regions()` was too high (40th percentile), causing the function to miss quieter speech, especially:
1. Speech at the end of audio files (often has trailing characteristics)
2. Quieter utterances or speakers
3. Speech mixed at lower amplitudes

## Solution
Adjusted two key parameters in `detect_speech_regions()`:

### 1. Lowered Energy Threshold Percentile
**Before:** `energy_threshold_percentile=40` (40th percentile)  
**After:** `energy_threshold_percentile=25` (25th percentile)

This makes the detection more sensitive - it now considers frames in the lower 25% of energy as potential silence, rather than the lower 40%. This means more frames with actual speech energy are correctly identified as speech.

### 2. Lowered Minimum Energy Threshold  
**Before:** `min_threshold = 1e-6`  
**After:** `min_threshold = 5e-7`

This catches even quieter speech that might be mixed at lower volumes or naturally quieter speakers.

## Changes Made

**File:** `src/generate_overlapping_utterances.py`

```python
# Line ~58: Function signature
def detect_speech_regions(audio, sr=16000, frame_len_ms=20, energy_threshold_percentile=25):
    # Changed from 40 to 25 ^^^

# Line ~107: Minimum threshold
min_threshold = 5e-7  # Lowered from 1e-6 to catch quieter speech
```

## Verification

Added test case `test_speech_at_end()` that specifically checks detection of speech at the very end of audio:
- Creates 3 seconds of audio with speech only in last 0.5 seconds
- Verifies speech is detected and extends near the end (>2.4s)
- **Result:** ✓ PASS

All 6 tests passing:
1. ✓ Pure silence detection
2. ✓ Continuous speech detection  
3. ✓ Speech-Silence-Speech pattern
4. ✓ Silence-Speech-Silence pattern
5. ✓ Very quiet speech
6. ✓ **Speech at end** (new test)

## Impact

- **Before:** NTSS speech at end often labeled as class 0 (silence)
- **After:** NTSS speech correctly labeled as class 1 (NTSS)
- More accurate training labels, especially for:
  - End regions
  - Quieter speakers
  - Lower-amplitude mixed audio

## Regenerating Dataset

To apply this fix to your data, regenerate with:

```bash
python src/generate_overlapping_utterances.py \
    --libri_root /path/to/LibriSpeech \
    --concat_dir data/overlap_50pct_fixed \
    --count 1000 \
    --overlap_pct 50 \
    --base-speaker 84
```

The improved speech detection will automatically apply.

## Technical Details

**Energy-based Detection Algorithm:**
1. Divide audio into 20ms frames with 50% overlap
2. Calculate mean squared energy for each frame
3. Set threshold at 25th percentile of frame energies (adaptive)
4. Apply minimum threshold of 5e-7 to avoid false positives on noise
5. Mark frames above threshold as speech
6. Merge nearby speech regions (<100ms gap)
7. Filter out very short regions (<50ms)

**Why 25th Percentile Works:**
- Most silence/background noise falls below 25th percentile
- Actual speech (even quiet) typically has higher energy
- Adapts to each audio file's characteristics
- Conservative enough to avoid labeling pure silence as speech
- Sensitive enough to catch real speech at boundaries

## Compatibility

This fix is backward compatible:
- Same function signature
- Same return format
- Just more accurate detection
- No changes needed to calling code
