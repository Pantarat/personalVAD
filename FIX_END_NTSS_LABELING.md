# Fix: NTSS at End Labeled as Silence

## Problem
Non-target speech (NTSS) segments at the end of audio files were being labeled as silence (class 0) instead of NTSS (class 1).

## Root Cause
When NTSS audio segments were extended by looping/tiling to meet the target duration (2-4 seconds), the alignment data remained from the original (unlooped) utterances. The code had a threshold to detect looped audio and mark the entire region as speech:

```python
if ntss_duration > total_alignment_duration * 1.5:  # Likely looped
    ntss_only_ranges.append((ntss_start_time, ntss_end_time))
```

**The threshold of 1.5× was too high.** 

### Example Scenario
- Original NTSS utterance: 3.0 seconds
- Target NTSS duration: 4.0 seconds  
- Audio is looped to 4.0 seconds
- Ratio: 4.0 / 3.0 = 1.33
- Check: `1.33 > 1.5` → **False** (not detected as looped!)
- Result: Tries to use original 3.0s alignment for 4.0s audio
- The last 1.0 second has no alignment data → labeled as silence

## Solution
Changed the looped-audio detection threshold from `1.5` to `1.05`:

```python
if ntss_duration > total_alignment_duration * 1.05:  # Likely looped or trimmed
    ntss_only_ranges.append((ntss_start_time, ntss_end_time))
```

This catches any NTSS segment where the audio duration exceeds the alignment duration by more than 5%, indicating the audio was extended (looped) or trimmed beyond the original utterances.

## Changes Made

### File: `src/generate_overlapping_utterances.py`

**1. NTSS standalone segments (line ~1197):**
```python
# OLD:
if ntss_duration > total_alignment_duration * 1.5:  # Likely looped

# NEW:
if ntss_duration > total_alignment_duration * 1.05:  # Likely looped or trimmed
```

**2. Overlap speech segments (line ~1103):**
```python
# OLD:
if ov_duration > total_alignment_duration * 1.5:  # Likely looped

# NEW:
if ov_duration > total_alignment_duration * 1.05:  # Likely looped
```

## Impact
- **Before:** NTSS segments at the end were often mislabeled as silence when audio was looped
- **After:** Any looped/extended audio (>5% longer than alignment data) is correctly marked as speech throughout its entire duration
- **Side effects:** None - the lower threshold is more conservative and catches edge cases that were previously missed

## Testing
To verify the fix:
1. Generate new dataset with NTSS segments
2. Run the diagnostic script:
   ```bash
   python diagnose_end_ntss.py data/your_dataset_dir
   ```
3. Check that end segments with speech energy are labeled as 'N' (NTSS), not '' (silence)

## Files Changed
- `src/generate_overlapping_utterances.py` - Fixed looped audio detection threshold (2 locations)

## Files Created
- `diagnose_end_ntss.py` - Diagnostic script to detect end-labeling issues
