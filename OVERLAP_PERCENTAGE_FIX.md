# Overlap Percentage Fix

## Problem
When generating datasets with `OVERLAP_PERCENTAGE=0`, the generated audio still contained overlapping speech between the target speaker and non-target speakers.

## Root Cause
The overlap positioning logic in `generate_overlapping_utterances.py` was not respecting the `overlap_pct` parameter. It was randomly placing the non-target speaker audio, which could overlap with target speaker speech regardless of the specified overlap percentage.

## Solution
Modified the overlap positioning logic to handle `overlap_pct == 0` as a special case:

1. **For 0% overlap**: The non-target speaker audio is now placed exclusively in silence/gaps:
   - Before the first target speech segment
   - Between target speech segments
   - After the last target speech segment
   
2. **For non-zero overlap**: Original behavior maintained (random placement within target speech region)

## Changes Made
- **File**: `src/generate_overlapping_utterances.py`
- **Lines**: ~235-305 (overlap positioning logic)

### Key improvements:
1. Added logic to identify all silence regions in target audio
2. Filter silence regions large enough to fit the non-target speaker audio
3. Randomly select a valid silence region and place non-target audio within it
4. If no silence region is large enough, place after all target speech

## Testing
Use the provided test script to verify overlap percentages:

```bash
python test_overlap_percentage.py data/overlap
```

This will:
- Parse all generated samples
- Calculate actual overlap percentage for each sample
- Report statistics (avg, min, max)
- Flag any samples with unexpected overlap

## Expected Results
For `OVERLAP_PERCENTAGE=0`:
- All samples should have overlap < 1%
- Non-target speaker should only appear during target silence
- Target speaker speech should never have simultaneous non-target speech

## Regeneration Required
After applying this fix, you need to regenerate your dataset:

```bash
bash prepare_overlap_dataset.sh 0
```

Then verify with:

```bash
python test_overlap_percentage.py kaldi/egs/pvad/data/overlap  # or data/overlap if AUGMENT=false
```
