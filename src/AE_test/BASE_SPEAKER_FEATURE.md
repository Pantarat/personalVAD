# Base Speaker Feature for Overlap Generation

## Overview
Added a new feature to `generate_non_target_overlap_utterances.py` that allows specifying a **base speaker** who will be present in ALL generated overlap samples.

## What Changed

### 1. Configuration Parameter
```python
# New parameter at top of script
BASE_SPEAKER = None  # Set to speaker ID (e.g., '84') to use as base for all overlaps
```

### 2. Behavior

**Without base speaker (original behavior):**
- Generates random combinations of speakers
- Example: (174, 251), (174, 422), (251, 422), etc.

**With base speaker (new feature):**
- Base speaker is in ALL samples
- Other speakers are mixed on top
- Example with `BASE_SPEAKER = '84'`:
  - (84, 174), (84, 251), (84, 422), etc.
  - Speaker 84 is the "main" speaker, others are "overlapping"

### 3. Command Line Usage

```bash
# Original: Random combinations
python generate_non_target_overlap_utterances.py \
    --other-speakers 174 251 422 652 1272 \
    --n-overlapping-speakers 2

# New: Speaker 84 as base in all overlaps
python generate_non_target_overlap_utterances.py \
    --base-speaker 84 \
    --other-speakers 174 251 422 652 1272 \
    --n-overlapping-speakers 2
```

### 4. Example Script
Created `example_generate_with_base_speaker.sh` showing how to use the feature.

## Use Cases

1. **Testing specific speaker**: Want to test how model performs when speaker 84 is always the main speaker with various overlapping speakers

2. **Targeted training**: Train the model to handle a specific speaker in overlap scenarios

3. **Data analysis**: Analyze how a specific speaker's characteristics affect overlap detection

## Technical Details

- Base speaker is always loaded first (position 0 in the mix)
- Other speakers are mixed with random SNR on top of the base speaker
- Metadata includes `base_speaker` field to track which samples use this feature
- Sample names include base speaker ID for easy identification

## Metadata Structure

```json
{
  "sample_id": "sample_0001_speakers84_174_00",
  "base_speaker": "84",
  "overlapping_speakers": ["84", "174"],
  "n_speakers": 2,
  ...
}
```

## Benefits

✅ More controlled data generation  
✅ Easier to test specific scenarios  
✅ Maintains backward compatibility (default is None)  
✅ Clear naming and metadata tracking
