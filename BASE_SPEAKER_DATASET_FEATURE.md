# Base Speaker Feature for Overlap Dataset Generation

## Overview
Added base speaker functionality to the overlap dataset generation pipeline. You can now specify a fixed target speaker (e.g., speaker 84) who will be the main speaker in ALL generated overlap samples.

## Modified Files

### 1. `src/generate_overlapping_utterances.py`
**Changes:**
- Added `--base-speaker` command line argument
- Modified speaker selection logic to use specified base speaker
- Updated function signature and documentation

**New Parameter:**
```python
--base-speaker SPEAKER_ID
```
Example: `--base-speaker 84` to use speaker 84 as target in all samples

### 2. `prepare_overlap_dataset.sh`
**Changes:**
- Added `BASE_SPEAKER` configuration variable
- Updated console output to show base speaker when used
- Modified Python call to include base speaker parameter

**New Configuration:**
```bash
# Set to speaker ID to use as base, or leave empty for random selection
BASE_SPEAKER=""  # Example: BASE_SPEAKER="84"
```

## Usage

### Method 1: Edit the Shell Script Directly

Open `prepare_overlap_dataset.sh` and set:
```bash
BASE_SPEAKER="84"  # Or any other speaker ID
```

Then run normally:
```bash
bash prepare_overlap_dataset.sh 0
```

### Method 2: Use Python Script Directly

```bash
python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir data/overlap_with_84 \
    --count 50 \
    --overlap_pct 50 \
    --base-speaker 84 \
    dev-clean dev-other
```

### Method 3: Use Example Script

```bash
bash example_prepare_with_base_speaker.sh
```

## Behavior Comparison

### Without Base Speaker (Original)
```bash
BASE_SPEAKER=""
```
- Random speaker selection for each utterance
- Maximum variety in target speakers
- Example outputs:
  - Utterance 1: Speaker 174 as target
  - Utterance 2: Speaker 251 as target
  - Utterance 3: Speaker 422 as target

### With Base Speaker (New Feature)
```bash
BASE_SPEAKER="84"
```
- Speaker 84 is target in ALL utterances
- Other speakers are mixed as overlaps
- Example outputs:
  - Utterance 1: Speaker 84 (base) + Speaker 174 (overlap)
  - Utterance 2: Speaker 84 (base) + Speaker 251 (overlap)
  - Utterance 3: Speaker 84 (base) + Speaker 422 (overlap)

## Use Cases

### 1. **PersonalVAD Training for Specific User**
Train a model specifically for one target user (e.g., your voice):
```bash
BASE_SPEAKER="84"  # Your speaker ID
```

### 2. **Testing Model Performance on Specific Speaker**
Evaluate how well the model handles a particular speaker with various overlaps:
```bash
BASE_SPEAKER="1673"
OVERLAP_PERCENTAGE=50
```

### 3. **Targeted Data Augmentation**
Generate additional training data for a speaker that needs more samples:
```bash
BASE_SPEAKER="422"
utt_count=100
```

### 4. **Comparative Analysis**
Compare model performance across different speakers:
```bash
# Run 1: Speaker 84 as base
BASE_SPEAKER="84"
feature_dir_name=overlap_50pct_50_base84

# Run 2: Speaker 1673 as base  
BASE_SPEAKER="1673"
feature_dir_name=overlap_50pct_50_base1673
```

## Technical Details

### Speaker Selection Logic

**Without base speaker:**
```python
main_speakers = random.sample(dataset, n_main_speakers)
```

**With base speaker:**
```python
base_speaker_data = [spk for spk in dataset if spk[0].split('-')[0] == base_speaker]
main_speakers = [base_speaker_data[0]]
```

### Utterance Exhaustion Handling

- **Random mode**: Removes speakers from pool when exhausted
- **Base speaker mode**: Continues until base speaker's utterances are exhausted
- Script prints message: "Base speaker {ID} has no more utterances, ending..."

### Output Structure

Same structure as original, labels still track:
- TSS (Target Speaker Speech): When base speaker is talking
- NTSS (Non-Target Speaker Speech): When other speakers are talking
- Overlaps: When base + others are talking simultaneously

## Configuration Tips

### For Maximum Overlap Challenge
```bash
BASE_SPEAKER="84"
OVERLAP_PERCENTAGE=80  # 80% of target speech has overlap
OVERLAP_AMPLITUDE=1.2  # Overlap louder than target
```

### For Realistic Conversation
```bash
BASE_SPEAKER="84"
OVERLAP_PERCENTAGE=30  # 30% overlap (natural)
OVERLAP_AMPLITUDE=0.7  # Overlap quieter than target
```

### For Testing Edge Cases
```bash
BASE_SPEAKER="84"
OVERLAP_PERCENTAGE=100  # Complete overlap
OVERLAP_AMPLITUDE=1.0  # Equal volume
```

## Verification

After generation, check the metadata:
```bash
# Count utterances per speaker in wav.scp
cut -d'-' -f1 data/overlap/wav.scp | sort | uniq -c

# With base speaker, you should see speaker 84 in ALL entries
```

## Backward Compatibility

✅ **Fully backward compatible**
- Default `BASE_SPEAKER=""` maintains original random behavior
- Existing scripts work without modification
- No changes to output format or file structure

## Common Issues

### Issue: "Base speaker has no more utterances"
**Cause:** Requested more utterances than speaker has available
**Solution:** Reduce `utt_count` or use different speaker with more utterances

### Issue: Speaker ID not found
**Cause:** Speaker ID doesn't exist in specified LibriSpeech folders
**Solution:** Check available speakers in `data/LibriSpeech/dev-clean/` or other folders

### Issue: No overlaps generated
**Cause:** Not enough other speakers available for mixing
**Solution:** Add more LibriSpeech folders to `libri_folders` list

## Examples

### Example 1: Generate 100 samples with speaker 84
```bash
# Edit prepare_overlap_dataset.sh
BASE_SPEAKER="84"
utt_count=100
OVERLAP_PERCENTAGE=50

# Run
bash prepare_overlap_dataset.sh 0
```

### Example 2: Generate test set for evaluation
```bash
python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir data/test_overlap_84 \
    --count 20 \
    --overlap_pct 50 \
    --base-speaker 84 \
    test-clean
```

### Example 3: Compare two speakers
```bash
# Speaker 84
python src/generate_overlapping_utterances.py \
    --concat_dir data/overlap_speaker84 \
    --base-speaker 84 \
    --count 50 \
    --libri_root data/LibriSpeech \
    dev-clean

# Speaker 1673
python src/generate_overlapping_utterances.py \
    --concat_dir data/overlap_speaker1673 \
    --base-speaker 1673 \
    --count 50 \
    --libri_root data/LibriSpeech \
    dev-clean
```

## Related Files
- `src/generate_overlapping_utterances.py` - Main generation script
- `prepare_overlap_dataset.sh` - Shell wrapper script
- `example_prepare_with_base_speaker.sh` - Example usage
- `src/AE_test/generate_non_target_overlap_utterances.py` - Similar feature for non-target overlaps
