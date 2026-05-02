# Non-Target Overlap Training Guide

## Overview
The autoencoder training script (`train_dvector_autoencoder.py`) has been enhanced to support **discriminative training** using non-target overlap samples as negative examples.

## What Are Non-Target Overlaps?
- **Full overlap samples**: Audio with main speaker + another speaker → Model should predict main speaker's d-vector
- **Non-target overlap samples**: Audio with 2 other speakers (no main speaker) → Model should predict a different representation

This creates a discrimination task that helps the model distinguish whether the main speaker is present in overlapped audio.

## Configuration Parameters

### Non-Target Settings (Lines 36-39)
```python
NON_TARGET_OVERLAP_SAMPLES_DIR = 'test_outputs/non_target_overlap_samples'
INCLUDE_NON_TARGET_OVERLAPS = True  # Enable/disable non-target training
N_NON_TARGET_OVERLAP_SAMPLES = 50   # Number of non-target samples to use
NON_TARGET_STRATEGY = 'zero'        # Target representation strategy
```

### Three Training Strategies

#### 1. 'zero' Strategy
- **Target**: Zero vector (all zeros)
- **Use case**: Simplest approach - model learns to output zeros when main speaker is absent
- **Interpretation**: Binary presence/absence signal
- **Model behavior**: Should produce near-zero output for non-target overlaps

#### 2. 'mean' Strategy  
- **Target**: Mean of all other speakers' d-vectors
- **Use case**: Semantic discrimination - model learns a generic "not main speaker" representation
- **Interpretation**: Represents the average other speaker
- **Model behavior**: Should produce the other speakers' mean when main is absent
- **Note**: Requires extraction of other speakers' clean d-vectors

#### 3. 'random' Strategy
- **Target**: Random normalized vector
- **Use case**: Adversarial-like training - model learns to reject non-target
- **Interpretation**: Teaches model to output unpredictable noise for non-target
- **Model behavior**: Should produce random-like output for non-target overlaps

## Architecture Changes

### 1. Configuration Section (Lines 36-39)
Added 4 new parameters to control non-target training behavior.

### 2. Extraction Functions (Lines 309-398)
Added two key functions:

**`extract_non_target_overlap_dvectors()`**:
- Loads non-target overlap samples from metadata
- Extracts d-vectors using voice encoder
- Handles `overlapping_speakers` and `is_target_present` fields
- Returns dict of `{sample_key: {'dvector': array, 'audio': array}}`

**`create_non_target_representation()`**:
- Creates target vector based on strategy ('zero', 'mean', or 'random')
- For 'mean': computes normalized mean of other speakers' d-vectors
- For 'zero': returns zero vector of correct dimension
- For 'random': returns normalized random vector

### 3. Dataset Class (Lines 145-193)
**`DvectorPairDataset`** now supports mixed training:

**Constructor parameters**:
- `overlap_dvectors`: Full overlap samples (main + other)
- `clean_dvectors`: Clean main speaker samples
- `speaker_pairs`: List of (overlap_key, clean_key) tuples
- `non_target_dvectors`: Non-target overlap samples (other1 + other2)
- `non_target_pairs`: List of (non_target_key, None) tuples
- `non_target_target`: Single target vector for all non-target samples

**Data handling**:
- Combines full overlap and non-target pairs into `all_pairs`
- Uses `pair[1] is None` as sentinel to distinguish sample types
- Returns appropriate target based on sample type

### 4. Main Training Pipeline (Lines 750-920)
Enhanced to conditionally extract and integrate non-target samples:

**Steps**:
1. Print configuration including non-target settings if enabled
2. Extract full overlap and clean main speaker d-vectors (existing)
3. **NEW**: If `INCLUDE_NON_TARGET_OVERLAPS = True`:
   - Extract non-target overlap d-vectors
   - For 'mean' strategy: extract other speakers' clean d-vectors
   - Create target representation based on strategy
   - Create non-target pairs: `[(key, None) for key in non_target_dvectors]`
4. Split both full overlap and non-target pairs into train/val/test
5. Compute target representation for dataset instantiation
6. Create datasets with non-target support
7. Train model on mixed positive/negative examples

## Expected Behavior

### Training Objective
- **Full overlap samples**: Minimize `||predicted - main_speaker_dvector||`
- **Non-target samples**: Minimize `||predicted - target_representation||`

Where `target_representation` is:
- Zero vector (strategy='zero')
- Mean of other speakers (strategy='mean')  
- Random normalized vector (strategy='random')

### Model Improvements
Training with non-target overlaps should improve:
1. **Discrimination**: Model learns whether main speaker is present
2. **Robustness**: Better performance on real-world scenarios where main speaker may be absent
3. **Confidence**: Output magnitude/similarity can indicate presence confidence

### Evaluation Metrics
The model will be evaluated on:
- **Full overlap reconstruction**: Cosine similarity between predicted and actual main speaker d-vector
- **Non-target rejection**: Cosine similarity between predicted and target representation (should be high)
- **Separation**: Distance between full overlap predictions and non-target predictions (should be large)

## Usage

### Enable Non-Target Training
```python
INCLUDE_NON_TARGET_OVERLAPS = True
NON_TARGET_STRATEGY = 'zero'  # or 'mean' or 'random'
N_NON_TARGET_OVERLAP_SAMPLES = 50
```

### Disable Non-Target Training (Original Behavior)
```python
INCLUDE_NON_TARGET_OVERLAPS = False
```

### Generate Non-Target Samples First
Before training with non-target overlaps, ensure you have generated the samples:
```bash
python generate_non_target_overlap_utterances.py
```

This will create samples in `test_outputs/non_target_overlap_samples/` with metadata containing:
- `overlapping_speakers`: List of speaker IDs in the overlap
- `is_target_present`: False (always for non-target)
- `target_speaker`: The main speaker ID (for reference)

## Example Training Output

```
📋 Configuration:
  LibriSpeech: ../../data/LibriSpeech
  Overlap samples: test_outputs/full_overlap_samples
  Non-target overlap samples: test_outputs/non_target_overlap_samples
  Non-target strategy: zero
  Non-target samples per split: 50
  ...
  
🎯 Training objective: Overlap (main + other) -> Clean main speaker 84
   + Non-target overlaps (other1 + other2) -> zero representation

✓ Clean main speaker d-vectors: 50

📦 Extracting non-target overlap samples...
✓ Non-target overlap d-vectors: 50
✓ Created 50 non-target training pairs

✓ Full overlap pairs: 200

📊 Dataset split:
  Train: 140 full overlap pairs + 35 non-target pairs
  Val: 30 full overlap pairs + 7 non-target pairs
  Test: 30 full overlap pairs + 8 non-target pairs
```

## Implementation Status

✅ **Complete**:
- Configuration parameters added
- Non-target extraction functions implemented
- Dataset class supports mixed training
- Main pipeline integrates non-target samples
- Three training strategies implemented

⏳ **Recommended Future Enhancements**:
- Separate evaluation metrics for full overlap vs non-target performance
- Visualization of prediction distributions for both sample types
- Confidence scoring based on output similarity to zero/mean
- Adaptive strategy selection based on validation performance

## Files Modified
- `train_dvector_autoencoder.py`: Enhanced with non-target training support
- `dvector_utils.py`: Updated metadata format detection for non-target samples
- `visualize_bottleneck_tsne.py`: Already supports non-target visualization

## Testing
To test the implementation:
1. Generate non-target samples: `python generate_non_target_overlap_utterances.py`
2. Run training with non-target enabled: `python train_dvector_autoencoder.py`
3. Compare model performance with/without non-target training
4. Visualize results: `python visualize_bottleneck_tsne.py` with `INCLUDE_NON_TARGET_OVERLAPS = True`
