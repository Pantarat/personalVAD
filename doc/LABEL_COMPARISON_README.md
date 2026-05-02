# Label Comparison Feature - README

## Overview

Added functionality to compare ground truth labels with model predictions during inference in PersonalVAD.

## Files Added/Modified

### 1. `src/compare_labels_with_predictions.py` (NEW)
A comprehensive Python module for comparing ground truth labels with model predictions.

**Key Features:**
- **LabelComparator class**: Main class for label comparison
- **parse_overlap_meta()**: Parses `.overlap_meta` files from overlap dataset
- **create_ground_truth_labels()**: Generates frame-level labels from metadata
- **compute_metrics()**: Calculates accuracy, precision, recall, F1, confusion matrix
- **visualize_comparison()**: Creates 5-panel visualization:
  1. Audio waveform
  2. Ground truth labels
  3. Model predictions (smoothed)
  4. Prediction probabilities
  5. Difference visualization (mismatch in red, match in green)
- **print_metrics()**: Pretty-prints evaluation metrics

### 2. `src/personal_vad_demo.ipynb` (MODIFIED)
Added new section "Step 12a: Compare Predictions with Ground Truth Labels"

**Two comparison methods added:**

#### Method 1: Automatic from Metadata (recommended for overlap dataset)
- Automatically loads `.overlap_meta` files
- Extracts overlap segments and NTSS segments
- Creates ground truth labels
- Computes and visualizes comparison
- Shows interpretation notes

#### Method 2: Manual Label Definition
- Allows users to manually define time segments with labels
- Format: `[(start, end, label), ...]` where label: 0=NS, 1=NTSS, 2=TSS
- Useful when you know the ground truth but don't have metadata files
- Generates same comprehensive metrics and visualization

## Usage

### Method 1: Using Kaldi ARK Files (Recommended - Most Accurate)

This method loads the actual frame-level ground truth labels from your dataset's ark files.

```python
# In the demo notebook, after inference:

USE_ARK_LABELS = True

# Update path to your dataset's labels.scp file
LABELS_SCP_PATH = '../data/eval_dir/data/features_full/labels.scp'

# Utterance ID is auto-detected from audio filename
# Or manually set: UTTERANCE_ID = '100-121669-0024_6943-85172-0023'

# The comparison cell will:
# 1. Load labels from ark file using kaldiio
# 2. Match with predictions frame-by-frame
# 3. Compute metrics (accuracy, precision, recall, F1)
# 4. Generate visualization comparing GT vs Predictions
# 5. Save comparison plot as 'ark_label_comparison_result.png'
```

**Common label.scp locations:**
- `data/eval_dir/data/features_full/labels.scp` - Full dataset
- `data/eval_dir/data/features_augment_500/labels.scp` - Training data
- Your overlap dataset folder (after running feature extraction)

### Method 2: Using Overlap Metadata (Alternative)

If ark files aren't available, this creates approximate labels from metadata files.

```python
# In the demo notebook, after inference:

USE_ARK_LABELS = False  # Disable ark loading

# The code will automatically look for .overlap_meta file
META_PATH = 'path/to/audio.overlap_meta'

# The comparison cell will:
# 1. Parse overlap metadata
# 2. Create approximate frame-level labels
# 3. Compute metrics
# 4. Generate visualization
# 5. Save comparison plot as 'metadata_label_comparison_result.png'
```

### Using with Manual Labels

```python
# Set this to True
USE_MANUAL_LABELS = True

# Define your ground truth segments
manual_segments = [
    (0.0, 2.0, 0),    # 0-2s: Non-speech (NS)
    (2.0, 5.5, 2),    # 2-5.5s: Target speaker (TSS)
    (5.5, 6.0, 0),    # 5.5-6s: Non-speech
    (6.0, 10.0, 1),   # 6-10s: Non-target speaker (NTSS)
    (10.0, 15.0, 2),  # 10-15s: Target speaker
]

# Run the cell - it will compute all metrics and visualize
```

## Metrics Provided

### Overall Metrics
- **Accuracy**: Frame-level accuracy across all classes
- **Mismatch Percentage**: Percentage of frames where prediction ≠ ground truth

### Per-Class Metrics
For each class (NS, NTSS, TSS):
- **Precision**: Of frames predicted as this class, how many were correct
- **Recall**: Of frames that are this class, how many were detected
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of ground truth frames for this class

### Confusion Matrix
Shows how predictions are distributed across actual classes
- Rows = Actual (ground truth)
- Columns = Predicted
- Diagonal = correct predictions

## Visualization Outputs

### 5-Panel Comparison Plot
1. **Audio Waveform**: Shows audio amplitude over time
2. **Ground Truth**: Color-coded ground truth labels (gray=NS, orange=NTSS, green=TSS)
3. **Predictions**: Color-coded model predictions (same colors)
4. **Probabilities**: Line plot showing prediction confidence per class
5. **Differences**: Red=mismatch, Green=match

Saved as: `label_comparison_result.png` (metadata method) or `manual_label_comparison.png` (manual method)

## Interpretation Notes

**Important considerations:**

1. **Metadata-based labels are simplified**
   - Based on segment timing, not word-level alignments
   - Overlap regions marked as TSS (target present during overlap)
   - Some mismatches expected at boundaries

2. **Frame-level granularity**
   - Each frame = 10ms (160 samples at 16kHz)
   - Smoothing applied to predictions (5-frame median filter)

3. **Expected behavior**
   - Model uses actual audio features
   - Ground truth uses segment/metadata timing
   - Mismatches at silence/speech boundaries are normal
   - High mismatch in overlap regions is expected (model detects interference)

## Example Output

```
📊 EVALUATION METRICS: Ground Truth vs Model Predictions
======================================================================

🎯 Overall Accuracy: 87.34%

📈 Per-Class Metrics:
Class      Precision    Recall       F1-Score     Support   
----------------------------------------------------------------------
NS             89.23%      85.67%      87.41%        1234
NTSS           78.56%      82.34%      80.41%         876
TSS            91.45%      93.12%      92.28%        2345

🔢 Confusion Matrix:
         Predicted ->
         NS       NTSS     TSS     
Actual ↓
NS          1057     123      54  
NTSS         145     721      10  
TSS          112      48    2185  
```

## Requirements

The comparison feature requires:
- `numpy`
- `matplotlib`
- `scikit-learn` (for metrics)
- `scipy` (for median filtering)

All dependencies should already be available if you can run PersonalVAD.

## Future Enhancements

Possible improvements:
1. Load precise word-level alignments from LibriSpeech
2. Support for other ground truth formats (Kaldi labels.scp, etc.)
3. Time-weighted metrics (weight by segment duration)
4. Per-segment error analysis
5. Export metrics to CSV for batch evaluation
