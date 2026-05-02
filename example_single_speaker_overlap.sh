#!/bin/bash
#
# Example: Create overlap dataset using single speaker dataset
#
# This demonstrates the full pipeline:
# 1. Extract single speaker and create 5s chunks
# 2. Use those chunks as target speaker in overlap dataset
#

set -e  # Exit on error

SPEAKER_ID="84"
LIBRI_ROOT="data/LibriSpeech"

echo "================================================"
echo "Step 1: Extract single speaker and create chunks"
echo "================================================"

# Extract speaker 84, create 5s chunks, split 80/20 train/test
python extract_single_speaker.py \
    --speaker_id $SPEAKER_ID \
    --libri_root $LIBRI_ROOT \
    --output_dir data/speaker_${SPEAKER_ID} \
    --chunk_duration 5.0 \
    --test_ratio 0.2

echo ""
echo "================================================"
echo "Step 2: Create overlap dataset using train chunks"
echo "================================================"

# Now use the train chunks as target speaker in overlap generation
python src/generate_overlapping_utterances.py \
    --libri_root $LIBRI_ROOT \
    --concat_dir data/overlap_speaker${SPEAKER_ID}_train \
    --count 100 \
    --overlap_pct 50 \
    --amplitude_ratio 1.0 \
    --single-speaker-dataset data/speaker_${SPEAKER_ID}/train \
    test-clean test-other

echo ""
echo "================================================"
echo "Step 3: Create overlap dataset using test chunks"
echo "================================================"

# Use test chunks for evaluation set
python src/generate_overlapping_utterances.py \
    --libri_root $LIBRI_ROOT \
    --concat_dir data/overlap_speaker${SPEAKER_ID}_test \
    --count 20 \
    --overlap_pct 50 \
    --amplitude_ratio 1.0 \
    --single-speaker-dataset data/speaker_${SPEAKER_ID}/test \
    test-clean test-other

echo ""
echo "✓ Complete!"
echo ""
echo "Single speaker chunks:"
echo "  Train: data/speaker_${SPEAKER_ID}/train/"
echo "  Test:  data/speaker_${SPEAKER_ID}/test/"
echo ""
echo "Overlap datasets:"
echo "  Train: data/overlap_speaker${SPEAKER_ID}_train/"
echo "  Test:  data/overlap_speaker${SPEAKER_ID}_test/"
