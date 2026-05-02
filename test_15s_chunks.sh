#!/bin/bash

# Test script for 15-second chunk generation with standalone non-target speech

# Configuration
SINGLE_SPEAKER_DIR="data/single_speaker_84"
OUTPUT_DIR="data/overlap_test_15s"
LIBRI_ROOT="data/LibriSpeech"
NUM_SAMPLES=10
OVERLAP_PCT=50

# Clean output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Generating overlap dataset with 15-second chunks..."
python src/generate_overlapping_utterances.py \
    --libri-root "$LIBRI_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --n-samples "$NUM_SAMPLES" \
    --overlap-percentage "$OVERLAP_PCT" \
    --single-speaker-dataset "$SINGLE_SPEAKER_DIR" \
    --flac

echo ""
echo "Checking generated files..."
ls -lh "$OUTPUT_DIR/train/"*.flac | head -20

echo ""
echo "Checking metadata..."
cat "$OUTPUT_DIR/train/"*.overlap_meta | head -50

echo ""
echo "Checking text file with labels..."
head -20 "$OUTPUT_DIR/train/text"

echo ""
echo "Testing label extraction..."
python scripts/extract_labels.py \
    --labels-scp "$OUTPUT_DIR/train/labels.scp" \
    --output-dir "test_outputs/labels_15s" \
    --target-utt $(head -1 "$OUTPUT_DIR/train/wav.scp" | cut -d' ' -f1)

echo ""
echo "Done! Check test_outputs/labels_15s for visualization."
