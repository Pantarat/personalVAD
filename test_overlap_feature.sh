#!/bin/bash
#
# File: test_overlap_feature.sh
# Quick test script for overlap generation feature
#
# This script generates a small test dataset to verify the overlap feature works correctly.
#

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║  PersonalVAD - Overlap Feature Test                            ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
green=`tput setaf 2`
yellow=`tput setaf 3`
red=`tput setaf 1`
reset=`tput sgr0`

# Configuration
TEST_DIR="data/overlap_test"
TEST_COUNT=10

echo "${yellow}Test Configuration:${reset}"
echo "  Output: $TEST_DIR"
echo "  Count: $TEST_COUNT utterances"
echo "  Overlap: 30%"
echo "  Amplitude: 0.7x"
echo ""

# Check dependencies
echo "${green}Checking dependencies...${reset}"

if ! python -c "import numpy" 2>/dev/null; then
    echo "${red}✗ numpy not found. Install: pip install numpy${reset}"
    exit 1
fi

if ! python -c "import librosa" 2>/dev/null; then
    echo "${red}✗ librosa not found. Install: pip install librosa${reset}"
    exit 1
fi

if ! python -c "import soundfile" 2>/dev/null; then
    echo "${red}✗ soundfile not found. Install: pip install soundfile${reset}"
    exit 1
fi

echo "${green}✓ All dependencies found${reset}"
echo ""

# Check LibriSpeech data
echo "${green}Checking LibriSpeech data...${reset}"

if [ ! -d "data/LibriSpeech/test-clean" ]; then
    echo "${red}✗ LibriSpeech test-clean not found${reset}"
    echo "${yellow}Download from: https://www.openslr.org/12${reset}"
    exit 1
fi

if [ ! -d "data/LibriSpeech/test-other" ]; then
    echo "${red}✗ LibriSpeech test-other not found${reset}"
    echo "${yellow}Download from: https://www.openslr.org/12${reset}"
    exit 1
fi

echo "${green}✓ LibriSpeech data found${reset}"
echo ""

# Generate overlapping utterances
echo "${green}Generating $TEST_COUNT overlapping utterances...${reset}"
echo ""

python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir $TEST_DIR \
    --count $TEST_COUNT \
    --overlap_pct 30 \
    --amplitude_ratio 0.7 \
    test-clean test-other

if [ $? -ne 0 ]; then
    echo "${red}✗ Generation failed${reset}"
    exit 1
fi

echo ""
echo "${green}✓ Generation complete!${reset}"
echo ""

# Check output
echo "${green}Checking output files...${reset}"

if [ ! -f "$TEST_DIR/wav.scp" ]; then
    echo "${red}✗ wav.scp not found${reset}"
    exit 1
fi

if [ ! -f "$TEST_DIR/text" ]; then
    echo "${red}✗ text not found${reset}"
    exit 1
fi

if [ ! -f "$TEST_DIR/utt2spk" ]; then
    echo "${red}✗ utt2spk not found${reset}"
    exit 1
fi

# Count files
wav_count=$(wc -l < "$TEST_DIR/wav.scp")
meta_count=$(find "$TEST_DIR" -name "*.overlap_meta" | wc -l)

echo "${green}✓ Output files verified${reset}"
echo "  wav.scp entries: $wav_count"
echo "  metadata files: $meta_count"
echo ""

# Show sample metadata
echo "${green}Sample metadata:${reset}"
meta_file=$(find "$TEST_DIR" -name "*.overlap_meta" | head -1)
if [ -f "$meta_file" ]; then
    echo "─────────────────────────────────────────"
    cat "$meta_file"
    echo "─────────────────────────────────────────"
else
    echo "${yellow}No metadata files found${reset}"
fi
echo ""

# Visualize (if matplotlib available)
echo "${green}Attempting to visualize sample...${reset}"
if python -c "import matplotlib" 2>/dev/null; then
    python src/play_overlap_sample.py --overlap_dir $TEST_DIR --no_play
    if [ $? -eq 0 ]; then
        echo "${green}✓ Visualization successful${reset}"
    else
        echo "${yellow}⚠ Visualization failed (non-critical)${reset}"
    fi
else
    echo "${yellow}⚠ matplotlib not found, skipping visualization${reset}"
    echo "${yellow}  Install: pip install matplotlib${reset}"
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║  ${green}✓ Test Completed Successfully!${reset}                               ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "${yellow}Next steps:${reset}"
echo ""
echo "1. Listen to a sample:"
echo "   ${green}python src/play_overlap_sample.py --overlap_dir $TEST_DIR${reset}"
echo ""
echo "2. Generate full dataset:"
echo "   ${green}bash prepare_overlap_dataset.sh 0${reset}"
echo ""
echo "3. Read documentation:"
echo "   ${green}cat doc/OVERLAP_GENERATION.md${reset}"
echo ""
echo "4. See implementation summary:"
echo "   ${green}cat OVERLAP_IMPLEMENTATION_SUMMARY.md${reset}"
echo ""
