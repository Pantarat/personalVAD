#!/bin/bash
#
# File: export_augmented.sh
# 
# Export augmented audio files from Kaldi wav.scp to actual audio files.
# This script can be run standalone or called from the augmentation pipeline.
#
# Usage:
#   bash export_augmented.sh [format] [output_dir]
#
# Arguments:
#   format      - Output format: 'wav' or 'flac' (default: wav)
#   output_dir  - Output directory (default: data/augmented_audio)
#

# Colors
green=$(tput setaf 2)
yellow=$(tput setaf 3)
reset=$(tput sgr0)

# Default settings
FORMAT=${1:-wav}
OUTPUT_DIR=${2:-data/augmented_audio}
KALDI_DIR="kaldi/egs/pvad"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if export_augmented_audio.py exists
EXPORT_SCRIPT="$SCRIPT_DIR/export_augmented_audio.py"
if [ ! -f "$EXPORT_SCRIPT" ]; then
  echo "❌ Error: export_augmented_audio.py not found at: $EXPORT_SCRIPT"
  exit 1
fi

# Check if Kaldi data directory exists
if [ ! -d "$KALDI_DIR" ]; then
  echo "❌ Error: Kaldi directory not found: $KALDI_DIR"
  echo "   Please run augmentation first: bash prepare_dataset_features.sh 0"
  exit 1
fi

echo "${green}====================================================================${reset}"
echo "${green}  Exporting Augmented Audio Files${reset}"
echo "${green}====================================================================${reset}"
echo ""
echo "  Format:      $FORMAT"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Kaldi dir:   $KALDI_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Track statistics
total_files=0
total_success=0

# Function to export a single augmentation type
export_augmentation_type() {
  local aug_type=$1
  local scp_file="$KALDI_DIR/data/$aug_type/wav.scp"
  local output_subdir="$OUTPUT_DIR/$aug_type"
  
  if [ ! -f "$scp_file" ]; then
    echo "${yellow}⚠ Skipping $aug_type: wav.scp not found${reset}"
    return
  fi
  
  # Count utterances
  local count=$(wc -l < "$scp_file")
  total_files=$((total_files + count))
  
  echo ""
  echo "${green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${reset}"
  echo "${green}📁 Exporting: $aug_type ($count files)${reset}"
  echo "${green}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${reset}"
  
  # Create output subdirectory
  mkdir -p "$output_subdir"
  
  # Run export script
  python3 "$EXPORT_SCRIPT" "$scp_file" "$output_subdir" --format "$FORMAT"
  
  # Check success
  local exported=$(find "$output_subdir" -name "*.$FORMAT" | wc -l)
  total_success=$((total_success + exported))
  
  echo "${green}✓ Exported $exported/$count files to: $output_subdir${reset}"
}

# Export each augmentation type
echo "${green}Starting export...${reset}"

export_augmentation_type "reverb"
export_augmentation_type "noise"
export_augmentation_type "music"
export_augmentation_type "babble"

# Also export clean if available
if [ -f "$KALDI_DIR/data/clean/wav.scp" ]; then
  export_augmentation_type "clean"
fi

# Print summary
echo ""
echo "${green}====================================================================${reset}"
echo "${green}  Export Complete!${reset}"
echo "${green}====================================================================${reset}"
echo ""
echo "  Total files:     $total_files"
echo "  Successfully exported: $total_success"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "${green}You can now play augmented audio files:${reset}"
echo "  find $OUTPUT_DIR -name '*.$FORMAT' | head -5"
echo ""
echo "${green}Or use a media player:${reset}"
echo "  vlc $OUTPUT_DIR/babble/*.${FORMAT}"
echo ""
echo "${green}====================================================================${reset}"
