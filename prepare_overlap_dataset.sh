#!/bin/bash
#
# File: prepare_overlap_dataset.sh
# Author: Enhanced by AI Assistant
# Based on: prepare_dataset_features.sh by Simon Sedlacek
#
# This script handles data preparation with OVERLAPPING utterances.
#
# Usage:
# $ bash prepare_overlap_dataset.sh 0
#
# STAGES:
# 0) Generate overlapping concatenations and ground truth labels
# 1) Fix kaldi-specific files (if AUGMENT==true)
# 2) Run augmentation in kaldi folder (if AUGMENT==true)
# 3) Extract features
#

#================ EDIT HERE =========================

# Overlap-specific settings
USE_OVERLAP=true
OVERLAP_PERCENTAGE=100  # 0-100: percentage of TARGET SPEAKER SPEECH with overlap (excludes silence)
                        # E.g., 50 means half of target speaker's speech has other speakers talking over it
OVERLAP_AMPLITUDE=1  # 0-1: amplitude ratio of overlapped speech relative to main speaker

# Single speaker dataset option (HIGHEST PRIORITY)
# Set to the path(s) of single speaker dataset(s) (from extract_single_speaker.py)
# This will use the pre-generated chunks as the target speaker(s)
# Can specify multiple datasets as an array
# Leave empty array to use BASE_SPEAKER or random selection instead
# Examples:
#   Single: SINGLE_SPEAKER_DATASETS=("data/speaker_84/train")
#   Multiple: SINGLE_SPEAKER_DATASETS=(
#               "data/speaker_84/train1"
#               "data/speaker_84/train2"
#               "data/speaker_174/train"
#             )
# SINGLE_SPEAKER_DATASETS=(
#     "data/speaker_84/train1"
#     "data/speaker_84/train2"
#     "data/speaker_84/train3"
# )
SINGLE_SPEAKER_DATASETS=(
    "data/speaker_5105/test"
)

# Number of utterances to use from EACH single speaker dataset
# Set to 0 or leave empty to use all utterances from each dataset
# If set, will randomly sample this many utterances from EACH dataset
SINGLE_SPEAKER_UTT_COUNT=0  # Example: 100 to use 100 utterances per dataset

# Base speaker option (NEW FEATURE)
# Set to a speaker ID (e.g., "84") to have that speaker as the target in ALL overlaps
# Leave empty or set to "" for random speaker selection (original behavior)
BASE_SPEAKER="5105"  # Example: BASE_SPEAKER="84"

# No target speaker option
# Set to true to generate samples with NO target speakers (all speakers are non-target)
# When enabled, all speech will be labeled as 'N' (non-target), no 'T' labels
NO_TARGET_SPEAKER=false

# Standard data-prep flags
AUGMENT=true
repo_root=$PWD
KALDI=$repo_root/kaldi
nj_features=4
utt_count=500
kaldi_root=$repo_root/kaldi/egs/pvad
feature_dir_name=5105_ov_test_ov100pct_main8463_babble_500_9-4-2026

# LibriSpeech folders (space-separated list)
libri_folders="test-clean test-other"
# libri_folders="dev-clean dev-other train-clean-100"
# Add more as needed:
# libri_folders="dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500"
export NAME=overlap

if [ -z ${NAME+x} ]; then
  export NAME=overlap
fi

#==================================================

# Colors
red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
reset=`tput sgr0`

# Set destination directory
if [ "$AUGMENT" = true ]; then
  mkdir -p $repo_root/kaldi/egs/pvad/data
  concat_dir=$repo_root/kaldi/egs/pvad/data/$NAME
else
  concat_dir=$repo_root/data/$NAME
fi

cd $repo_root

if [ -z "$1" ]; then
  echo "Please specify the data preparation stage."
  exit 0
else
  stage=$1
fi

# Generate overlapping concatenations
if [ $stage -le 0 ]; then

  # Check if LibriSpeech directory exists
  if [[ ! -d "data/LibriSpeech" ]]; then
    cd data
    for subset in $libri_folders
    do
      tar -xf $subset.tar.gz || { 
        echo "${red}LibriSpeech subset not available. Download from https://www.openslr.org/12${reset}"
        exit 1
      }
    done

    unzip LibriSpeech-Alignments.zip || { 
      echo "${red}LibriSpeech alignments not available. Download from https://zenodo.org/record/2619474${reset}"
      exit 1
    }
    cd ../
  fi

  echo "${green}================================================${reset}"
  echo "${green}Generating MIXED multi-speaker utterances...${reset}"
  echo "${green}================================================${reset}"
  echo "${yellow}Settings:${reset}"
  if [ "$NO_TARGET_SPEAKER" = true ]; then
    if [ -n "$SINGLE_SPEAKER_DATASET" ]; then
      echo "${yellow}  - Mode: NO TARGET SPEAKERS (all speech is non-target)${reset}"
      echo "${yellow}  - Single speaker dataset ${SINGLE_SPEAKER_DATASET} in metadata ONLY${reset}"
    elif [ -n "$BASE_SPEAKER" ]; then
      echo "${yellow}  - Mode: NO TARGET SPEAKERS (all speech is non-target)${reset}"
      echo "${yellow}  - Base speaker ${BASE_SPEAKER} in metadata ONLY (not in audio)${reset}"
    else:
      echo "${yellow}  - Mode: NO TARGET SPEAKERS (all speech is non-target)${reset}"
    fi
  elif [ ${#SINGLE_SPEAKER_DATASETS[@]} -gt 0 ]; then
    echo "${yellow}  - Using ${#SINGLE_SPEAKER_DATASETS[@]} single speaker dataset(s):${reset}"
    for dataset in "${SINGLE_SPEAKER_DATASETS[@]}"; do
      echo "${yellow}      - ${dataset}${reset}"
    done
    echo "${yellow}  - These datasets' chunks will be the target speaker(s)${reset}"
    if [ "$SINGLE_SPEAKER_UTT_COUNT" -gt 0 ]; then
      echo "${yellow}  - Using ${SINGLE_SPEAKER_UTT_COUNT} utterances per dataset${reset}"
    fi
  elif [ -n "$BASE_SPEAKER" ]; then
    echo "${yellow}  - Base speaker (target in all samples): ${BASE_SPEAKER}${reset}"
  else
    echo "${yellow}  - Target speaker: Random selection${reset}"
  fi
  echo "${yellow}  - Overlap percentage: ${OVERLAP_PERCENTAGE}% (of target speech with interference)${reset}"
  echo "${yellow}  - Overlap amplitude: ${OVERLAP_AMPLITUDE}x${reset}"
  echo "${yellow}  - Utterance count: ${utt_count}${reset}"
  echo "${yellow}  - Source datasets: ${libri_folders}${reset}"
  echo ""
  echo "${yellow}Generated audio will contain:${reset}"
  echo "${yellow}  1. Target speaker speech (TSS)${reset}"
  echo "${yellow}  2. Overlapping speech (other speakers talking over target)${reset}"
  echo "${yellow}  3. Standalone non-target speech (other speakers in gaps/silence)${reset}"
  echo ""

  # Build command with optional parameters
  base_speaker_arg=""
  if [ -n "$BASE_SPEAKER" ]; then
    base_speaker_arg="--base-speaker $BASE_SPEAKER"
  fi

  single_speaker_arg=""
  if [ ${#SINGLE_SPEAKER_DATASETS[@]} -gt 0 ]; then
    for dataset in "${SINGLE_SPEAKER_DATASETS[@]}"; do
      single_speaker_arg="$single_speaker_arg --single-speaker-dataset \"$dataset\""
    done
    if [ "$SINGLE_SPEAKER_UTT_COUNT" -gt 0 ]; then
      single_speaker_arg="$single_speaker_arg --single-speaker-count $SINGLE_SPEAKER_UTT_COUNT"
    fi
  fi

  no_target_arg=""
  if [ "$NO_TARGET_SPEAKER" = true ]; then
    no_target_arg="--no-target-speaker"
  fi

  eval python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir $concat_dir \
    --count $utt_count \
    --overlap_pct $OVERLAP_PERCENTAGE \
    --amplitude_ratio $OVERLAP_AMPLITUDE \
    --scp_prefix $concat_dir/ \
    $base_speaker_arg \
    $single_speaker_arg \
    $no_target_arg \
    $libri_folders || { 
      echo "${red}Overlap utterance generation failed. Exiting...${reset}"
      exit 1
    }

  echo "${green}✓ Overlapping utterances saved to $concat_dir${reset}"
  echo ""

fi

# Augmentation (if specified)
if [ "$AUGMENT" = true ]; then
  if [ $stage -le 1 ]; then
    echo "${green}Moving to kaldi directory...${reset}"
    cd $kaldi_root

    utils/fix_data_dir.sh data/$NAME
    utils/utt2spk_to_spk2utt.pl data/$NAME/utt2spk > data/$NAME/spk2utt

  fi

  if [ $stage -le 2 ]; then
    cd $kaldi_root

    echo "${green}Running reverberation and augmentation...${reset}"
    bash ./reverberate_augment.sh 0
    cd $repo_root
  fi
fi

# Feature extraction
if [ $stage -le 3 ]; then
  if [ "$AUGMENT" = true ]; then
    # Copy embeddings if they exist
    if [ -d "$repo_root/data/embeddings" ]; then
      cp -r $repo_root/data/embeddings $kaldi_root/data/
    fi

    cd $kaldi_root
    export PATH="$KALDI/src/featbin:$PATH"
    cd data/augmented
    feature_dir=$repo_root/data/$feature_dir_name
  else
    cd $concat_dir
    feature_dir=$concat_dir/../$feature_dir_name
  fi

  mkdir -p $feature_dir
  echo "${green}Splitting wav.scp into $nj_features parts...${reset}"

  split -n l/$nj_features --additional-suffix .scp -d wav.scp split_
  
  cd $repo_root
  echo "${green}Running feature extraction...${reset}"

  if [ "$AUGMENT" = true ]; then
    python3 src/extract_features.py \
      --data_root $kaldi_root/data/augmented \
      --dest_path $repo_root/data/$feature_dir_name \
      --embed_path $kaldi_root/data/embeddings \
      --use_kaldi \
      --kaldi_root "$kaldi_root" || { 
        echo "${red}Feature extraction failed. Exiting...${reset}"
        exit 1
      }
  else
    python3 src/extract_features.py \
      --data_root data/$NAME \
      --dest_path data/$feature_dir_name \
      --embed_path data/embeddings || { 
        echo "${red}Feature extraction failed. Exiting...${reset}"
        exit 1
      }
  fi

  # Combine feature scps
  cd $feature_dir
  cat fbanks_*.scp > fbanks.scp
  cat scores_*.scp > scores.scp
  cat labels_*.scp > labels.scp
  cat targets_*.scp > targets.scp

  # Copy wav files and wav.scp to feature directory for score recomputation
  echo "${green}Copying audio files to feature directory...${reset}"
  mkdir -p $feature_dir/wav
  
  # Initialize counters
  success_count=0
  fail_count=0
  
  if [ "$AUGMENT" = true ]; then
    # Copy wav.scp
    cp $kaldi_root/data/augmented/wav.scp $feature_dir/wav.scp.orig
    
    # Initialize temp file
    > $feature_dir/wav.scp.tmp
    > $feature_dir/failed_wav_copy.log
    
    # Copy actual wav files and create new wav.scp with relative paths
    while IFS=' ' read -r utt_id wav_path; do
      echo "${yellow}Processing: $utt_id${reset}"
      
      # Handle pipe commands in wav_path (e.g., "flac -d ... | wav-reverberate ... |")
      if [[ "$wav_path" == *"|" ]]; then
        # Remove trailing |
        wav_cmd="${wav_path%|}"
        
        echo "  Executing pipe command..."
        echo "  Command: $wav_cmd"
        
        # Execute in Kaldi environment with proper PATH and convert to WAV
        # Show all errors and stop on failure
        if ! (cd "$kaldi_root" && source path.sh && eval "$wav_cmd" | sox -t wav - -t wav "$feature_dir/wav/${utt_id}.wav"); then
          echo "${red}ERROR: Failed to execute pipe command${reset}"
          echo "${red}Utterance: $utt_id${reset}"
          echo "${red}Command: $wav_cmd${reset}"
          echo "${red}Working directory: $kaldi_root${reset}"
          exit 1
        fi
        
        # Verify the output file is a valid WAV
        if [ ! -f "$feature_dir/wav/${utt_id}.wav" ]; then
          echo "${red}ERROR: Output file not created${reset}"
          echo "${red}Expected: $feature_dir/wav/${utt_id}.wav${reset}"
          exit 1
        fi
        
        if [ ! -s "$feature_dir/wav/${utt_id}.wav" ]; then
          echo "${red}ERROR: Output file is empty${reset}"
          echo "${red}File: $feature_dir/wav/${utt_id}.wav${reset}"
          exit 1
        fi
        
        # Check if it's a valid audio file using sox
        if ! sox --i "$feature_dir/wav/${utt_id}.wav" >/dev/null 2>&1; then
          echo "${red}ERROR: Invalid WAV file (corrupted audio)${reset}"
          echo "${red}File: $feature_dir/wav/${utt_id}.wav${reset}"
          echo "${yellow}Running sox info for details:${reset}"
          sox --i "$feature_dir/wav/${utt_id}.wav"
          exit 1
        fi
        
        echo "$utt_id $feature_dir/wav/${utt_id}.wav" >> $feature_dir/wav.scp.tmp
        ((success_count++))
        echo "${green}  ✓ Success${reset}"
        
      else
        # Direct file copy
        echo "  Copying file: $wav_path"
        if ! cp "$wav_path" "$feature_dir/wav/${utt_id}.wav"; then
          echo "${red}ERROR: Failed to copy file${reset}"
          echo "${red}Source: $wav_path${reset}"
          echo "${red}Destination: $feature_dir/wav/${utt_id}.wav${reset}"
          exit 1
        fi
        echo "$utt_id $feature_dir/wav/${utt_id}.wav" >> $feature_dir/wav.scp.tmp
        ((success_count++))
        echo "${green}  ✓ Success${reset}"
      fi
    done < $kaldi_root/data/augmented/wav.scp
    
    mv $feature_dir/wav.scp.tmp $feature_dir/wav.scp
    echo "${green}✓ Copied $success_count audio files from augmented data${reset}"
    if [ $fail_count -gt 0 ]; then
      echo "${yellow}⚠ Failed to copy $fail_count files (see $feature_dir/failed_wav_copy.log)${reset}"
      echo "${yellow}  This is normal if some MUSAN files are corrupted - they will be skipped${reset}"
    fi
  else
    # Copy wav.scp
    cp $concat_dir/wav.scp $feature_dir/wav.scp.orig
    
    # Initialize temp file
    > $feature_dir/wav.scp.tmp
    > $feature_dir/failed_wav_copy.log
    
    # Copy actual wav files and create new wav.scp with relative paths
    while IFS=' ' read -r utt_id wav_path; do
      echo "${yellow}Processing: $utt_id${reset}"
      
      # Handle pipe commands in wav_path (e.g., "flac -d -c -s file.flac |")
      if [[ "$wav_path" == *"|" ]]; then
        # Remove trailing |
        wav_cmd="${wav_path%|}"
        
        echo "  Executing pipe command..."
        
        # Execute pipe command and convert to WAV
        if eval "$wav_cmd" | sox -t wav - -t wav "$feature_dir/wav/${utt_id}.wav" 2>/dev/null; then
          # Verify the output file
          if [ -f "$feature_dir/wav/${utt_id}.wav" ] && [ -s "$feature_dir/wav/${utt_id}.wav" ]; then
            echo "$utt_id $feature_dir/wav/${utt_id}.wav" >> $feature_dir/wav.scp.tmp
            ((success_count++))
            echo "${green}  ✓ Success${reset}"
          else
            echo "$utt_id: Empty or missing output file" >> $feature_dir/failed_wav_copy.log
            ((fail_count++))
            echo "${red}  ✗ Failed (empty output)${reset}"
          fi
        else
          echo "$utt_id: Could not execute $wav_cmd" >> $feature_dir/failed_wav_copy.log
          ((fail_count++))
          echo "${red}  ✗ Failed (command error)${reset}"
        fi
        
      else
        # Direct file copy
        echo "  Copying file: $wav_path"
        if cp "$wav_path" "$feature_dir/wav/${utt_id}.wav" 2>/dev/null; then
          echo "$utt_id $feature_dir/wav/${utt_id}.wav" >> $feature_dir/wav.scp.tmp
          ((success_count++))
          echo "${green}  ✓ Success${reset}"
        else
          echo "$utt_id: Could not copy $wav_path" >> $feature_dir/failed_wav_copy.log
          ((fail_count++))
          echo "${red}  ✗ Failed (copy error)${reset}"
        fi
      fi
    done < $concat_dir/wav.scp
    
    mv $feature_dir/wav.scp.tmp $feature_dir/wav.scp
    echo "${green}✓ Copied $success_count audio files from concatenated data${reset}"
    if [ $fail_count -gt 0 ]; then
      echo "${yellow}⚠ Failed to copy $fail_count files (see $feature_dir/failed_wav_copy.log)${reset}"
      echo "${yellow}  This is normal if some files are corrupted - they will be skipped${reset}"
    fi
  fi

  # Clean up
  for name in fbanks_ targets_ scores_ labels_
  do
    rm $name*.scp
  done

  # Move features if using Kaldi
  if [ "$AUGMENT" = true ]; then
    echo "${green}Moving extracted features to repo root...${reset}"
    if [[ -d $repo_root/data/$feature_dir_name ]]; then
      echo "${yellow}Feature directory already exists. Not moving.${reset}"
      echo "${yellow}Features remain in $kaldi_root/data/$feature_dir_name${reset}"
    else
      cd ..
      mv $feature_dir_name $repo_root/data/
      echo "${green}Features saved to $repo_root/data/$feature_dir_name${reset}"
    fi
  else
    echo "${green}Features saved to $repo_root/data/$feature_dir_name${reset}"
  fi

  cd $repo_root
  echo "${green}✓ Feature extraction done!${reset}"
fi

echo ""
echo "${green}================================================${reset}"
echo "${green}✓ Overlap dataset preparation complete!${reset}"
echo "${green}================================================${reset}"
echo ""
echo "${yellow}Next steps:${reset}"
echo "  1. Listen to samples: python src/play_overlap_sample.py --overlap_dir $concat_dir"
echo "  2. Train model with: data/$feature_dir_name"
echo ""
