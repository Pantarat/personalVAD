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
OVERLAP_PERCENTAGE=100  # 0-100: percentage of time with overlap
OVERLAP_AMPLITUDE=1.5  # 0-1: amplitude ratio of overlapped speech

# Standard data-prep flags
AUGMENT=false
repo_root=$PWD
KALDI=$repo_root/kaldi
nj_features=4
utt_count=100
kaldi_root=$repo_root/kaldi/egs/pvad
feature_dir_name=overlap_100pct_15apt_100

# LibriSpeech folders
libri_folders=()
libri_folders+=" test-clean"
libri_folders+=" test-other"

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
  echo "${green}Generating OVERLAPPING concatenated utterances...${reset}"
  echo "${green}================================================${reset}"
  echo "${yellow}Settings:${reset}"
  echo "${yellow}  - Overlap percentage: ${OVERLAP_PERCENTAGE}%${reset}"
  echo "${yellow}  - Overlap amplitude: ${OVERLAP_AMPLITUDE}x${reset}"
  echo "${yellow}  - Utterance count: ${utt_count}${reset}"
  echo "${yellow}  - Source datasets: ${libri_folders}${reset}"
  echo ""

  python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir $concat_dir \
    --count $utt_count \
    --overlap_pct $OVERLAP_PERCENTAGE \
    --amplitude_ratio $OVERLAP_AMPLITUDE \
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
    ./reverberate_augment.sh 0
    cd $repo_root
  fi
fi

# Feature extraction
if [ $stage -le 3 ]; then
  if [ "$AUGMENT" = true ]; then
    cp -r data/embeddings $kaldi_root/data/

    cd $kaldi_root
    export PATH="$KALDI/src/featbin:$PATH"
    cd data/augmented
    feature_dir=$kaldi_root/data/$feature_dir_name
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
      --data_root data/augmented \
      --dest_path data/$feature_dir_name \
      --embed_path data/embeddings \
      --use_kaldi \
      --kaldi_root $kaldi_root || { 
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
