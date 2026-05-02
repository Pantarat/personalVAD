#!/bin/bash
#
# File: reverberate_augment.sh
# Author: Simon Sedlacek
# Email: xsedla1h@stud.fit.vutbr.cz
#
# This script handles Kaldi data reverberation and agumentation. It is supposed to be
# invoked by the prepare_dataset_features.sh script from the root directory of the
# project.
#
# The reverberation and MUSAN augmentation in this script was based on the reverberation
# and augmentation approaches from the run.sh script from the Kaldi SITW v2 recipe, which
# is available here: https://github.com/kaldi-asr/kaldi/blob/master/egs/sitw/v2/run.sh
# 

# Source path.sh to set up Kaldi paths
. ./path.sh || exit 1

#================ EDIT HERE =========================

use_noise=false
use_music=false
use_babble=true
use_reverb=false

# Include original non-augmented data in output
# Set to false to have ONLY augmented data (no original utterances)
INCLUDE_ORIGINAL=false

# MUSAN augmentation mode: 'train' or 'test'
# This determines which subset of MUSAN data to use (80/20 split)
AUGMENT_MODE="test"  # Options: 'train' or 'test'

# Set to true to export augmented audio files (WAV format)
# This will save actual audio files instead of just generating them on-the-fly
export_augmented_audio=false
export_audio_dir="data/augmented_audio"  # Directory to save exported audio
export_format="flac"  # Output format: 'wav' or 'flac'

export KALDI_HOME=/mnt/c/Work/Coding/Diarization/kaldi

#====================================================

# first, setup the NAME variable
if [ -z ${NAME+x} ]; then
  export NAME=clean # name is empty, default to 'clean'
fi

red=`tput setaf 1`
green=`tput setaf 2`
magenta=`tput setaf 5`
reset=`tput sgr0`

train_cmd="run.pl"
decode_cmd="run.pl"
musan_root=musan

if [ -e $1 ]; then
  echo "Please specifiy the data preparation stage."
  exit 0
else
  stage=$1
fi

# Download and unzip the rirs_noises corpus if missing (do this before stage check)
if [ ! -d "RIRS_NOISES" ]; then
  if [ ! -f "rirs_noises.zip" ]; then
    echo "${magenta}Downloading and unzipping rirs_noises${reset}"
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
  fi
  unzip rirs_noises.zip
fi

# Split RIRS_NOISES into train (80%) and test (20%) subsets (do this before stage check)
echo "${green}Checking RIRS_NOISES train/test splits...${reset}"
for room_type in smallroom mediumroom; do
  rir_list_orig="RIRS_NOISES/simulated_rirs/${room_type}/rir_list"
  rir_list_train="RIRS_NOISES/simulated_rirs/${room_type}/rir_list_train"
  rir_list_test="RIRS_NOISES/simulated_rirs/${room_type}/rir_list_test"
  
  if [ ! -f "$rir_list_train" ] || [ ! -f "$rir_list_test" ]; then
    echo "  Splitting ${room_type} RIRs (80/20)..."
    
    # Get total number of RIRs
    total_rirs=$(wc -l < "$rir_list_orig")
    train_rirs=$(echo "$total_rirs * 0.8" | bc | cut -d'.' -f1)
    
    # Shuffle and split rir_list
    shuf "$rir_list_orig" > "${rir_list_orig}.shuffled"
    head -n $train_rirs "${rir_list_orig}.shuffled" > "$rir_list_train"
    tail -n +$(($train_rirs + 1)) "${rir_list_orig}.shuffled" > "$rir_list_test"
    rm "${rir_list_orig}.shuffled"
    
    echo "    Train: $(wc -l < "$rir_list_train") RIRs"
    echo "    Test: $(wc -l < "$rir_list_test") RIRs"
  else
    echo "  ${room_type} RIRs already split"
  fi
done

echo "${green}✓ RIRS_NOISES splits ready${reset}"
echo "${yellow}Splits stored in RIRS_NOISES/simulated_rirs/ (persistent)${reset}"
echo ""

if [ $stage -le 0 ]; then

  echo "${green}Reverberating the dataset using ${AUGMENT_MODE} RIRs...${reset}"
  # Make a version with reverberated speech using train or test RIRs
  rirs_suffix="_${AUGMENT_MODE}"
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list${rirs_suffix}")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list${rirs_suffix}")

  # Reverberate our data folder
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/$NAME data/reverb

  # Add a suffix to the reverberated data..
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/reverb data/reverb.new
  rm -rf data/reverb
  mv data/reverb.new data/reverb
fi

if [ $stage -le 1 ]; then

  if [ ! -d "musan" ]; then
    # Download and unzip musan, if missing
    if [ ! -f "musan.tar.gz" ]; then
      echo "${magenta}Downloading and unzipping musan${magenta}"
      wget --no-check-certificate https://www.openslr.org/resources/17/musan.tar.gz
    fi
    tar -xf musan.tar.gz
    rm musan.tar.gz
  fi

  echo "${green}Augmenting the dataset using ${AUGMENT_MODE} MUSAN data...${reset}"

  # prepare musan
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done
  
  # Split MUSAN data into train (80%) and test (20%) subsets
  # Store splits in musan root directory to persist across different dataset generations
  echo "${green}Splitting MUSAN data into train/test (80/20)...${reset}"
  for name in speech noise music; do
    if [ ! -f "${musan_root}/musan_${name}_train/wav.scp" ] || [ ! -f "${musan_root}/musan_${name}_test/wav.scp" ]; then
      echo "  Splitting musan_${name}..."
      
      # Create train/test directories in musan root
      mkdir -p ${musan_root}/musan_${name}_train ${musan_root}/musan_${name}_test
      
      # Get total number of utterances
      total_utts=$(wc -l < data/musan_${name}/wav.scp)
      train_utts=$(echo "$total_utts * 0.8" | bc | cut -d'.' -f1)
      
      # Shuffle and split wav.scp
      shuf data/musan_${name}/wav.scp > data/musan_${name}/wav.scp.shuffled
      head -n $train_utts data/musan_${name}/wav.scp.shuffled > ${musan_root}/musan_${name}_train/wav.scp
      tail -n +$(($train_utts + 1)) data/musan_${name}/wav.scp.shuffled > ${musan_root}/musan_${name}_test/wav.scp
      rm data/musan_${name}/wav.scp.shuffled
      
      # Copy reco2dur for both splits
      if [ -f "data/musan_${name}/reco2dur" ]; then
        # Extract only the utterances in each split
        cut -d' ' -f1 ${musan_root}/musan_${name}_train/wav.scp | \
          grep -F -f - data/musan_${name}/reco2dur > ${musan_root}/musan_${name}_train/reco2dur
        cut -d' ' -f1 ${musan_root}/musan_${name}_test/wav.scp | \
          grep -F -f - data/musan_${name}/reco2dur > ${musan_root}/musan_${name}_test/reco2dur
      fi
      
      # Copy utt2spk and spk2utt if they exist
      if [ -f "data/musan_${name}/utt2spk" ]; then
        cut -d' ' -f1 ${musan_root}/musan_${name}_train/wav.scp | \
          grep -F -f - data/musan_${name}/utt2spk > ${musan_root}/musan_${name}_train/utt2spk
        cut -d' ' -f1 ${musan_root}/musan_${name}_test/wav.scp | \
          grep -F -f - data/musan_${name}/utt2spk > ${musan_root}/musan_${name}_test/utt2spk
        
        utils/utt2spk_to_spk2utt.pl ${musan_root}/musan_${name}_train/utt2spk > ${musan_root}/musan_${name}_train/spk2utt
        utils/utt2spk_to_spk2utt.pl ${musan_root}/musan_${name}_test/utt2spk > ${musan_root}/musan_${name}_test/spk2utt
      fi
      
      echo "    Train: $(wc -l < ${musan_root}/musan_${name}_train/wav.scp) utterances"
      echo "    Test: $(wc -l < ${musan_root}/musan_${name}_test/wav.scp) utterances"
    else
      echo "  musan_${name} already split (found in ${musan_root}/)"
    fi
  done
  
  echo "${green}✓ MUSAN data split complete${reset}"
  echo "${yellow}Train/test splits stored in ${musan_root}/ (persistent across dataset generations)${reset}"
  
  # Clean up temporary full MUSAN data directories to avoid duplication
  echo "${green}Cleaning up temporary MUSAN data directories...${reset}"
  for name in speech noise music; do
    if [ -d "data/musan_${name}" ]; then
      rm -rf "data/musan_${name}"
      echo "  Removed data/musan_${name}"
    fi
  done
  echo "${green}✓ Cleanup complete${reset}"

  # Use train or test MUSAN data based on AUGMENT_MODE
  musan_suffix="_${AUGMENT_MODE}"
  echo "${yellow}Using MUSAN ${AUGMENT_MODE} data for augmentation${reset}"
  
  # Augment original data only - reverb is treated as a separate augmentation
  # This creates: data/noise, data/music, data/babble (all from original)
  # Reference MUSAN splits directly from musan root directory
  
  # noise
  if $use_noise; then
    echo "${green}Creating noise augmentation...${reset}"
    steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "${musan_root}/musan_noise${musan_suffix}" data/$NAME data/noise
    if [ $? -ne 0 ]; then
      echo "${red}Error: Noise augmentation failed${reset}"
      exit 1
    fi
    echo "${green}✓ Noise augmentation complete: $(wc -l < data/noise/wav.scp) utterances${reset}"
  fi
  # music
  if $use_music; then
    echo "${green}Creating music augmentation...${reset}"
    steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "${musan_root}/musan_music${musan_suffix}" data/$NAME data/music
    if [ $? -ne 0 ]; then
      echo "${red}Error: Music augmentation failed${reset}"
      exit 1
    fi
    echo "${green}✓ Music augmentation complete: $(wc -l < data/music/wav.scp) utterances${reset}"
  fi
  # speech
  if $use_babble; then
    echo "${green}Creating babble augmentation...${reset}"
    steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "${musan_root}/musan_speech${musan_suffix}" data/$NAME data/babble
    if [ $? -ne 0 ]; then
      echo "${red}Error: Babble augmentation failed${reset}"
      exit 1
    fi
    echo "${green}✓ Babble augmentation complete: $(wc -l < data/babble/wav.scp) utterances${reset}"
  fi
fi

# combine the resulted augmented and reverberated scps
echo "${green}Combining augmented data...${reset}"
combine="data/augmented"
if $INCLUDE_ORIGINAL; then 
  combine+=" data/$NAME"
  echo "  ✓ Including original non-augmented data"
else
  echo "  ℹ Excluding original data (augmented only)"
fi
if $use_reverb; then combine+=" data/reverb"; fi
if $use_noise; then combine+=" data/noise"; fi
if $use_music; then combine+=" data/music"; fi
if $use_babble; then combine+=" data/babble"; fi

echo "${yellow}Combining: ${combine}${reset}"
utils/combine_data.sh ${combine}

if [ $? -ne 0 ]; then
  echo "${red}Error: Failed to combine augmented data${reset}"
  exit 1
fi

# Copy reco2dur from original data
if [ -f "data/$NAME/reco2dur" ]; then
  cp data/$NAME/reco2dur data/augmented
else
  echo "${yellow}Warning: data/$NAME/reco2dur not found${reset}"
fi

echo "${green}✓ Data combination complete${reset}"

# Fix labels for augmented utterances
echo "${green}Fixing labels for augmented utterances...${reset}"
echo "${yellow}This ensures text file entries match augmented utterance IDs${reset}"

# The combine_data.sh merges wav.scp, utt2spk, etc but the text file (labels)
# needs special handling because augmented IDs have suffixes (-reverb, -noise, etc)
python3 ../../../src/fix_augmented_labels.py \
  --data_dir data/augmented \
  --original_text data/$NAME/text

if [ $? -eq 0 ]; then
  echo "${green}✓ Labels fixed successfully${reset}"
else
  echo "${red}❌ Error fixing labels. Please check manually.${reset}"
  exit 1
fi

# Export augmented audio files if requested
if [ "$export_augmented_audio" = true ]; then
  echo "${green}Exporting augmented audio files using Python...${reset}"
  
  # Create export directory
  mkdir -p $export_audio_dir
  if [ $? -ne 0 ]; then
    echo "${red}Error: Failed to create export directory: $export_audio_dir${reset}"
    exit 1
  fi
  
  # Use Python script to export audio (handles complex Kaldi pipelines properly)
  python_script="../../../src/export_augmented_audio_pure_python.py"
  
  if [ ! -f "$python_script" ]; then
    echo "${red}Error: Python export script not found: $python_script${reset}"
    exit 1
  fi
  
  # Export each augmentation type using Python (limit to 10 files for testing)
  if $use_reverb && [ -f "data/reverb/wav.scp" ]; then
    echo "  Exporting reverb audio..."
    python3 "$python_script" data/reverb/wav.scp "$export_audio_dir/reverb" --format "$export_format" --max-files 10
  fi
  
  if $use_noise && [ -f "data/noise/wav.scp" ]; then
    echo "  Exporting noise audio..."
    python3 "$python_script" data/noise/wav.scp "$export_audio_dir/noise" --format "$export_format" --max-files 10
  fi
  
  if $use_music && [ -f "data/music/wav.scp" ]; then
    echo "  Exporting music audio..."
    python3 "$python_script" data/music/wav.scp "$export_audio_dir/music" --format "$export_format" --max-files 10
  fi
  
  if $use_babble && [ -f "data/babble/wav.scp" ]; then
    echo "  Exporting babble audio..."
    python3 "$python_script" data/babble/wav.scp "$export_audio_dir/babble" --format "$export_format" --max-files 10
  fi
  
  echo "${green}✓ All augmented audio files exported to: $export_audio_dir${reset}"
fi
