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

# use_noise=true
# use_music=true
# use_babble=true
# use_reverb=true

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

if [ $stage -le 0 ]; then

  # Download and unzip the rirs_noises corpus if missing
  if [ ! -d "RIRS_NOISES" ]; then
    if [ ! -f "rirs_noises.zip" ]; then
      echo "${magenta}Downloading and unzipping rirs_noises${magenta}"
      wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    fi
    unzip rirs_noises.zip
  fi

  echo "${green}Reverberating the dataset...${reset}"
   # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

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

  echo "${green}Augmenting the dataset...${reset}"

  # prepare musan
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # noise
  if $use_noise; then
    steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/$NAME data/noise
  fi
  # music
  if $use_music; then
    steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/$NAME data/music
  fi
  # speech
  if $use_babble; then
    steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/$NAME data/babble
  fi
fi

# combine the resulted augmented and reverberated scps
combine="data/augmented data/$NAME data/reverb"
if $use_noise; then combine+=" data/noise"; fi
if $use_music; then combine+=" data/music"; fi
if $use_babble; then combine+=" data/babble"; fi
utils/combine_data.sh ${combine}
cp data/$NAME/reco2dur data/augmented

# Export augmented audio files if requested
if [ "$export_augmented_audio" = true ]; then
  echo "${green}Exporting augmented audio files...${reset}"
  
  EXPORT_SCRIPT="../../../src/export_augmented_audio_pure_python.py"
  
  # Check if export script exists
  if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "${red}Error: export_augmented_audio.py not found at: $EXPORT_SCRIPT${reset}"
    echo "${red}export_augmented_audio is enabled but export script is missing!${reset}"
    exit 1
  fi
  
  # Check if KALDI_HOME is set
  if [ -z "$KALDI_HOME" ]; then
    echo "${red}Error: KALDI_HOME environment variable is not set!${reset}"
    echo "${red}Please set KALDI_HOME before running with export_augmented_audio=true${reset}"
    echo "${yellow}Example: export KALDI_HOME=/mnt/c/Work/Coding/Diarization/kaldi${reset}"
    exit 1
  fi
  
  # Add Kaldi binaries to PATH (needed for wav-reverberate, flac, sox)
  export PATH="$KALDI_HOME/src/featbin:$KALDI_HOME/tools/openfst/bin:$PATH"
  
  # Check if wav-reverberate is available
  if ! command -v wav-reverberate &> /dev/null; then
    echo "${red}Error: wav-reverberate command not found!${reset}"
    echo "${red}Make sure Kaldi is compiled: cd $KALDI_HOME/src && make${reset}"
    exit 1
  fi
  
  # Save current directory
  CURRENT_DIR=$(pwd)
  
  # Create export directory
  mkdir -p $export_audio_dir
  if [ $? -ne 0 ]; then
    echo "${red}Error: Failed to create export directory: $export_audio_dir${reset}"
    cd "$CURRENT_DIR"
    exit 1
  fi
  
  # Export each augmentation type
  if $use_reverb && [ -f "data/reverb/wav.scp" ]; then
    echo "  Exporting reverb audio..."
    mkdir -p $export_audio_dir/reverb
    python3 "$EXPORT_SCRIPT" data/reverb/wav.scp $export_audio_dir/reverb --format $export_format
    if [ $? -ne 0 ]; then
      echo "${red}Error: Failed to export reverb audio${reset}"
      cd "$CURRENT_DIR"
      exit 1
    fi
  fi
  
  if $use_noise && [ -f "data/noise/wav.scp" ]; then
    echo "  Exporting noise audio..."
    mkdir -p $export_audio_dir/noise
    python3 "$EXPORT_SCRIPT" data/noise/wav.scp $export_audio_dir/noise --format $export_format
    if [ $? -ne 0 ]; then
      echo "${red}Error: Failed to export noise audio${reset}"
      cd "$CURRENT_DIR"
      exit 1
    fi
  fi
  
  if $use_music && [ -f "data/music/wav.scp" ]; then
    echo "  Exporting music audio..."
    mkdir -p $export_audio_dir/music
    python3 "$EXPORT_SCRIPT" data/music/wav.scp $export_audio_dir/music --format $export_format
    if [ $? -ne 0 ]; then
      echo "${red}Error: Failed to export music audio${reset}"
      cd "$CURRENT_DIR"
      exit 1
    fi
  fi
  
  if $use_babble && [ -f "data/babble/wav.scp" ]; then
    echo "  Exporting babble audio..."
    mkdir -p $export_audio_dir/babble
    python3 "$EXPORT_SCRIPT" data/babble/wav.scp $export_audio_dir/babble --format $export_format
    if [ $? -ne 0 ]; then
      echo "${red}Error: Failed to export babble audio${reset}"
      cd "$CURRENT_DIR"
      exit 1
    fi
  fi
  
  # Return to original directory
  cd "$CURRENT_DIR"
  
  echo "${green}✓ Augmented audio files exported to: $export_audio_dir${reset}"
fi
