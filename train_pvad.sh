#!/bin/bash
#
# File: train_pvad.sh
# Author: Simon Sedlacek
# Email: xsedla1h@stud.fit.vutbr.cz
#
# This is a demonstration script for personal VAD training meant to show examples
# of how to begin training the different personal VAD architectures.
#
# The dataset used for training is the sample validation dataset provided for model
# evaluation. To properly train the models, a new training dataset has to be 
# generated first. When it is ready, feel free take inspiration in this script to
# write your own training script. The scripts will not save the models to avoid cluttering
# the repository.
#

#================ EDIT HERE =========================

# here, the scoring method for the st and set architectures can be specified
# 0 (baseline), 1 (PC), 2 (LI)
st_score_type=1
set_score_type=1

# Autoencoder model path for set_ae architecture
# Options:
#   - dvector_ae_many_main_w_other_singles: Multi-speaker model with averaging
#   - dvector_autoencoder_non_target_aware_zero: Non-target aware model
#   - dvector_ae_sumNotmain_w_other_singles: Custom trained model
ae_model_path="src/AE_test/test_outputs/dvector_ae_identity"

# Limit number of utterances (leave empty or set to 0 to use all)
MAX_TRAIN_UTTERANCES=""  # e.g., "1000" to use only 1000 training utterances
MAX_TEST_UTTERANCES=""   # e.g., "200" to use only 200 test utterances

# Training/test dataset selection
# Option 1: Use demo datasets (default)
USE_OVERLAP_DATASET=true
TRAIN_NAME=features_full
TEST_NAME=test

# Option 2: Use overlap dataset (set USE_OVERLAP_DATASET=true)
# Specify the feature directory name created by prepare_overlap_dataset.sh
# Example: overlap_50pct_babble_main84_50
# OVERLAP_TRAIN_NAME=overlap_50pct_aug_main84_1000
OVERLAP_TRAIN_NAME=84_ov_train_ov50pct_main84_data50pct_38utt_aug_500
# OVERLAP_TEST_NAME=overlap_50pct_babble_main84_50
OVERLAP_TEST_NAME=84_ov_test_ov50pct_main84_aug_100
#===================================================

# some colors..
red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
reset=`tput sgr0`

if [ -z "$1" ]; then
  echo "Please, in order to train a model, specify the target architecture."
  echo "Choose one of: vad, et, st, set, set_ae"
  exit 0
else
  arch=$1
fi

# move to the eval directory..
BASE=$PWD

# Determine which dataset to use
if [ "$USE_OVERLAP_DATASET" = true ]; then
  TRAIN_NAME=$OVERLAP_TRAIN_NAME
  TEST_NAME=$OVERLAP_TEST_NAME
  TRAIN_DIR=$BASE/data/$TRAIN_NAME
  TEST_DIR=$BASE/data/$TEST_NAME
  EMBED_DIR=$BASE/data/embeddings
  
  echo "${green}================================================${reset}"
  echo "${green}Training with OVERLAP DATASET${reset}"
  echo "${green}================================================${reset}"
  echo "${yellow}Train dataset: $TRAIN_NAME${reset}"
  echo "${yellow}Test dataset: $TEST_NAME${reset}"
  [ -n "$MAX_TRAIN_UTTERANCES" ] && echo "${yellow}Max train utterances: $MAX_TRAIN_UTTERANCES${reset}"
  [ -n "$MAX_TEST_UTTERANCES" ] && echo "${yellow}Max test utterances: $MAX_TEST_UTTERANCES${reset}"
  echo ""
  
  # Check if overlap dataset exists
  if [ ! -d "$TRAIN_DIR" ]; then
    echo "${red}Error: Overlap dataset not found at $TRAIN_DIR${reset}"
    echo "${yellow}Please run prepare_overlap_dataset.sh first to generate the dataset.${reset}"
    echo "${yellow}Example: bash prepare_overlap_dataset.sh 0${reset}"
    exit 1
  fi
else
  TRAIN_DIR=data/$TRAIN_NAME
  TEST_DIR=data/$TEST_NAME
  EMBED_DIR=embeddings
  
  echo "${green}================================================${reset}"
  echo "${green}Training with DEMO DATASET${reset}"
  echo "${green}================================================${reset}"
  echo "${yellow}Train dataset: $TRAIN_NAME${reset}"
  echo "${yellow}Test dataset: $TEST_NAME${reset}"
  [ -n "$MAX_TRAIN_UTTERANCES" ] && echo "${yellow}Max train utterances: $MAX_TRAIN_UTTERANCES${reset}"
  [ -n "$MAX_TEST_UTTERANCES" ] && echo "${yellow}Max test utterances: $MAX_TEST_UTTERANCES${reset}"
  echo ""
fi

# Change to appropriate directory
if [ "$USE_OVERLAP_DATASET" = true ]; then
  cd $BASE
else
  cd data/eval_dir
fi

if [[ $arch == "vad" ]]; then
  echo "${green}Beginning training the classic vad architecture.${reset}"
  cmd="python $BASE/src/vad.py --train_dir $TRAIN_DIR --test_dir $TEST_DIR"
  [ -n "$MAX_TRAIN_UTTERANCES" ] && cmd="$cmd --max_train_utterances $MAX_TRAIN_UTTERANCES"
  [ -n "$MAX_TEST_UTTERANCES" ] && cmd="$cmd --max_test_utterances $MAX_TEST_UTTERANCES"
  eval $cmd
fi

if [[ $arch == "et" ]]; then
  echo "${green}Beginning training the ET vad architecture.${reset}"
  cmd="python $BASE/src/vad_et.py --embed_path $EMBED_DIR --train_dir $TRAIN_DIR --test_dir $TEST_DIR"
  [ -n "$MAX_TRAIN_UTTERANCES" ] && cmd="$cmd --max_train_utterances $MAX_TRAIN_UTTERANCES"
  [ -n "$MAX_TEST_UTTERANCES" ] && cmd="$cmd --max_test_utterances $MAX_TEST_UTTERANCES"
  eval $cmd
fi

if [[ $arch == "st" ]]; then
  echo "${green}Beginning training the ST vad architecture.${reset}"
  echo "${yellow}Using scoring method" $st_score_type "${reset}"
  cmd="python $BASE/src/vad_st.py --score_type $st_score_type --train_dir $TRAIN_DIR --test_dir $TEST_DIR"
  [ -n "$MAX_TRAIN_UTTERANCES" ] && cmd="$cmd --max_train_utterances $MAX_TRAIN_UTTERANCES"
  [ -n "$MAX_TEST_UTTERANCES" ] && cmd="$cmd --max_test_utterances $MAX_TEST_UTTERANCES"
  eval $cmd
fi

if [[ $arch == "set" ]]; then
  echo "${green}Beginning training the SET vad architecture.${reset}"
  echo "${yellow}Using scoring method" $set_score_type "${reset}"
  cmd="python $BASE/src/vad_set.py --embed_path $EMBED_DIR --score_type $set_score_type --train_dir $TRAIN_DIR --test_dir $TEST_DIR"
  [ -n "$MAX_TRAIN_UTTERANCES" ] && cmd="$cmd --max_train_utterances $MAX_TRAIN_UTTERANCES"
  [ -n "$MAX_TEST_UTTERANCES" ] && cmd="$cmd --max_test_utterances $MAX_TEST_UTTERANCES"
  eval $cmd
fi

if [[ $arch == "set_ae" ]]; then
  echo "${green}Beginning training the SET-AE vad architecture (with autoencoder-compressed d-vectors).${reset}"
  echo "${yellow}Using scoring method" $set_score_type "${reset}"
  echo "${yellow}Autoencoder model path:" $ae_model_path "${reset}"
  
  # Check if autoencoder model exists
  if [ ! -d "$BASE/$ae_model_path" ]; then
    echo "${red}Error: Autoencoder model not found at $BASE/$ae_model_path${reset}"
    echo "${yellow}Available models in src/AE_test/test_outputs/:${reset}"
    ls -1 $BASE/src/AE_test/test_outputs/ 2>/dev/null || echo "${red}Directory not found${reset}"
    exit 1
  fi
  
  cmd="python $BASE/src/vad_set_ae.py --embed_path $EMBED_DIR --score_type $set_score_type --ae_model_path $ae_model_path --train_dir $TRAIN_DIR --test_dir $TEST_DIR --use_autoencoder"
  [ -n "$MAX_TRAIN_UTTERANCES" ] && cmd="$cmd --max_train_utterances $MAX_TRAIN_UTTERANCES"
  [ -n "$MAX_TEST_UTTERANCES" ] && cmd="$cmd --max_test_utterances $MAX_TEST_UTTERANCES"
  eval $cmd
fi
