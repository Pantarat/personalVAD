#!/bin/bash
#
# File: prepare_kaldi_dir.sh
# Author: Simon Sedlacek
# Email: xsedla1h@stud.fit.vutbr.cz
#
# NOTE: before running this script, please make SURE that you have kaldi downloaded
# and compiled in the personal VAD project root folder (the kaldi/ folder should be on
# the same level as the data/ and src/ directories among others..) 
# 
# This script will create a kaldi project folder for personal VAD data augmentation
# purposes, copy and create necessary symlinks.
#
# The created kaldi project folder will be (relative to the pvad repo root) kaldi/egs/pvad/.
#
# After running this script, data augmentation will be possible
#


# Allow using an external Kaldi installation via KALDI_HOME env var.
# If not set, fall back to local ./kaldi

set -e

REPO_ROOT=$(pwd)
if [[ -n "${KALDI_HOME}" ]]; then
  KALDI_PATH="${KALDI_HOME}"
else
  KALDI_PATH="$REPO_ROOT/kaldi"
fi

if [[ ! -d "$KALDI_PATH" ]]; then
  echo "Kaldi root not found at $KALDI_PATH"
  echo "Set KALDI_HOME to your Kaldi path (e.g., /mnt/c/Work/Coding/Diarization/kaldi) or clone Kaldi into $REPO_ROOT/kaldi"
  exit 1
fi

echo "Using Kaldi at: $KALDI_PATH"

mkdir -p "$KALDI_PATH/egs"
cd "$KALDI_PATH/egs"

# Copy the pvad recipe directory into Kaldi egs if not already present
if [[ ! -d pvad ]]; then
  # The pvad recipe folder content is stored in repo src/kaldi/egs/pvad
  if [[ -d "$REPO_ROOT/src/kaldi/egs/pvad" ]]; then
    cp -r "$REPO_ROOT/src/kaldi/egs/pvad" .
  else
    echo "Source pvad recipe not found at $REPO_ROOT/src/kaldi/egs/pvad"
    exit 1
  fi
fi

cd pvad

# create symlinks for kaldi binaries and utilities from wsj s5
if [[ ! -d steps ]]; then
  if [[ -d ../wsj/s5/steps ]]; then
    ln -s ../wsj/s5/steps steps
  else
    echo "Cannot find ../wsj/s5/steps. Ensure wsj s5 exists inside your Kaldi egs."
    exit 1
  fi
fi
if [[ ! -d utils ]]; then
  if [[ -d ../wsj/s5/utils ]]; then
    ln -s ../wsj/s5/utils utils
  else
    echo "Cannot find ../wsj/s5/utils. Ensure wsj s5 exists inside your Kaldi egs."
    exit 1
  fi
fi

# link the pvad repo src for helper scripts
if [[ ! -L src ]]; then
  ln -s "$REPO_ROOT/src" src
fi

echo "Kaldi pvad directory prepared at: $KALDI_PATH/egs/pvad"
