#!/bin/bash
#
# File: eval_pvad_with_different_ae.sh
# 
# Evaluate a trained VAD SET-AE model with different autoencoder weights
# without retraining the VAD model.
#
# This script allows you to:
# 1. Test different autoencoder models with the same VAD model
# 2. Compare performance across different AE training objectives
# 3. Quickly iterate on autoencoder improvements
#
# Usage:
#   bash eval_pvad_with_different_ae.sh <test_dir> <embed_dir>
#   bash eval_pvad_with_different_ae.sh data/test embeddings
#

# Colors
red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
blue=`tput setaf 4`
reset=`tput sgr0`

# Configuration
BASE=$PWD
VAD_MODEL_PATH="vad_set_ae.pt"  # The trained VAD model
SCORE_TYPE=1  # 0=baseline, 1=PC, 2=LI

# Get test directory and embeddings path from arguments or use defaults
TEST_DIR=${1:-"data/test"}
EMBED_PATH=${2:-"embeddings"}

# List of autoencoder models to test
AE_MODELS=(
    "src/AE_test/test_outputs/dvector_ae_many_main_w_other_singles"
    "src/AE_test/test_outputs/dvector_ae_sumNotmain_w_other_singles"
    "src/AE_test/test_outputs/dvector_autoencoder_non_target_aware_zero"
)

# Check if VAD model exists
if [ ! -f "$VAD_MODEL_PATH" ]; then
    echo "${red}Error: VAD model not found at $VAD_MODEL_PATH${reset}"
    echo "${yellow}Please train a VAD model first using:${reset}"
    echo "${yellow}  bash train_pvad.sh set_ae${reset}"
    exit 1
fi

echo "${green}========================================${reset}"
echo "${green}VAD SET-AE EVALUATION${reset}"
echo "${green}Swappable Autoencoder Weights${reset}"
echo "${green}========================================${reset}"
echo ""
echo "${yellow}VAD Model: $VAD_MODEL_PATH${reset}"
echo "${yellow}Test Dir: $TEST_DIR${reset}"
echo "${yellow}Embeddings: $EMBED_PATH${reset}"
echo "${yellow}Score Type: $SCORE_TYPE${reset}"
echo ""

# Create results file
RESULTS_FILE="ae_comparison_results.txt"
echo "VAD SET-AE Autoencoder Comparison Results" > $RESULTS_FILE
echo "=========================================" >> $RESULTS_FILE
echo "VAD Model: $VAD_MODEL_PATH" >> $RESULTS_FILE
echo "Test Dir: $TEST_DIR" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Test each autoencoder model
for ae_model in "${AE_MODELS[@]}"; do
    echo ""
    echo "${blue}========================================${reset}"
    echo "${blue}Testing Autoencoder:${reset}"
    echo "${blue}$ae_model${reset}"
    echo "${blue}========================================${reset}"
    
    # Check if model exists
    if [ ! -d "$ae_model" ]; then
        echo "${red}⚠️  Model not found, skipping...${reset}"
        echo "" >> $RESULTS_FILE
        echo "Model: $ae_model" >> $RESULTS_FILE
        echo "Status: NOT FOUND" >> $RESULTS_FILE
        continue
    fi
    
    # Run evaluation
    echo "" >> $RESULTS_FILE
    echo "Model: $ae_model" >> $RESULTS_FILE
    echo "----------------------------------------" >> $RESULTS_FILE
    
    python src/vad_set_ae_eval.py \
        --vad_model $VAD_MODEL_PATH \
        --ae_model_path $ae_model \
        --test_dir $TEST_DIR \
        --embed_path $EMBED_PATH \
        --score_type $SCORE_TYPE \
        2>&1 | tee -a temp_output.txt
    
    # Extract results and add to comparison file
    grep -E "(Accuracy|mAP|Per-class AP)" temp_output.txt >> $RESULTS_FILE
    rm temp_output.txt
    
    echo ""
done

echo ""
echo "${green}========================================${reset}"
echo "${green}✅ COMPARISON COMPLETE${reset}"
echo "${green}========================================${reset}"
echo ""
echo "${yellow}Results saved to: $RESULTS_FILE${reset}"
echo ""
echo "${yellow}Summary:${reset}"
cat $RESULTS_FILE | grep -E "(Model:|Accuracy|mAP:)" | head -20

echo ""
echo "${blue}To test a specific autoencoder model:${reset}"
echo "${blue}  python src/vad_set_ae_eval.py \\${reset}"
echo "${blue}    --vad_model $VAD_MODEL_PATH \\${reset}"
echo "${blue}    --ae_model_path path/to/your/ae/model \\${reset}"
echo "${blue}    --test_dir $TEST_DIR \\${reset}"
echo "${blue}    --embed_path $EMBED_PATH${reset}"
