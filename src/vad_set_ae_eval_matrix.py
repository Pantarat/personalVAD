"""@package vad_set_ae_eval_matrix

Matrix evaluation script for SET-AE models - varies both autoencoder and VAD models.

This script evaluates all combinations of AE models and VAD models, producing
two result matrices: one for mAP and one for accuracy.

Usage:
    # Edit AE_MODEL_LIST and VAD_MODEL_LIST in the file, then run:
    python vad_set_ae_eval_matrix.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
from pathlib import Path
import csv
from datetime import datetime
import pandas as pd

# Import from vad_set_ae_eval to reuse configuration
from vad_set_ae_eval import (
    DATA_TEST, 
    EMBED_PATH,
    SCORE_TYPE,
    BATCH_SIZE,
    NUM_WORKERS,
    USE_AE_RECONSTRUCTION,
    device
)

# Import from vad_set_ae
from vad_set_ae import VadSETAEDataset, load_autoencoder, pad_collate_with_metadata
from personal_vad import PersonalVAD

# ============================================================================
# CONFIGURATION - Edit these lists to specify which models to evaluate
# ============================================================================

# List of AE models to evaluate
AE_MODEL_LIST = [
    'NO_AE',  # Baseline without autoencoder
    'src/AE_test/test_outputs/dvector_ae_sumNotmain_500',
    'src/AE_test/test_outputs/dvector_ae_sumNotmain_1000',
    'src/AE_test/test_outputs/dvector_ae_sumNotmain_1500',
    'src/AE_test/test_outputs/dvector_ae_sumNotmain_3000',
    'src/AE_test/test_outputs/dvector_ae_sumNotmain_7500',
    'src/AE_test/test_outputs/dvector_ae_sumNotmain_15000',
    'src/AE_test/test_outputs/dvector_ae_sumNotmain_30000',
]

# List of VAD models to evaluate
VAD_MODEL_LIST = [
    'data/eval_dir/models/vad_set_tanh_score1_10ep.pt',
    'vad_set_overlap_trained/vad_set_84_overlap_tanh_score1_200_new.pt',
    'vad_set_overlap_trained/vad_set_84_overlap_tanh_score1_500_new.pt',
    'vad_set_overlap_trained/vad_set_84_overlap_tanh_score1_1000_new.pt',
    'vad_set_overlap_trained/vad_set_84_overlap_tanh_score1_2000_new.pt',
    'vad_set_overlap_trained/vad_set_84_overlap_tanh_score1_5000_new.pt',
    'vad_set_overlap_trained/vad_set_84_overlap_tanh_score1_10000_new.pt',
]

# Output directory and filename prefix
OUTPUT_DIR = 'model_evaluation_results'
OUTPUT_PREFIX = 'matrix_eval'

# Other settings
QUIET = False
FILTER_MAIN_SPEAKER = None  # Set to speaker ID like '84' to filter, or None for all speakers

# Score recomputation settings
# WARNING: Requires wav.scp in test directory or valid AUDIO_ROOT path
# Set to False to use pre-computed scores.scp (faster, works without audio files)
RECOMPUTE_SCORES = False  # Set to True only if you have audio files available
AUDIO_ROOT = None  # Path to audio root (None = auto-detect from test_dir/wav.scp)


def evaluate_model_combination(vad_model_path, ae_model_path, test_data, test_loader_template, recompute_scores=True):
    """
    Evaluate a single combination of VAD model and AE model.
    
    Args:
        recompute_scores: If True, recompute scores with AE. If False, use pre-computed scores.scp
    
    Returns:
        dict: Results including accuracy, mAP, and AP scores
    """
    # Determine if using autoencoder
    if ae_model_path == 'NO_AE':
        use_autoencoder = False
        use_ae_reconstruction = False
    else:
        use_autoencoder = True
        use_ae_reconstruction = USE_AE_RECONSTRUCTION
    
    # Load autoencoder and update dataset
    if use_autoencoder:
        autoencoder, ae_config = load_autoencoder(ae_model_path, device)
        autoencoder.eval()
        encoded_dim = ae_config['hidden_dims'][len(ae_config['hidden_dims'])//2]
        
        if use_ae_reconstruction:
            input_dim = 297  # 40 fbanks + 256 reconstructed + 1 score
        else:
            input_dim = 105  # 40 fbanks + 64 compressed + 1 score
    else:
        autoencoder = None
        use_ae_reconstruction = False
        encoded_dim = 256
        ae_config = {}
        input_dim = 297  # 40 fbanks + 256 full + 1 score
    
    # Update the dataset's autoencoder and reprocess embeddings
    actual_dataset = test_data.dataset if isinstance(test_data, Subset) else test_data
    
    actual_dataset.autoencoder = autoencoder
    actual_dataset.use_autoencoder = use_autoencoder
    actual_dataset.use_ae_reconstruction = use_ae_reconstruction
    
    # Reprocess enrolled d-vectors with the new autoencoder
    # Note: When recompute_scores=True, stream d-vectors are processed in __getitem__
    if use_autoencoder and autoencoder is not None:
        if use_ae_reconstruction:
            # Full reconstruction: keep enrolled as original, stream will be reconstructed in __getitem__
            actual_dataset.processed_embed = None
        else:
            # Bottleneck mode: compress enrolled d-vectors
            actual_dataset.processed_embed = {}
            autoencoder.eval()
            with torch.no_grad():
                for target, dvector in actual_dataset.embed.items():
                    dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0)
                    compressed = autoencoder.encode(dvector_tensor)
                    actual_dataset.processed_embed[target] = compressed.cpu().numpy().squeeze()
    else:
        actual_dataset.processed_embed = None
    
    # Create DataLoader
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate_with_metadata
    )
    
    # Load VAD model
    hidden_dim = 64
    num_layers = 2
    out_dim = 3
    
    vad_model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=True, linear=False)
    
    # Load weights
    if device == torch.device('cuda') and torch.cuda.is_available():
        checkpoint = torch.load(vad_model_path)
    else:
        checkpoint = torch.load(vad_model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Verify input dimension matches
        if 'lstm.weight_ih_l0' in state_dict:
            trained_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
            if trained_input_dim != input_dim:
                print(f"\n⚠️  WARNING: Input dimension mismatch!")
                print(f"   Model trained with: {trained_input_dim}-dim")
                print(f"   Current config expects: {input_dim}-dim")
                print(f"   Skipping this combination...")
                return None
        
        vad_model.load_state_dict(state_dict)
    else:
        vad_model.load_state_dict(checkpoint)
    
    vad_model = vad_model.to(device)
    vad_model.eval()
    
    # Evaluate
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        targets = []
        outputs = []
        
        for x_padded, y_padded, x_lens, y_lens, keys in test_loader:
            y_padded = y_padded.to(device)
            out_padded, _ = vad_model(x_padded.to(device), x_lens, None)
            
            for j in range(out_padded.size(0)):
                p = softmax(out_padded[j][:y_lens[j]])
                outputs.append(p.cpu().numpy())
                targets.append(y_padded[j][:y_lens[j]].cpu().numpy())
        
        targets = np.concatenate(targets)
        outputs = np.concatenate(outputs)
        predictions = np.argmax(outputs, axis=1)
        
        # Compute metrics
        targets_oh = np.eye(3)[targets]
        
        # Average precision per class
        from sklearn.metrics import average_precision_score
        out_AP = average_precision_score(targets_oh, outputs, average=None)
        mAP = average_precision_score(targets_oh, outputs, average='micro')
        
        # Accuracy
        accuracy = (predictions == targets).mean() * 100
    
    return {
        'accuracy': accuracy,
        'mAP': mAP * 100,
        'AP_NS': out_AP[0] * 100,
        'AP_NTSS': out_AP[1] * 100,
        'AP_TSS': out_AP[2] * 100,
    }


def format_model_name(model_path, is_ae=False):
    """Format model name for display in matrix"""
    if model_path == 'NO_AE':
        return 'NO_AE'
    
    basename = os.path.basename(model_path)
    
    if is_ae:
        # For AE models, extract key info
        basename = basename.replace('dvector_ae_', '')
        return basename
    else:
        # For VAD models, extract key info
        basename = basename.replace('vad_set_84_overlap_', '').replace('.pt', '')
        return f'VAD_{basename}'


if __name__ == '__main__':
    print("=" * 80)
    print("VAD SET-AE MATRIX EVALUATION")
    print("=" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   Test Dir: {DATA_TEST}")
    print(f"   Score Type: {SCORE_TYPE}")
    if FILTER_MAIN_SPEAKER:
        print(f"   Main Speaker: {FILTER_MAIN_SPEAKER} (filtering enabled)")
    print(f"\n📋 Matrix Evaluation Mode:")
    print(f"   AE models: {len(AE_MODEL_LIST)}")
    print(f"   VAD models: {len(VAD_MODEL_LIST)}")
    print(f"   Total combinations: {len(AE_MODEL_LIST) * len(VAD_MODEL_LIST)}")
    
    # Set audio root to test directory (where wav.scp should be located)
    AUDIO_ROOT_OVERRIDE = DATA_TEST if RECOMPUTE_SCORES else None
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load test dataset once
    print(f"\n📂 Loading test dataset...")
    if RECOMPUTE_SCORES:
        print(f"   Score recomputation: ENABLED (scores will be computed with each AE)")
    else:
        print(f"   ⚠️  Score recomputation: DISABLED (using pre-computed scores.scp - results will be identical!)")
    
    test_data = VadSETAEDataset(
        DATA_TEST,
        EMBED_PATH,
        SCORE_TYPE,
        autoencoder=None,
        use_autoencoder=False,
        use_ae_reconstruction=False,
        recompute_scores=RECOMPUTE_SCORES,
        audio_root=AUDIO_ROOT_OVERRIDE
    )
    
    # Filter by main_speaker if specified
    if FILTER_MAIN_SPEAKER:
        print(f"🔍 Filtering dataset for speaker: {FILTER_MAIN_SPEAKER}")
        original_indices = []
        for idx in range(len(test_data)):
            key = test_data.keys[idx]
            speaker_id = key.split('-')[0]
            if speaker_id == FILTER_MAIN_SPEAKER:
                original_indices.append(idx)
        
        if len(original_indices) == 0:
            print(f"\n❌ ERROR: No samples found for speaker '{FILTER_MAIN_SPEAKER}' in test set!")
            sys.exit(1)
        
        test_data = Subset(test_data, original_indices)
        print(f"   Found {len(test_data)} samples for speaker {FILTER_MAIN_SPEAKER}")
    
    print(f"✅ Test samples: {len(test_data)}")
    
    # Initialize result matrices
    n_ae = len(AE_MODEL_LIST)
    n_vad = len(VAD_MODEL_LIST)
    
    accuracy_matrix = np.zeros((n_vad, n_ae))
    mAP_matrix = np.zeros((n_vad, n_ae))
    
    # Row and column labels
    vad_labels = [format_model_name(m, is_ae=False) for m in VAD_MODEL_LIST]
    ae_labels = [format_model_name(m, is_ae=True) for m in AE_MODEL_LIST]
    
    # Store detailed results
    all_results = []
    
    # Main evaluation loop
    total_combinations = n_ae * n_vad
    current_combination = 0
    
    for vad_idx, vad_model_path in enumerate(VAD_MODEL_LIST):
        for ae_idx, ae_model_path in enumerate(AE_MODEL_LIST):
            current_combination += 1
            
            vad_name = format_model_name(vad_model_path, is_ae=False)
            ae_name = format_model_name(ae_model_path, is_ae=True)
            
            print(f"\n{'='*80}")
            print(f"📦 Evaluating combination {current_combination}/{total_combinations}")
            print(f"   VAD: {vad_name}")
            print(f"   AE:  {ae_name}")
            print(f"{'='*80}")
            
            # Evaluate this combination
            result = evaluate_model_combination(
                vad_model_path, 
                ae_model_path, 
                test_data, 
                None,
                recompute_scores=RECOMPUTE_SCORES
            )
            
            if result is None:
                print(f"⚠️  Skipped due to incompatibility")
                accuracy_matrix[vad_idx, ae_idx] = np.nan
                mAP_matrix[vad_idx, ae_idx] = np.nan
                continue
            
            # Store in matrices
            accuracy_matrix[vad_idx, ae_idx] = result['accuracy']
            mAP_matrix[vad_idx, ae_idx] = result['mAP']
            
            # Store detailed results
            all_results.append({
                'vad_model': vad_name,
                'vad_model_path': vad_model_path,
                'ae_model': ae_name,
                'ae_model_path': ae_model_path if ae_model_path != 'NO_AE' else 'N/A',
                'accuracy': result['accuracy'],
                'mAP': result['mAP'],
                'AP_NS': result['AP_NS'],
                'AP_NTSS': result['AP_NTSS'],
                'AP_TSS': result['AP_TSS'],
            })
            
            print(f"✓ Accuracy: {result['accuracy']:.2f}% | mAP: {result['mAP']:.2f}%")
    
    # Save detailed results CSV
    csv_output = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_detailed_{timestamp}.csv')
    df_detailed = pd.DataFrame(all_results)
    df_detailed.to_csv(csv_output, index=False)
    print(f"\n✅ Detailed results saved to: {csv_output}")
    
    # Save accuracy matrix
    accuracy_csv = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_accuracy_{timestamp}.csv')
    df_accuracy = pd.DataFrame(accuracy_matrix, index=vad_labels, columns=ae_labels)
    df_accuracy.to_csv(accuracy_csv)
    print(f"✅ Accuracy matrix saved to: {accuracy_csv}")
    
    # Save mAP matrix
    mAP_csv = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_mAP_{timestamp}.csv')
    df_mAP = pd.DataFrame(mAP_matrix, index=vad_labels, columns=ae_labels)
    df_mAP.to_csv(mAP_csv)
    print(f"✅ mAP matrix saved to: {mAP_csv}")
    
    # Print matrices to console
    print("\n" + "=" * 80)
    print("📊 ACCURACY MATRIX (%)")
    print("=" * 80)
    print("\nRows: VAD models | Columns: AE models\n")
    print(df_accuracy.to_string(float_format=lambda x: f'{x:.2f}'))
    
    print("\n" + "=" * 80)
    print("📊 mAP MATRIX (%)")
    print("=" * 80)
    print("\nRows: VAD models | Columns: AE models\n")
    print(df_mAP.to_string(float_format=lambda x: f'{x:.2f}'))
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print(f"   - {os.path.basename(csv_output)}")
    print(f"   - {os.path.basename(accuracy_csv)}")
    print(f"   - {os.path.basename(mAP_csv)}")
