"""@package vad_set_ae_eval_batch

Batch evaluation script for SET-AE models with swappable autoencoder weights.

This script allows you to evaluate multiple AE models at once and output CSV results.

Usage:
    # Edit AE_MODEL_LIST and other settings in the file, then run:
    python vad_set_ae_eval_batch.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
import shutil
import re
from pathlib import Path
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Import from vad_set_ae_eval to reuse evaluation logic
from vad_set_ae_eval import (
    evaluate_vad_with_ae, 
    VAD_MODEL_PATH, 
    DATA_TEST, 
    EMBED_PATH,
    SCORE_TYPE,
    BATCH_SIZE,
    USE_AE_RECONSTRUCTION,
    RECOMPUTE_SCORES,
    MAIN_SPEAKER_ID,
    BINARY_CLASSIFICATION,
    device,
    TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION,
    TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION,
)

# Import from vad_set_ae
try:
    from AE_test.autoencoder_utils import DvectorAutoencoder, load_autoencoder
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AE_test'))
    from autoencoder_utils import DvectorAutoencoder, load_autoencoder
from vad_set_ae import VadSETAEDataset, pad_collate_with_metadata
from personal_vad import PersonalVAD

# List of AE models to evaluate (can be edited directly in this file)
AE_MODEL_LIST = [
    # 'dvector_ae-84_100pct_76utt-300otherSpk_100pctmainspk_100pctAmp-mainOnly_withNoise-2000',
    # 'dvector_ae-84_100pct_76utt-300otherSpk_100pctmainspk_100pctAmp-mainOnly_otherSingles_withNoise-2000',
    # 'dvector_ae-84_100pct_76utt-300otherSpk_5pctmainspk_100pctAmp-manyMain_withNoise-2000',
    # 'dvector_ae-84_100pct_76utt-300otherSpk_5pctmainspk_100pctAmp-manyMain_otherSingles_withNoise-2000',
    # 'dvector_ae-84_100pct_76utt-300otherSpk_5pctmainspk_100pctAmp-sumNotmain_withNoise-2000',
    # 'dvector_ae-84_100pct_76utt-300otherSpk_5pctmainspk_100pctAmp-sumNotmain_otherSingles_withNoise-2000',
    # 'dvector_ae-84_100pct_76utt-300otherSpk_1pctmainspk_100pctAmp-sumNotmain_otherSingles_withNoise-2000',
    # 'dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000',
    # 'dvector_ae_identity_50x1Dev_noOV_0,01s',
    # 'dvector_ae_identity_1000x1Dev_noOV_1,6s',
    # 'dvector_ae_identity_1000x1Dev_noOV_1,6s_100latent',
    # 'dvector_ae_identity_600x30Dev_noOV_1,6s',
    # 'dvector_ae_identity_600x30Dev_noOV',
    # 'dvector_ae_identity_1000x40Dev_noOV_1,6s',
    # '../../dvector_ae_identity_1100x50Dev_noOV_1,6s_20-2-27',
    # '../dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_1-4-26',
    # '../dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_v2_10-4-26',
    # '../dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_v2_12-4-26',
    # '../dvector_ae_intermediate_babble_finetune_18-4-26', # Different than pretrain data (0.3 mse, 0.7 cos, lr 1e-4 0-15 SNR)
    # '../dvector_ae_intermediate_babble_finetune_20-4-26', # Different than pretrain data (0.1 mse, 0.9 cos, lr 1e-4 0-15 SNR)
    # '../dvector_ae_intermediate_babble_finetune_21-4-26', # Same as pretrain data (0.3 mse, 0.7 cos, lr 1e-5 0-15 SNR)
    # '../dvector_ae_intermediate_babble_finetune_22-4-26', # Same as pretrain data (0.3 mse, 0.7 cos, lr 1e-5 (20.0, 17.0, 15.0, 13.0)SNR)
    # '../dvector_ae_deep_stacked_libri_babble_1_29-4-26',
    # '../dvector_ae_deep_stacked_libri_babble_2_29-4-26',
    # '../dvector_ae_deep_stacked_libri_babble_3_30-4-26',
    # '../dvector_ae_deep_stacked_libri_babble_4_30-4-26',
    # '../dvector_ae_deep_stacked_libri_babble_5_30-4-26',
    # '../dvector_ae_deep_stacked_libri_babble_6_30-4-26',
    # '../dvector_ae_deep_stacked_libri_babble_7_1-5-26',
    # '../dvector_ae_deep_stacked_libri_babble_8_1-5-26',
    # '../dvector_ae_deep_stacked_libri_babble_9_1-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_1_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_2_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_3_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_4_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_5_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_6_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_7_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_8_2-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_9_5-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_10_5-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_11_5-5-26',
    # '../greedy/dvector_ae_greedy_layerwise_14_6-5-26',

    # '../../greedy/dvector_ae_greedy_layerwise_15_6-5-26',
    # 'dvector_ae-908_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # '../61/dvector_ae-61_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # 'dvector_ae-908_8pct_6utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # 'dvector_ae-908_12pct_9utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # 'dvector_ae-908_16pct_12utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # 'dvector_ae-908_20pct_15utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # 'dvector_ae-908_40pct_30utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # 'dvector_ae-908_60pct_45utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    # 'dvector_ae-908_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',


    # '../dvector_ae_deep_stacked_greedy_babble_10_4-5-26',
    # '../dvector_ae_deep_stacked_greedy_babble_11_4-5-26',
    
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_5ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_10ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_20ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_30ep',
    
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-manyMain_pretrain-20000_0.00001lr_10ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-manyMain_pretrain-20000_0.00001lr_30ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-manyMain_pretrain-20000_0.00001lr_50ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-manyMain_pretrain-20000_0.00001lr_200ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-manyMain_pretrain-20000_0.00001lr_500ep',
    
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-sumNotmain_pretrain-20000_0.00001lr_10ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-sumNotmain_pretrain-20000_0.00001lr_30ep',
    # 'dvector_ae-84_100pct_76utt-50spk+300Dev_5s_5pctmainspk_100pctAmp-sumNotmain_pretrain-20000_0.00001lr_200ep',
    
    # 'dvector_ae-84_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_10ep',
    # 'dvector_ae-84_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_15ep',

    # 'dvector_ae-84_8pct_6utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_10ep',
    # 'dvector_ae-84_8pct_6utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_15ep',

    # 'greedy_finetunev4_61_1_29-5-26',

    # 'dvector_ae-61_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_50ep',
    # 'dvector_ae-61_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_50ep',
    # 'dvector_ae-61_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_50ep',
    # 'dvector_ae-61_20pct_15utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_50ep',
    # 'dvector_ae-61_40pct_30utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_50ep',
    # 'dvector_ae-61_60pct_45utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_50ep',
    # 'dvector_ae-61_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_50ep',

    # 'dvector_ae-908_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-908_8pct_6utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-908_12pct_9utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-908_16pct_12utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-908_20pct_15utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-908_40pct_30utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-908_60pct_45utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-908_80pct_60utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym',
    # 'dvector_ae-61_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infoNCE_asym23',

    # 'dvector_ae-1462_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_5ep(1)',
    # '../61/dvector_ae-61_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_5ep',
    # '../1462/dvector_ae-1462_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_5ep',
    # '../174/dvector_ae-174_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_5ep',
    # 'dvector_ae-84_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_0.000001lr_100ep',
    # 'dvector_ae-84_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_0.000001lr_1000ep',
    
    # 'dvector_ae-84_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_0.00001lr_50ep',
    # 'dvector_ae-84_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_0.00001lr_50ep(1)'
    'src/AE_test/test_outputs/models/greedy_finetune/121_v2_tests/12_28-7-26/'
    # '../61/dvector_ae-61_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_pretrain-2000_0.00001lr_5ep'
]

# AE_MODEL_LIST = [f"mse{0}p{0}_cos{j//10}p{j%10}_neg{(10-j)//10}p{(10-j)%10}" for j in range(11)]

# AE_MODEL_LIST = ['src/AE_test/test_outputs/models/greedy_finetune_newv2_sweep/61/' + model_name for model_name in AE_MODEL_LIST]

OUTPUT_CSV = 'model_evaluation_results/test.csv'

# Include no-AE baseline in evaluation
INCLUDE_NO_AE = False

# Score recomputation settings (imported from vad_set_ae_eval.py)
# To change: edit RECOMPUTE_SCORES in vad_set_ae_eval.py
# WARNING: Requires wav.scp in test directory or valid AUDIO_ROOT path
# Set to False to use pre-computed scores.scp (faster, works without audio files)
AUDIO_ROOT = "None"  # Path to audio root (None = auto-detect from test_dir/wav.scp)

# DataLoader settings
# IMPORTANT: When using GPU with score recomputation, set NUM_WORKERS=0
# Multi-processing doesn't work well with GPU models in DataLoader
NUM_WORKERS = 0  # Set to 0 for GPU compatibility with score recomputation

# Label example export settings
N_EXAMPLE_UTTERANCES = 3  # Number of utterances per model to export
LABEL_FRAME_STEP_SEC = 0.01  # Label frame duration in seconds (txt format: start\tend\tlabel)
EXAMPLE_LABELS_DIR = 'model_evaluation_results/example_labels'
SHOW_CONFUSION_MATRIX_PERCENT = True

# Similarity score plotting
PLOT_SIMILARITY_SCORES = True
SIMILARITY_PLOTS_DIR = 'model_evaluation_results/similarity_scores'
SIMILARITY_SCORE_SAMPLE_STEP = 5
SIMILARITY_SCORE_MAX_UTTS = 50

# Per-model activation override (tanh/relu). Leave empty to use config.
AE_ACTIVATION_OVERRIDE = {
    # 'dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_v2_12-4-26': 'relu',
}

# Reconstruction mode behavior for enrolled d-vectors
# TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION = False  # Safer default: avoids inflated target false positives
# TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION = False


def _get_ae_bottleneck_dim(ae_config):
    """Resolve bottleneck dimension from AE config for standard or stacked variants."""
    model_type = str(ae_config.get('model_type', 'dvector_autoencoder')).lower()
    if model_type in ('deep_stacked_dae', 'greedy_layerwise_stacked'):
        return int(ae_config.get('input_dim', 256))

    hidden_dims = ae_config.get('hidden_dims')
    if not hidden_dims:
        raise KeyError("AE config is missing non-empty 'hidden_dims'; cannot determine bottleneck dimension.")
    return int(hidden_dims[len(hidden_dims) // 2])


def _sanitize_filename(name):
    """Make a safe filename from model/key text."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')


def _write_label_sequence_txt(file_path, labels, frame_step_sec=0.01):
    """Write frame-wise labels in format: start\tend\tlabel."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for idx, label in enumerate(labels):
            start_t = idx * frame_step_sec
            end_t = (idx + 1) * frame_step_sec
            f.write(f"{start_t:.3f}\t{end_t:.3f}\t{int(label)}\n")


def _export_example_label_files(base_dir, model_name, examples, frame_step_sec=0.01):
    """
    Export 3 label files per utterance:
      - true label
      - predicted label
      - match/mismatch label (1=match, 0=mismatch)
    """
    model_dir = Path(base_dir) / _sanitize_filename(model_name)

    # Rewrite previous run's examples for this model
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for ex in examples:
        key_safe = _sanitize_filename(ex['key'])
        true_labels = np.asarray(ex['true'], dtype=np.int32)
        pred_labels = np.asarray(ex['pred'], dtype=np.int32)
        match_labels = (true_labels == pred_labels).astype(np.int32)

        _write_label_sequence_txt(model_dir / f"{key_safe}_true_label.txt", true_labels, frame_step_sec)
        _write_label_sequence_txt(model_dir / f"{key_safe}_predicted_label.txt", pred_labels, frame_step_sec)
        _write_label_sequence_txt(model_dir / f"{key_safe}_match_mismatch_label.txt", match_labels, frame_step_sec)

    return model_dir


def _plot_similarity_scores(model_name, score_dict_by_class, output_dir):
    if not score_dict_by_class:
        return None

    class_names = {0: 'NS', 1: 'NTSS', 2: 'TSS'}
    class_scores = {}
    for cls in (0, 1, 2):
        entries = score_dict_by_class.get(cls, [])
        if entries:
            class_scores[cls] = np.concatenate(entries)
        else:
            class_scores[cls] = np.array([], dtype=np.float32)

    if not any(scores.size for scores in class_scores.values()):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(model_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {0: 'steelblue', 1: 'darkorange', 2: 'seagreen'}
    for cls in (0, 1, 2):
        if class_scores[cls].size:
            axes[0].hist(
                class_scores[cls],
                bins=60,
                alpha=0.55,
                color=colors[cls],
                label=class_names[cls],
            )
    axes[0].set_title('Cosine Similarity Distribution (by label)')
    axes[0].set_xlabel('Cosine similarity')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    mean_vals = [np.mean(class_scores[c]) if class_scores[c].size else np.nan for c in (0, 1, 2)]
    axes[1].bar([class_names[c] for c in (0, 1, 2)], mean_vals, color=[colors[c] for c in (0, 1, 2)])
    axes[1].set_title('Mean Similarity by Label')
    axes[1].set_xlabel('True label')
    axes[1].set_ylabel('Mean cosine similarity')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'SV Similarity Scores: {model_name}', fontsize=12, fontweight='bold')
    fig.tight_layout()

    plot_path = output_dir / f'{safe_name}_similarity.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def _swap_tanh_to_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Tanh):
            setattr(module, name, nn.ReLU())
        else:
            _swap_tanh_to_relu(child)


def _apply_activation_override(model, activation_type):
    activation = str(activation_type).lower()
    if activation == 'relu':
        _swap_tanh_to_relu(model)
    elif activation == 'tanh':
        return
    else:
        raise ValueError(f"Unsupported activation override: {activation_type}")


def _apply_forced_inference_target(dataset, forced_target_speaker_id, quiet=False):
    """
    Force all utterances to use one target speaker for inference.

        In forced mode, ALL target labels are removed globally:
            - 2 (target speech) -> 1 (non-target speech)
        This produces labels with no target class.
    """
    if forced_target_speaker_id is None:
        return

    forced_target = str(forced_target_speaker_id).strip()
    if forced_target == '':
        return

    if forced_target not in dataset.embed:
        raise ValueError(
            f"Forced target speaker '{forced_target}' not found in enrolled embeddings ({dataset.embed_path}/dvectors.scp)."
        )

    # Track stats
    changed_target_utts = 0
    relabeled_utts = 0
    relabeled_frames_demoted_2_to_1 = 0

    # Keep original map (for debugging/reference)
    dataset.original_targets = dict(dataset.targets)
    dataset.forced_inference_target = forced_target
    dataset.forced_relabel_keys = set()

    for key in dataset.keys:
        original_target = str(dataset.targets[key])

        # Force target for scoring/enrolled embedding usage
        if original_target != forced_target:
            changed_target_utts += 1

        # Relabel is global in forced mode (actual relabel happens in dataset __getitem__).
        dataset.forced_relabel_keys.add(key)
        y = dataset.labels[key]
        n_tss = int(np.sum(y == 2))
        if n_tss > 0:
            relabeled_utts += 1
            relabeled_frames_demoted_2_to_1 += n_tss

        dataset.targets[key] = forced_target

    if not quiet:
        print(f"\n🎯 Forced inference target speaker enabled: {forced_target}")
        print(f"   Utterances with overridden target: {changed_target_utts}/{len(dataset.keys)}")
        print(f"   Utterances containing class-2 labels: {relabeled_utts}")
        print(f"   Frames relabeled (2→1): {relabeled_frames_demoted_2_to_1}")
        print(f"   Forced mode output labels contain no class-2 target frames")


def _print_confusion_matrix_pct(conf_cm_pct, binary_mode=False):
    """Print confusion matrix as percentage of total frames (base-1 normalization)."""
    if conf_cm_pct is None:
        print("Confusion matrix (% of total frames): unavailable")
        return

    if binary_mode:
        class_names = ['NonTarget', 'Target']
    else:
        class_names = ['NS', 'NTSS', 'TSS']

    print("Confusion matrix (% of total frames, rows=true, cols=pred):")
    print(f"{'True/Pred':<12}" + "".join([f"{name:>12}" for name in class_names]))

    for i, true_name in enumerate(class_names):
        row_vals = "".join([f"{(100.0 * conf_cm_pct[i, j]):>11.2f}%" for j in range(len(class_names))])
        print(f"{true_name:<12}{row_vals}")


def _get_csv_fieldnames(binary_mode):
    fieldnames = [
        'ae_model_name',
        'ae_model_path',
        'mAP',
        'accuracy',
        'precision_micro',
        'recall_micro',
        'f1_micro',
    ]

    if binary_mode:
        fieldnames.extend([
            'AP_NonTarget',
            'AP_Target',
            'precision_NonTarget',
            'recall_NonTarget',
            'f1_NonTarget',
            'precision_Target',
            'recall_Target',
            'f1_Target',
            'conf_NT_as_NT',
            'conf_NT_as_T',
            'conf_T_as_NT',
            'conf_T_as_T',
        ])
    else:
        fieldnames.extend([
            'AP_NS',
            'AP_NTSS',
            'AP_TSS',
            'precision_NS',
            'recall_NS',
            'f1_NS',
            'precision_NTSS',
            'recall_NTSS',
            'f1_NTSS',
            'precision_TSS',
            'recall_TSS',
            'f1_TSS',
            'conf_NSasNS',
            'conf_NSasNTSS',
            'conf_NSasTSS',
            'conf_NTSSasNS',
            'conf_NTSSasNTSS',
            'conf_NTSSasTSS',
            'conf_TSSasNS',
            'conf_TSSasNTSS',
            'conf_TSSasTSS',
        ])

    return fieldnames


def _load_existing_model_names(output_path):
    """Return set of model names already present in the CSV."""
    existing = set()
    if not os.path.exists(output_path):
        return existing

    try:
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or 'ae_model_name' not in reader.fieldnames:
                return existing
            for row in reader:
                name = row.get('ae_model_name')
                if name:
                    existing.add(name)
    except OSError:
        return existing

    return existing


def init_results_csv(output_path, binary_mode=False):
    """Create/overwrite CSV and write header only."""
    fieldnames = _get_csv_fieldnames(binary_mode)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
    print(f"\nResults CSV initialized: {output_path}")


def append_result_to_csv(result, output_path, binary_mode=False):
    """Append a single result row to the CSV."""
    if not result:
        return

    fieldnames = _get_csv_fieldnames(binary_mode)
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writerow(result)


def write_results_to_csv(results_list, output_path, binary_mode=False):
    """Write evaluation results to CSV file"""
    if not results_list:
        print("No results to write!")
        return

    init_results_csv(output_path, binary_mode=binary_mode)
    for result in results_list:
        append_result_to_csv(result, output_path, binary_mode=binary_mode)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    # Configuration - Edit these values directly in the file
    # OUTPUT_CSV = 'model_evaluation_results/set_ae_sumNotmain_vary_percent_main_babble_main84_100_results.csv'

    QUIET = False
    FILTER_MAIN_SPEAKER = None  # Set to speaker ID like '84' to filter, or None for all speakers
    BINARY_MODE = BINARY_CLASSIFICATION  # True: 2-class (target vs non-target), False: 3-class
    FORCE_INFERENCE_TARGET_SPEAKER_ID = None  # e.g. '84' to force inference target across all utterances
    FORCE_RECOMPUTE_SCORES_WITH_FORCED_TARGET = False  # Recommended: recompute scores when forcing target speaker

    force_target_enabled = FORCE_INFERENCE_TARGET_SPEAKER_ID is not None and str(FORCE_INFERENCE_TARGET_SPEAKER_ID).strip() != ''
    effective_recompute_scores = RECOMPUTE_SCORES or (FORCE_RECOMPUTE_SCORES_WITH_FORCED_TARGET and force_target_enabled)
    
    # Set audio root to test directory (where wav.scp should be located)
    # Override this if wav.scp is in a different location
    AUDIO_ROOT_OVERRIDE = DATA_TEST if effective_recompute_scores else None
    
    # Prepare list of models to evaluate
    ae_models_to_evaluate = AE_MODEL_LIST.copy()
    if INCLUDE_NO_AE:
        ae_models_to_evaluate.append('NO_AE')
    
    print("=" * 80)
    print("VAD SET-AE BATCH EVALUATION")
    print("=" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   VAD Model: {VAD_MODEL_PATH}")
    print(f"   Test Dir: {DATA_TEST}")
    print(f"   Score Type: {SCORE_TYPE}")
    print(f"   Reconstruction enrolled transform (scoring): {TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION}")
    print(f"   Reconstruction enrolled transform (VAD input): {TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION}")
    print(f"   Forced Inference Target: {FORCE_INFERENCE_TARGET_SPEAKER_ID}")
    if force_target_enabled and not RECOMPUTE_SCORES and FORCE_RECOMPUTE_SCORES_WITH_FORCED_TARGET:
        print(f"   ⚠️  Forcing recompute_scores=True because forced target is enabled")
    if FILTER_MAIN_SPEAKER:
        print(f"   Main Speaker: {FILTER_MAIN_SPEAKER} (filtering enabled)")
    print(f"\n📋 Batch Evaluation Mode:")
    print(f"   Models to evaluate: {len(ae_models_to_evaluate)}")
    print(f"   Output CSV: {OUTPUT_CSV}")

    existing_model_names = set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        existing_model_names = _load_existing_model_names(OUTPUT_CSV)
        if existing_model_names and not QUIET:
            print(f"   ✓ Found {len(existing_model_names)} existing entries in CSV; will skip them")
    else:
        init_results_csv(OUTPUT_CSV, binary_mode=BINARY_MODE)
    
    # Use reconstruction mode
    use_ae_reconstruction = USE_AE_RECONSTRUCTION
    
    # Store results for batch evaluation
    all_results = []

    # Rewrite previous run's global example label directory
    example_labels_root = Path(EXAMPLE_LABELS_DIR)
    if example_labels_root.exists():
        shutil.rmtree(example_labels_root)
    example_labels_root.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset once (will update autoencoder for each model)
    print(f"\n📂 Loading test dataset (will be reused for all models)...")
    if effective_recompute_scores:
        print(f"   Score recomputation: ENABLED (scores will be computed with each AE)")
    else:
        print(f"   ⚠️  Score recomputation: DISABLED (using pre-computed scores.scp - results will be identical!)")
    
    # Initialize with None autoencoder - we'll update it for each model
    test_data = VadSETAEDataset(
        DATA_TEST,
        EMBED_PATH,
        SCORE_TYPE,
        autoencoder=None,
        use_autoencoder=False,
        use_ae_reconstruction=False,
        transform_enrolled_in_reconstruction=TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION,
        transform_enrolled_vadinput_in_reconstruction=TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION,
        recompute_scores=effective_recompute_scores,
        audio_root=AUDIO_ROOT_OVERRIDE,
        collect_similarity_scores=PLOT_SIMILARITY_SCORES and effective_recompute_scores,
        similarity_score_sample_step=SIMILARITY_SCORE_SAMPLE_STEP,
        similarity_score_max_items=SIMILARITY_SCORE_MAX_UTTS,
    )
    
    total_samples = len(test_data)

    # Optionally force inference to one target speaker for the whole dataset.
    # If the forced target differs from an utterance's original target, relabel 2->1.
    _apply_forced_inference_target(
        test_data,
        FORCE_INFERENCE_TARGET_SPEAKER_ID,
        quiet=QUIET
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
    
    # Main evaluation loop
    for model_idx, ae_model_path_current in enumerate(ae_models_to_evaluate):
        if not QUIET:
            print("\n" + "=" * 80)
            print(f"📦 Evaluating model {model_idx + 1}/{len(ae_models_to_evaluate)}")
            print("=" * 80)
        
        # Determine if using autoencoder for this iteration
        if ae_model_path_current == 'NO_AE':
            use_autoencoder = False
            use_ae_reconstruction = False
        else:
            use_autoencoder = True
            use_ae_reconstruction = USE_AE_RECONSTRUCTION
        
        # Set current model path
        current_ae_path = ae_model_path_current if ae_model_path_current != 'NO_AE' else None
        
        model_name = Path(current_ae_path).name if current_ae_path else 'NO_AE'
        if model_name in existing_model_names:
            if not QUIET:
                print(f"   ⏭️  Skipping {model_name} (already in CSV)")
            continue

        # Load autoencoder and update dataset
        if use_autoencoder:
            if not QUIET:
                print(f"\n🔧 Loading autoencoder model...")
                print(f"   Model path: {current_ae_path}")
            autoencoder, ae_config = load_autoencoder(current_ae_path, device)
            autoencoder = autoencoder.to(device)  # Ensure autoencoder is on the correct device
            autoencoder.eval()
            model_key = os.path.basename(current_ae_path)
            override_activation = AE_ACTIVATION_OVERRIDE.get(model_key)
            if override_activation:
                _apply_activation_override(autoencoder, override_activation)
                if not QUIET:
                    print(f"   🔁 Activation override: {override_activation}")
            from torchsummary import summary
            summary(autoencoder, (256,))
            encoded_dim = _get_ae_bottleneck_dim(ae_config)
            ae_model_type = ae_config.get('model_type', 'dvector_autoencoder')

            if ae_model_type == 'greedy_layerwise_stacked' and not use_ae_reconstruction:
                use_ae_reconstruction = True
                if not QUIET:
                    print("   ⚠️  Greedy layer-wise AE has no bottleneck-only mode; forcing reconstruction")
            
            if not QUIET:
                print(f"\n✅ Autoencoder loaded successfully!")
                if ae_model_type == 'deep_stacked_dae':
                    first_block_hidden_dims = ae_config.get('first_block_hidden_dims', [])
                    if first_block_hidden_dims:
                        first_block_str = f"256→{first_block_hidden_dims}→256"
                    else:
                        first_block_str = "256→256"
                    print(
                        f"   Architecture: deep_stacked_dae ({ae_config.get('n_stacked_daes', 2)} DAEs)"
                    )
                    print(f"   First block: {first_block_str} (MLP)")
                    print(f"   Refinement blocks: {ae_config.get('stack_refinement_hidden_dims', [1024, 1024])}")
                    print(f"   Processed dim: {ae_config.get('input_dim', 256)}-dim (linear output)")
                elif ae_model_type == 'greedy_layerwise_stacked':
                    print("   Architecture: greedy_layerwise_stacked")
                    print(f"   Greedy hidden dims: {ae_config.get('greedy_hidden_dims', [])}")
                    print(f"   Processed dim: {ae_config.get('input_dim', 256)}-dim (linear output)")
                else:
                    print(f"   Architecture: {ae_config['hidden_dims']}")
                print(f"   Bottleneck/output dim: {encoded_dim}-dim")
                if use_ae_reconstruction:
                    print(f"   Mode: Full reconstruction (256→{encoded_dim}→256)")
                else:
                    print(f"   Mode: Bottleneck only (256→{encoded_dim})")
            
            if use_ae_reconstruction:
                input_dim = 297  # 40 fbanks + 256 reconstructed + 1 score
            else:
                input_dim = 40 + encoded_dim + 1
        else:
            if not QUIET:
                print(f"\n⚠️  Using full 256-dim d-vectors (no compression)")
            autoencoder = None
            use_ae_reconstruction = False
            encoded_dim = 256
            ae_config = {}
            input_dim = 297  # 40 fbanks + 256 full + 1 score
        
        # Update the dataset's autoencoder and reprocess embeddings
        actual_dataset = test_data.dataset if isinstance(test_data, Subset) else test_data
        
        # Ensure autoencoder is on correct device before assignment
        if autoencoder is not None:
            autoencoder = autoencoder.to(device)
        
        actual_dataset.autoencoder = autoencoder
        actual_dataset.use_autoencoder = use_autoencoder
        actual_dataset.use_ae_reconstruction = use_ae_reconstruction
        actual_dataset.transform_enrolled_in_reconstruction = (
            TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION and use_ae_reconstruction
        )
        actual_dataset.transform_enrolled_vadinput_in_reconstruction = (
            TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION and use_ae_reconstruction
        )
        actual_dataset.collect_similarity_scores = (
            PLOT_SIMILARITY_SCORES and effective_recompute_scores
        )
        actual_dataset.similarity_score_sample_step = SIMILARITY_SCORE_SAMPLE_STEP
        actual_dataset.similarity_score_max_items = SIMILARITY_SCORE_MAX_UTTS
        actual_dataset.similarity_scores = {}
        actual_dataset.similarity_scores_by_class = {0: [], 1: [], 2: []}
        actual_dataset.similarity_score_num_utts = 0
        
        # Reprocess enrolled d-vectors with the new autoencoder
        # Note: When recompute_scores=True, stream d-vectors are processed in __getitem__
        if use_autoencoder and autoencoder is not None:
            if not QUIET:
                print(f"♻️  Reprocessing enrolled d-vectors with new autoencoder...")
            if use_ae_reconstruction:
                need_reconstructed_enrolled = (
                    TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION or
                    TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION
                )
                if need_reconstructed_enrolled:
                    # Full reconstruction with transformed enrolled anchors.
                    actual_dataset.processed_embed = {}
                    autoencoder.eval()
                    with torch.no_grad():
                        for target, dvector in actual_dataset.embed.items():
                            dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device)
                            reconstructed = autoencoder(dvector_tensor)
                            actual_dataset.processed_embed[target] = reconstructed.cpu().numpy().squeeze()
                    if not QUIET:
                        print(f"   ✓ Reconstructed {len(actual_dataset.processed_embed)} enrolled d-vectors (256-dim)")
                        if TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION:
                            print(f"   ✓ Using transformed enrolled vectors for scoring anchor")
                        if TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION:
                            print(f"   ✓ Using transformed enrolled vectors for VAD input")
                        if effective_recompute_scores:
                            print(f"   ✓ Stream d-vectors will be reconstructed on-the-fly for scoring")
                else:
                    # Full reconstruction: keep enrolled original; only stream is reconstructed in __getitem__.
                    actual_dataset.processed_embed = None
                    if not QUIET:
                        print(f"   ✓ Keeping enrolled d-vectors original (256-dim)")
                        if effective_recompute_scores:
                            print(f"   ✓ Stream d-vectors will be reconstructed on-the-fly for scoring")
            else:
                # Bottleneck mode: compress enrolled d-vectors to AE bottleneck dim
                actual_dataset.processed_embed = {}
                autoencoder.eval()
                with torch.no_grad():
                    for target, dvector in actual_dataset.embed.items():
                        dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device)
                        compressed = autoencoder.encode(dvector_tensor)
                        actual_dataset.processed_embed[target] = compressed.cpu().numpy().squeeze()
                if not QUIET:
                    print(f"   ✓ Compressed {len(actual_dataset.processed_embed)} enrolled d-vectors ({encoded_dim}-dim)")
                    if effective_recompute_scores:
                        print(f"   ✓ Stream d-vectors will be compressed on-the-fly for scoring")
        else:
            actual_dataset.processed_embed = None
            if not QUIET:
                print(f"♻️  Using full 256-dim d-vectors (no compression)")
        
        # Create DataLoader (reusing filtered dataset)
        test_loader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=pad_collate_with_metadata
        )
        
        # Load VAD model (must be done for each iteration since input_dim may vary)
        if not QUIET:
            print(f"\n🏗️  Loading VAD model...")
            print(f"   Model path: {VAD_MODEL_PATH}")
            print(f"   Expected input dim: {input_dim}")
        
        hidden_dim = 64
        num_layers = 2
        out_dim = 3
        
        vad_model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=True, linear=False)
        
        # Load weights
        if device == torch.device('cuda') and torch.cuda.is_available():
            checkpoint = torch.load(VAD_MODEL_PATH)
        else:
            checkpoint = torch.load(VAD_MODEL_PATH, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Verify input dimension matches
            if 'lstm.weight_ih_l0' in state_dict:
                trained_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
                if trained_input_dim != input_dim:
                    print(f"\n❌ ERROR: Input dimension mismatch!")
                    print(f"   Model trained with: {trained_input_dim}-dim")
                    print(f"   Current config expects: {input_dim}-dim")
                    print(f"   VAD model: {VAD_MODEL_PATH}")
                    print(f"   AE model: {current_ae_path}")
                    print(f"\n💡 Make sure the VAD model was trained with the same AE configuration!")
                    sys.exit(1)
            
            vad_model.load_state_dict(state_dict)
        else:
            vad_model.load_state_dict(checkpoint)
        
        vad_model = vad_model.to(device)
        
        # Count and print model parameters
        vad_params = sum(p.numel() for p in vad_model.parameters())
        ae_params = 0
        if use_autoencoder and autoencoder is not None:
            ae_params = sum(p.numel() for p in autoencoder.parameters())
        
        if not QUIET:
            print(f"\n📊 Model Parameters:")
            if use_autoencoder and autoencoder is not None:
                print(f"   AE: {ae_params:,} parameters")
            else:
                print(f"   AE: Not used (0 parameters)")
            print(f"   VAD: {vad_params:,} parameters")
            print(f"   Total: {ae_params + vad_params:,} parameters")
        
        # Evaluate
        if not QUIET:
            print("\n" + "=" * 80)
            print("EVALUATING")
            print("=" * 80)
        
        # Single-pass evaluation: metrics, confusion matrix, and examples are returned together.
        acc, mAP, out_AP, avg_precision, avg_recall, avg_f1, eval_details = evaluate_vad_with_ae(
            vad_model,
            test_loader,
            device,
            show_examples=False,
            n_examples=N_EXAMPLE_UTTERANCES,
            binary_mode=BINARY_MODE,
            show_class_stats=False,
            return_details=True,
        )

        per_class_precision = eval_details['per_class_precision']
        per_class_recall = eval_details['per_class_recall']
        per_class_f1 = eval_details['per_class_f1']
        conf_cm_pct = eval_details['confusion_matrix_pct_total']
        example_sequences = eval_details['example_sequences']

        # Export per-model example label files (3 utterances x 3 files)
        model_name = Path(current_ae_path).name if current_ae_path else 'NO_AE'
        if PLOT_SIMILARITY_SCORES and effective_recompute_scores:
            plot_path = _plot_similarity_scores(
                model_name,
                getattr(actual_dataset, 'similarity_scores_by_class', {}),
                SIMILARITY_PLOTS_DIR,
            )
            if plot_path and not QUIET:
                print(f"📈 Similarity plot: {plot_path}")
        examples_out_dir = _export_example_label_files(
            example_labels_root,
            model_name,
            example_sequences,
            frame_step_sec=LABEL_FRAME_STEP_SEC
        )
        if not QUIET:
            print(f"\n📝 Saved {len(example_sequences)} utterance example label sets to: {examples_out_dir}")
        
        # Prepare row for CSV
        csv_row = {
            'ae_model_name': model_name,
            'ae_model_path': current_ae_path if current_ae_path else 'N/A',
            'mAP': f"{mAP * 100:.2f}",
            'accuracy': f"{acc:.2f}",
            'precision_micro': f"{avg_precision * 100:.2f}",
            'recall_micro': f"{avg_recall * 100:.2f}",
            'f1_micro': f"{avg_f1 * 100:.2f}",
        }
        
        # Add AP values based on number of classes
        if BINARY_MODE:
            csv_row['AP_NonTarget'] = f"{out_AP[0] * 100:.2f}"
            csv_row['AP_Target'] = f"{out_AP[1] * 100:.2f}"
            csv_row['precision_NonTarget'] = f"{per_class_precision[0] * 100:.2f}"
            csv_row['recall_NonTarget'] = f"{per_class_recall[0] * 100:.2f}"
            csv_row['f1_NonTarget'] = f"{per_class_f1[0] * 100:.2f}"
            csv_row['precision_Target'] = f"{per_class_precision[1] * 100:.2f}"
            csv_row['recall_Target'] = f"{per_class_recall[1] * 100:.2f}"
            csv_row['f1_Target'] = f"{per_class_f1[1] * 100:.2f}"
            # Confusion matrix for binary (2x2)
            csv_row['conf_NT_as_NT'] = f"{conf_cm_pct[0, 0]:.4f}"
            csv_row['conf_NT_as_T'] = f"{conf_cm_pct[0, 1]:.4f}"
            csv_row['conf_T_as_NT'] = f"{conf_cm_pct[1, 0]:.4f}"
            csv_row['conf_T_as_T'] = f"{conf_cm_pct[1, 1]:.4f}"
        else:
            csv_row['AP_NS'] = f"{out_AP[0] * 100:.2f}"
            csv_row['AP_NTSS'] = f"{out_AP[1] * 100:.2f}"
            csv_row['AP_TSS'] = f"{out_AP[2] * 100:.2f}"
            csv_row['precision_NS'] = f"{per_class_precision[0] * 100:.2f}"
            csv_row['recall_NS'] = f"{per_class_recall[0] * 100:.2f}"
            csv_row['f1_NS'] = f"{per_class_f1[0] * 100:.2f}"
            csv_row['precision_NTSS'] = f"{per_class_precision[1] * 100:.2f}"
            csv_row['recall_NTSS'] = f"{per_class_recall[1] * 100:.2f}"
            csv_row['f1_NTSS'] = f"{per_class_f1[1] * 100:.2f}"
            csv_row['precision_TSS'] = f"{per_class_precision[2] * 100:.2f}"
            csv_row['recall_TSS'] = f"{per_class_recall[2] * 100:.2f}"
            csv_row['f1_TSS'] = f"{per_class_f1[2] * 100:.2f}"
            # Confusion matrix for 3-class (3x3)
            csv_row['conf_NSasNS'] = f"{conf_cm_pct[0, 0]:.4f}"
            csv_row['conf_NSasNTSS'] = f"{conf_cm_pct[0, 1]:.4f}"
            csv_row['conf_NSasTSS'] = f"{conf_cm_pct[0, 2]:.4f}"
            csv_row['conf_NTSSasNS'] = f"{conf_cm_pct[1, 0]:.4f}"
            csv_row['conf_NTSSasNTSS'] = f"{conf_cm_pct[1, 1]:.4f}"
            csv_row['conf_NTSSasTSS'] = f"{conf_cm_pct[1, 2]:.4f}"
            csv_row['conf_TSSasNS'] = f"{conf_cm_pct[2, 0]:.4f}"
            csv_row['conf_TSSasNTSS'] = f"{conf_cm_pct[2, 1]:.4f}"
            csv_row['conf_TSSasTSS'] = f"{conf_cm_pct[2, 2]:.4f}"
        
        append_result_to_csv(csv_row, OUTPUT_CSV, binary_mode=BINARY_MODE)
        existing_model_names.add(model_name)
        all_results.append(csv_row)
        
        # Print summary for this model
        print(f"\n{'='*80}")
        print(f"📊 RESULTS: {model_name}")
        print(f"{'='*80}")
        if BINARY_MODE:
            print(
                f"Accuracy: {acc:.2f}% | mAP: {mAP*100:.2f}% | "
                f"P/R/F1 (micro): [{avg_precision*100:.2f}%, {avg_recall*100:.2f}%, {avg_f1*100:.2f}%] | "
                f"AP [Non-Target/Target]: [{out_AP[0]*100:.2f}%, {out_AP[1]*100:.2f}%]"
            )
            print(
                f"Per-class P/R/F1 [Non-Target]: [{per_class_precision[0]*100:.2f}%, {per_class_recall[0]*100:.2f}%, {per_class_f1[0]*100:.2f}%] | "
                f"[Target]: [{per_class_precision[1]*100:.2f}%, {per_class_recall[1]*100:.2f}%, {per_class_f1[1]*100:.2f}%]"
            )
            if SHOW_CONFUSION_MATRIX_PERCENT:
                _print_confusion_matrix_pct(conf_cm_pct, binary_mode=True)
        else:
            print(
                f"Accuracy: {acc:.2f}% | mAP: {mAP*100:.2f}% | "
                f"P/R/F1 (micro): [{avg_precision*100:.2f}%, {avg_recall*100:.2f}%, {avg_f1*100:.2f}%] | "
                f"AP [NS/NTSS/TSS]: [{out_AP[0]*100:.2f}%, {out_AP[1]*100:.2f}%, {out_AP[2]*100:.2f}%]"
            )
            print(
                f"Per-class P/R/F1 [NS]: [{per_class_precision[0]*100:.2f}%, {per_class_recall[0]*100:.2f}%, {per_class_f1[0]*100:.2f}%] | "
                f"[NTSS]: [{per_class_precision[1]*100:.2f}%, {per_class_recall[1]*100:.2f}%, {per_class_f1[1]*100:.2f}%] | "
                f"[TSS]: [{per_class_precision[2]*100:.2f}%, {per_class_recall[2]*100:.2f}%, {per_class_f1[2]*100:.2f}%]"
            )
            if SHOW_CONFUSION_MATRIX_PERCENT:
                _print_confusion_matrix_pct(conf_cm_pct, binary_mode=False)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 BATCH EVALUATION SUMMARY")
    if BINARY_MODE:
        print("Mode: BINARY CLASSIFICATION (Target vs Non-Target)")
    else:
        print("Mode: 3-CLASS CLASSIFICATION (NS, NTSS, TSS)")
    print("=" * 80)
    
    if BINARY_MODE:
        print(f"\n{'Model':<40} {'Acc %':<10} {'mAP':<10} {'P':<8} {'R':<8} {'F1':<8} {'AP_NT':<10} {'AP_T':<10}")
        print("-" * 100)
        for row in all_results:
            model_name = row['ae_model_name'][:38]
            print(
                f"{model_name:<40} {row['accuracy']:<10} {row['mAP']:<10} "
                f"{row['precision_micro']:<8} {row['recall_micro']:<8} {row['f1_micro']:<8} "
                f"{row['AP_NonTarget']:<10} {row['AP_Target']:<10}"
            )
    else:
        print(f"\n{'Model':<40} {'Acc %':<10} {'mAP':<10} {'P':<8} {'R':<8} {'F1':<8} {'AP_NS':<10} {'AP_NTSS':<10} {'AP_TSS':<10} {'F1_TSS':<10}")
        print("-" * 132)
        for row in all_results:
            model_name = row['ae_model_name'][:38]
            print(
                f"{model_name:<40} {row['accuracy']:<10} {row['mAP']:<10} "
                f"{row['precision_micro']:<8} {row['recall_micro']:<8} {row['f1_micro']:<8} "
                f"{row['AP_NS']:<10} {row['AP_NTSS']:<10} {row['AP_TSS']:<10} {row['f1_TSS']:<10}"
            )
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
