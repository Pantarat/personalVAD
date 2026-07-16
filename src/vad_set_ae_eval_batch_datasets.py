"""@package vad_set_ae_eval_batch_datasets

Batch evaluation script for one or more SET-AE models across multiple datasets.

Usage:
    # Edit AE_MODEL_PATH, DATASET_LIST, and other settings in the file, then run:
    python vad_set_ae_eval_batch_datasets.py
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
import matplotlib.pyplot as plt

# Import from vad_set_ae_eval to reuse evaluation logic and defaults
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
    AE_MODEL_PATH as DEFAULT_AE_MODEL_PATH,
)

# Import from vad_set_ae
try:
    from AE_test.autoencoder_utils import DvectorAutoencoder, load_autoencoder
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AE_test'))
    from autoencoder_utils import DvectorAutoencoder, load_autoencoder
from vad_set_ae import VadSETAEDataset, pad_collate_with_metadata
from personal_vad import PersonalVAD

# Fixed AE model(s) to evaluate on all datasets.
# Accepts either a single path string or a list/tuple of path strings.
AE_MODEL_PATH = [
    # 'src/AE_test/test_outputs/models/greedy_finetune_fromScratch/6829_0,7,3_6-6-26',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_8pct_6utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_12pct_9utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_16pct_12utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_20pct_15utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_40pct_30utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_60pct_45utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_80pct_60utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
    'src/AE_test/test_outputs/models/greedy_finetunev5/direct_finetune/908/dvector_ae-908_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep',
]

# List of datasets to evaluate
DATASET_LIST = [
    'data/908_ov_test_noOther_main908_500_3-6-2026',
    'data/908_ov_test_noOther_main908_babble_500_3-6-2026',
    'data/908_ov_test_ov0pct_main908_500_3-6-2026',
    'data/908_ov_test_ov0pct_main908_babble_500_3-6-2026',
    'data/908_ov_test_ov100pct_main908_500_3-6-2026',
    'data/908_ov_test_ov100pct_main908_babble_500_30-3-2026',
]

OUTPUT_CSV = 'model_evaluation_results/greedy_finetunev5_tt/direct_finetune/908/test.csv'

# Score recomputation settings (imported from vad_set_ae_eval.py)
# WARNING: Requires wav.scp in each test directory or valid AUDIO_ROOT override
AUDIO_ROOT_OVERRIDE = None  # Set to path; None = auto-detect from dataset_dir/wav.scp

# DataLoader settings
# IMPORTANT: When using GPU with score recomputation, set NUM_WORKERS=0
NUM_WORKERS = 0  # Set to 0 for GPU compatibility with score recomputation

# Label example export settings
N_EXAMPLE_UTTERANCES = 3  # Number of utterances per dataset to export
LABEL_FRAME_STEP_SEC = 0.01  # Label frame duration in seconds (txt format: start\tend\tlabel)
EXAMPLE_LABELS_DIR = 'model_evaluation_results/example_labels'
SHOW_CONFUSION_MATRIX_PERCENT = True

# Similarity score plotting
PLOT_SIMILARITY_SCORES = False
SIMILARITY_PLOTS_DIR = 'model_evaluation_results/similarity_scores'
SIMILARITY_SCORE_SAMPLE_STEP = 5
SIMILARITY_SCORE_MAX_UTTS = 50

# Activation override (tanh/relu) per AE model basename. Leave empty to use config.
AE_ACTIVATION_OVERRIDE = {
    # 'dvector_ae_identity_1100x50Dev_wOV_balanced_1,6s_v2_12-4-26': 'relu',
}

INCLUDE_NO_AE = True


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


def _normalize_ae_model_paths(ae_model_path_config):
    """Allow AE model config to be provided as a single path or a list of paths."""
    if ae_model_path_config is None:
        return []

    if isinstance(ae_model_path_config, (str, os.PathLike)):
        path_text = str(ae_model_path_config).strip()
        return [path_text] if path_text else []

    paths = []
    for raw_path in ae_model_path_config:
        path_text = str(raw_path).strip()
        if path_text:
            paths.append(path_text)
    return paths


def _build_output_csv_path(base_output_csv, model_name, multi_mode):
    base_path = Path(base_output_csv)
    if not multi_mode:
        return base_path

    safe_model_name = _sanitize_filename(model_name) or 'model'
    return base_path.with_name(f"{base_path.stem}_{safe_model_name}{base_path.suffix}")


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

    # Rewrite previous run's examples for this dataset/model
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


def _plot_similarity_scores(plot_name, score_dict_by_class, output_dir):
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
    safe_name = _sanitize_filename(plot_name)

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

    fig.suptitle(f'SV Similarity Scores: {plot_name}', fontsize=12, fontweight='bold')
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
        print(f"\nForced inference target speaker enabled: {forced_target}")
        print(f"   Utterances with overridden target: {changed_target_utts}/{len(dataset.keys)}")
        print(f"   Utterances containing class-2 labels: {relabeled_utts}")
        print(f"   Frames relabeled (2->1): {relabeled_frames_demoted_2_to_1}")
        print("   Forced mode output labels contain no class-2 target frames")


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
        'dataset_name',
        'dataset_path',
        'n_samples',
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


def _build_result_key(dataset_path, ae_model_name):
    return f"{dataset_path}::{ae_model_name}"


def _load_existing_result_keys(output_path):
    """Return set of dataset+model keys already present in the CSV."""
    existing = set()
    if not os.path.exists(output_path):
        return existing

    try:
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                return existing
            if 'dataset_path' not in reader.fieldnames or 'ae_model_name' not in reader.fieldnames:
                return existing
            for row in reader:
                dataset_path = row.get('dataset_path')
                model_name = row.get('ae_model_name')
                if dataset_path and model_name:
                    existing.add(_build_result_key(dataset_path, model_name))
    except OSError:
        return existing

    return existing


def init_results_csv(output_path, binary_mode=False):
    """Create/overwrite CSV and write header only."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _get_csv_fieldnames(binary_mode)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
    print(f"\nResults CSV initialized: {output_path}")


def append_result_to_csv(result, output_path, binary_mode=False):
    """Append a single result row to the CSV."""
    if not result:
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _get_csv_fieldnames(binary_mode)
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writerow(result)


def _build_eval_modes(effective_recompute_scores, quiet=False):
    eval_modes = []
    ae_model_paths = _normalize_ae_model_paths(AE_MODEL_PATH)

    for ae_model_path in ae_model_paths:
        if not quiet:
            print("\nLoading autoencoder model...")
            print(f"   Model path: {ae_model_path}")
        autoencoder, ae_config = load_autoencoder(ae_model_path, device)
        autoencoder = autoencoder.to(device)
        autoencoder.eval()

        model_key = os.path.basename(ae_model_path)
        override_activation = AE_ACTIVATION_OVERRIDE.get(model_key)
        if override_activation:
            _apply_activation_override(autoencoder, override_activation)
            if not quiet:
                print(f"   Activation override: {override_activation}")

        encoded_dim = _get_ae_bottleneck_dim(ae_config)
        ae_model_type = ae_config.get('model_type', 'dvector_autoencoder')
        use_ae_reconstruction = USE_AE_RECONSTRUCTION

        if ae_model_type == 'greedy_layerwise_stacked' and not use_ae_reconstruction:
            use_ae_reconstruction = True
            if not quiet:
                print("   Greedy layer-wise AE has no bottleneck-only mode; forcing reconstruction")

        if use_ae_reconstruction:
            ae_input_dim = 297  # 40 fbanks + 256 reconstructed + 1 score
        else:
            ae_input_dim = 40 + encoded_dim + 1

        eval_modes.append({
            'label': Path(ae_model_path).name,
            'ae_model_path': ae_model_path,
            'use_autoencoder': True,
            'autoencoder': autoencoder,
            'use_ae_reconstruction': use_ae_reconstruction,
            'input_dim': ae_input_dim,
            'encoded_dim': encoded_dim,
            'collect_similarity_scores': PLOT_SIMILARITY_SCORES and effective_recompute_scores,
        })

    if INCLUDE_NO_AE:
        eval_modes.append({
            'label': 'NO_AE',
            'ae_model_path': 'NO_AE',
            'use_autoencoder': False,
            'autoencoder': None,
            'use_ae_reconstruction': False,
            'input_dim': 297,  # 40 fbanks + 256 full + 1 score
            'encoded_dim': 256,
            'collect_similarity_scores': False,
        })

    return eval_modes


if __name__ == '__main__':
    # Configuration - Edit these values directly in the file
    QUIET = False
    FILTER_MAIN_SPEAKER = None  # Set to speaker ID like '84' to filter, or None for all speakers
    BINARY_MODE = BINARY_CLASSIFICATION  # True: 2-class (target vs non-target), False: 3-class
    FORCE_INFERENCE_TARGET_SPEAKER_ID = None  # e.g. '84' to force inference target across all utterances
    FORCE_RECOMPUTE_SCORES_WITH_FORCED_TARGET = False  # Recommended: recompute scores when forcing target speaker

    ae_model_paths = _normalize_ae_model_paths(AE_MODEL_PATH)
    if not ae_model_paths and not INCLUDE_NO_AE:
        print("ERROR: AE_MODEL_PATH is empty. Set AE_MODEL_PATH before running.")
        sys.exit(1)

    force_target_enabled = (
        FORCE_INFERENCE_TARGET_SPEAKER_ID is not None and str(FORCE_INFERENCE_TARGET_SPEAKER_ID).strip() != ''
    )
    effective_recompute_scores = RECOMPUTE_SCORES or (
        FORCE_RECOMPUTE_SCORES_WITH_FORCED_TARGET and force_target_enabled
    )

    dataset_paths = list(DATASET_LIST)
    if not dataset_paths:
        print("ERROR: DATASET_LIST is empty.")
        sys.exit(1)

    print("=" * 80)
    print("VAD SET-AE BATCH EVALUATION (MULTI-DATASET)")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"   VAD Model: {VAD_MODEL_PATH}")
    if ae_model_paths:
        print(f"   AE Models ({len(ae_model_paths)}):")
        for ae_model_path in ae_model_paths:
            print(f"      - {ae_model_path}")
    else:
        print("   AE Models: None")
    print(f"   Include NO_AE: {INCLUDE_NO_AE}")
    print(f"   Datasets: {len(dataset_paths)}")
    print(f"   Score Type: {SCORE_TYPE}")
    print(f"   Reconstruction enrolled transform (scoring): {TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION}")
    print(f"   Reconstruction enrolled transform (VAD input): {TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION}")
    print(f"   Forced Inference Target: {FORCE_INFERENCE_TARGET_SPEAKER_ID}")
    if force_target_enabled and not RECOMPUTE_SCORES and FORCE_RECOMPUTE_SCORES_WITH_FORCED_TARGET:
        print("   WARNING: Forcing recompute_scores=True because forced target is enabled")
    if FILTER_MAIN_SPEAKER:
        print(f"   Main Speaker: {FILTER_MAIN_SPEAKER} (filtering enabled)")

    # Rewrite previous run's global example label directory
    example_labels_root = Path(EXAMPLE_LABELS_DIR)
    if example_labels_root.exists():
        shutil.rmtree(example_labels_root)
    example_labels_root.mkdir(parents=True, exist_ok=True)

    eval_modes = _build_eval_modes(effective_recompute_scores, quiet=QUIET)
    multi_mode_output = len(eval_modes) > 1

    print("\nOutput CSV(s):")
    for eval_mode in eval_modes:
        output_csv_path = _build_output_csv_path(OUTPUT_CSV, eval_mode['label'], multi_mode_output)
        eval_mode['output_csv'] = str(output_csv_path)

        existing_result_keys = set()
        if output_csv_path.exists() and output_csv_path.stat().st_size > 0:
            existing_result_keys = _load_existing_result_keys(output_csv_path)
            if existing_result_keys and not QUIET:
                print(
                    f"   {eval_mode['label']}: {output_csv_path} "
                    f"({len(existing_result_keys)} existing entries will be skipped)"
                )
            else:
                print(f"   {eval_mode['label']}: {output_csv_path}")
        else:
            init_results_csv(output_csv_path, binary_mode=BINARY_MODE)

        eval_mode['existing_result_keys'] = existing_result_keys

    # Store results for summary
    all_results = []

    for dataset_idx, dataset_path in enumerate(dataset_paths):
        dataset_name = Path(dataset_path).name

        print("\n" + "=" * 80)
        print(f"Evaluating dataset {dataset_idx + 1}/{len(dataset_paths)}")
        print("=" * 80)
        print(f"   Dataset: {dataset_path}")

        for mode_idx, eval_mode in enumerate(eval_modes):
            model_name = eval_mode['label']
            result_key = _build_result_key(dataset_path, model_name)
            if result_key in eval_mode['existing_result_keys']:
                if not QUIET:
                    print(f"Skipping {dataset_name} / {model_name} (already in {eval_mode['output_csv']})")
                continue

            if mode_idx > 0:
                print("-" * 80)

            print(f"   Eval mode: {model_name}")
            if effective_recompute_scores:
                if eval_mode['use_autoencoder']:
                    print("   Score recomputation: ENABLED (scores will be computed with AE)")
                else:
                    print("   Score recomputation: ENABLED (scores will be computed without AE)")
            else:
                print("   Score recomputation: DISABLED (using pre-computed scores.scp)")

            if AUDIO_ROOT_OVERRIDE is not None:
                audio_root = AUDIO_ROOT_OVERRIDE
            else:
                audio_root = dataset_path if effective_recompute_scores else None

            test_data = VadSETAEDataset(
                dataset_path,
                EMBED_PATH,
                SCORE_TYPE,
                autoencoder=eval_mode['autoencoder'],
                use_autoencoder=eval_mode['use_autoencoder'],
                use_ae_reconstruction=eval_mode['use_ae_reconstruction'],
                transform_enrolled_in_reconstruction=(
                    TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION and eval_mode['use_ae_reconstruction']
                ),
                transform_enrolled_vadinput_in_reconstruction=(
                    TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION and eval_mode['use_ae_reconstruction']
                ),
                recompute_scores=effective_recompute_scores,
                audio_root=audio_root,
                collect_similarity_scores=eval_mode['collect_similarity_scores'],
                similarity_score_sample_step=SIMILARITY_SCORE_SAMPLE_STEP,
                similarity_score_max_items=SIMILARITY_SCORE_MAX_UTTS,
            )

            _apply_forced_inference_target(
                test_data,
                FORCE_INFERENCE_TARGET_SPEAKER_ID,
                quiet=QUIET,
            )

            if FILTER_MAIN_SPEAKER:
                print(f"Filtering dataset for speaker: {FILTER_MAIN_SPEAKER}")
                original_indices = []
                for idx in range(len(test_data)):
                    key = test_data.keys[idx]
                    speaker_id = key.split('-')[0]
                    if speaker_id == FILTER_MAIN_SPEAKER:
                        original_indices.append(idx)

                if len(original_indices) == 0:
                    print(f"ERROR: No samples found for speaker '{FILTER_MAIN_SPEAKER}' in test set")
                    sys.exit(1)

                test_data = Subset(test_data, original_indices)
                print(f"   Found {len(test_data)} samples for speaker {FILTER_MAIN_SPEAKER}")

            print(f"Total samples: {len(test_data)}")

            actual_dataset = test_data.dataset if isinstance(test_data, Subset) else test_data
            actual_dataset.autoencoder = eval_mode['autoencoder']
            actual_dataset.use_autoencoder = eval_mode['use_autoencoder']
            actual_dataset.use_ae_reconstruction = eval_mode['use_ae_reconstruction']
            actual_dataset.transform_enrolled_in_reconstruction = (
                TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION and eval_mode['use_ae_reconstruction']
            )
            actual_dataset.transform_enrolled_vadinput_in_reconstruction = (
                TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION and eval_mode['use_ae_reconstruction']
            )
            actual_dataset.collect_similarity_scores = eval_mode['collect_similarity_scores']
            actual_dataset.similarity_score_sample_step = SIMILARITY_SCORE_SAMPLE_STEP
            actual_dataset.similarity_score_max_items = SIMILARITY_SCORE_MAX_UTTS
            actual_dataset.similarity_scores = {}
            actual_dataset.similarity_scores_by_class = {0: [], 1: [], 2: []}
            actual_dataset.similarity_score_num_utts = 0

            if eval_mode['use_autoencoder']:
                if not QUIET:
                    print("Reprocessing enrolled d-vectors with autoencoder...")
                if eval_mode['use_ae_reconstruction']:
                    need_reconstructed_enrolled = (
                        TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION or
                        TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION
                    )
                    if need_reconstructed_enrolled:
                        actual_dataset.processed_embed = {}
                        eval_mode['autoencoder'].eval()
                        with torch.no_grad():
                            for target, dvector in actual_dataset.embed.items():
                                dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device)
                                reconstructed = eval_mode['autoencoder'](dvector_tensor)
                                actual_dataset.processed_embed[target] = reconstructed.cpu().numpy().squeeze()
                        if not QUIET:
                            print(
                                f"   Reconstructed {len(actual_dataset.processed_embed)} enrolled d-vectors (256-dim)"
                            )
                    else:
                        actual_dataset.processed_embed = None
                        if not QUIET:
                            print("   Keeping enrolled d-vectors original (256-dim)")
                else:
                    actual_dataset.processed_embed = {}
                    eval_mode['autoencoder'].eval()
                    with torch.no_grad():
                        for target, dvector in actual_dataset.embed.items():
                            dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device)
                            compressed = eval_mode['autoencoder'].encode(dvector_tensor)
                            actual_dataset.processed_embed[target] = compressed.cpu().numpy().squeeze()
                    if not QUIET:
                        print(
                            f"   Compressed {len(actual_dataset.processed_embed)} enrolled d-vectors "
                            f"({eval_mode['encoded_dim']}-dim)"
                        )
            else:
                actual_dataset.processed_embed = None
                if not QUIET:
                    print("Using original enrolled d-vectors (NO_AE)")

            test_loader = DataLoader(
                test_data,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                collate_fn=pad_collate_with_metadata,
            )

            if not QUIET:
                print("\nLoading VAD model...")
                print(f"   Model path: {VAD_MODEL_PATH}")
                print(f"   Expected input dim: {eval_mode['input_dim']}")

            hidden_dim = 64
            num_layers = 2
            out_dim = 3

            vad_model = PersonalVAD(eval_mode['input_dim'], hidden_dim, num_layers, out_dim, use_fc=True, linear=False)

            if device == torch.device('cuda') and torch.cuda.is_available():
                checkpoint = torch.load(VAD_MODEL_PATH)
            else:
                checkpoint = torch.load(VAD_MODEL_PATH, map_location='cpu')

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'lstm.weight_ih_l0' in state_dict:
                    trained_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
                    if trained_input_dim != eval_mode['input_dim']:
                        print("ERROR: Input dimension mismatch!")
                        print(f"   Model trained with: {trained_input_dim}-dim")
                        print(f"   Current config expects: {eval_mode['input_dim']}-dim")
                        print(f"   VAD model: {VAD_MODEL_PATH}")
                        print(f"   Eval mode: {model_name}")
                        print(f"   AE model: {eval_mode['ae_model_path']}")
                        sys.exit(1)
                vad_model.load_state_dict(state_dict)
            else:
                vad_model.load_state_dict(checkpoint)

            vad_model = vad_model.to(device)

            if not QUIET:
                print("\nEVALUATING")

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

            if eval_mode['collect_similarity_scores']:
                plot_name = f"{model_name}_{dataset_name}"
                plot_path = _plot_similarity_scores(
                    plot_name,
                    getattr(actual_dataset, 'similarity_scores_by_class', {}),
                    SIMILARITY_PLOTS_DIR,
                )
                if plot_path and not QUIET:
                    print(f"Similarity plot: {plot_path}")

            dataset_label_dir = example_labels_root / _sanitize_filename(dataset_name)
            examples_out_dir = _export_example_label_files(
                dataset_label_dir,
                model_name,
                example_sequences,
                frame_step_sec=LABEL_FRAME_STEP_SEC,
            )
            if not QUIET:
                print(f"Saved {len(example_sequences)} example label sets to: {examples_out_dir}")

            csv_row = {
                'dataset_name': dataset_name,
                'dataset_path': dataset_path,
                'n_samples': f"{len(test_data)}",
                'ae_model_name': model_name,
                'ae_model_path': eval_mode['ae_model_path'],
                'mAP': f"{mAP * 100:.2f}",
                'accuracy': f"{acc:.2f}",
                'precision_micro': f"{avg_precision * 100:.2f}",
                'recall_micro': f"{avg_recall * 100:.2f}",
                'f1_micro': f"{avg_f1 * 100:.2f}",
            }

            if BINARY_MODE:
                csv_row['AP_NonTarget'] = f"{out_AP[0] * 100:.2f}"
                csv_row['AP_Target'] = f"{out_AP[1] * 100:.2f}"
                csv_row['precision_NonTarget'] = f"{per_class_precision[0] * 100:.2f}"
                csv_row['recall_NonTarget'] = f"{per_class_recall[0] * 100:.2f}"
                csv_row['f1_NonTarget'] = f"{per_class_f1[0] * 100:.2f}"
                csv_row['precision_Target'] = f"{per_class_precision[1] * 100:.2f}"
                csv_row['recall_Target'] = f"{per_class_recall[1] * 100:.2f}"
                csv_row['f1_Target'] = f"{per_class_f1[1] * 100:.2f}"
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
                csv_row['conf_NSasNS'] = f"{conf_cm_pct[0, 0]:.4f}"
                csv_row['conf_NSasNTSS'] = f"{conf_cm_pct[0, 1]:.4f}"
                csv_row['conf_NSasTSS'] = f"{conf_cm_pct[0, 2]:.4f}"
                csv_row['conf_NTSSasNS'] = f"{conf_cm_pct[1, 0]:.4f}"
                csv_row['conf_NTSSasNTSS'] = f"{conf_cm_pct[1, 1]:.4f}"
                csv_row['conf_NTSSasTSS'] = f"{conf_cm_pct[1, 2]:.4f}"
                csv_row['conf_TSSasNS'] = f"{conf_cm_pct[2, 0]:.4f}"
                csv_row['conf_TSSasNTSS'] = f"{conf_cm_pct[2, 1]:.4f}"
                csv_row['conf_TSSasTSS'] = f"{conf_cm_pct[2, 2]:.4f}"

            append_result_to_csv(csv_row, eval_mode['output_csv'], binary_mode=BINARY_MODE)
            eval_mode['existing_result_keys'].add(result_key)
            all_results.append(csv_row)

            print("\n" + "=" * 80)
            print(f"RESULTS: {dataset_name} [{model_name}]")
            print("=" * 80)
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

    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    if BINARY_MODE:
        print("Mode: BINARY CLASSIFICATION (Target vs Non-Target)")
    else:
        print("Mode: 3-CLASS CLASSIFICATION (NS, NTSS, TSS)")
    print("=" * 80)

    if BINARY_MODE:
        print(
            f"\n{'Dataset':<34} {'Mode':<20} {'Acc %':<10} {'mAP':<10} "
            f"{'P':<8} {'R':<8} {'F1':<8} {'AP_NT':<10} {'AP_T':<10}"
        )
        print("-" * 122)
        for row in all_results:
            dataset_name = row['dataset_name'][:32]
            print(
                f"{dataset_name:<34} {row['ae_model_name']:<20} {row['accuracy']:<10} {row['mAP']:<10} "
                f"{row['precision_micro']:<8} {row['recall_micro']:<8} {row['f1_micro']:<8} "
                f"{row['AP_NonTarget']:<10} {row['AP_Target']:<10}"
            )
    else:
        print(
            f"\n{'Dataset':<34} {'Mode':<20} {'Acc %':<10} {'mAP':<10} {'P':<8} {'R':<8} {'F1':<8} "
            f"{'AP_NS':<10} {'AP_NTSS':<10} {'AP_TSS':<10} {'F1_TSS':<10}"
        )
        print("-" * 154)
        for row in all_results:
            dataset_name = row['dataset_name'][:32]
            print(
                f"{dataset_name:<34} {row['ae_model_name']:<20} {row['accuracy']:<10} {row['mAP']:<10} "
                f"{row['precision_micro']:<8} {row['recall_micro']:<8} {row['f1_micro']:<8} "
                f"{row['AP_NS']:<10} {row['AP_NTSS']:<10} {row['AP_TSS']:<10} {row['f1_TSS']:<10}"
            )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
