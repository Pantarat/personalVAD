"""@package vad_set_ae_eval

Evaluation script for SET-AE models with swappable autoencoder weights.

This script allows you to:
1. Load a trained VAD SET-AE model
2. Swap the autoencoder weights to test different AE models
3. Evaluate performance without retraining the VAD model

Usage:
    python vad_set_ae_eval.py --vad_model path/to/vad_set_ae.pt \
                              --ae_model_path path/to/autoencoder/model \
                              --test_dir data/test \
                              --embed_path embeddings

This is useful for testing different autoencoder models (e.g., different compression
ratios, different training objectives) with the same trained VAD model.
"""

import warnings
# Suppress FutureWarning from resemblyzer library (librosa positional args deprecation)
warnings.filterwarnings('ignore', category=FutureWarning, module='resemblyzer')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse as ap
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import sys
from pathlib import Path

# Import from vad_set_ae
try:
    from AE_test.autoencoder_utils import DvectorAutoencoder, load_autoencoder
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AE_test'))
    from autoencoder_utils import DvectorAutoencoder, load_autoencoder
from vad_set_ae import VadSETAEDataset, pad_collate_with_metadata
from personal_vad import PersonalVAD
from resemblyzer import VoiceEncoder

# Default paths
VAD_MODEL_PATH = 'data/eval_dir/models/vad_set_tanh_score1_10ep.pt'
# VAD_MODEL_PATH = 'vad_set_overlap_trained/vad_set_84_overlap_tanh_score1_200_new.pt'
AE_MODEL_PATH = 'src/AE_test/test_outputs/models/greedy_finetune/61_v2_tests/5_25-7-26'
# AE_MODEL_PATH = 'src/AE_test/test_outputs/models/908_wOV_finetune_asyminfoNCE_v2/dvector_ae-908_4pct_3utt-50spk+300Dev_5s_100pctmainspk_100pctAmp-mainOnly_otherSingles_pretrain-2000_1e-5_100ep_infonce0.2'
DATA_TEST = 'data/121_ov_test_ov100pct_main121_babble_500_27-3-2026'
# DATA_TEST = 'data/121_ov_test_noOther_main121_500_1-6-2026'
EMBED_PATH = 'data/embeddings'
SCORE_TYPE = 1
BATCH_SIZE = 64
NUM_WORKERS = 0
USE_AE = True
USE_AE_RECONSTRUCTION = True
TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION = False  # Safer default: avoids inflated target false positives
TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION = False  # Reconstruction mode only
RECOMPUTE_SCORES = True  # True: recompute scores with AE, False: use pre-computed scores.scp
MAIN_SPEAKER_ID = ''  # Default main speaker ID to filter on
BINARY_CLASSIFICATION = False  # True: 2-class (target vs non-target), False: 3-class (NS, NTSS, TSS)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _get_ae_bottleneck_dim(ae_config):
    """Resolve bottleneck dimension from AE config for standard or stacked variants."""
    if str(ae_config.get('model_type', 'dvector_autoencoder')).lower() == 'deep_stacked_dae':
        return int(ae_config.get('input_dim', 256))

    hidden_dims = ae_config.get('hidden_dims')
    if not hidden_dims:
        raise KeyError("AE config is missing non-empty 'hidden_dims'; cannot determine bottleneck dimension.")
    return int(hidden_dims[len(hidden_dims) // 2])


def evaluate_vad_with_ae(
    vad_model,
    test_loader,
    device,
    show_examples=True,
    n_examples=3,
    binary_mode=False,
    show_class_stats=True,
    return_details=False,
):
    """Evaluate VAD model on test set.

    Returns:
        Tuple of (acc, mAP, out_AP, avg_precision, avg_recall, avg_f1)
        If return_details=True, appends a details dict with:
            - per-class precision/recall/f1
            - confusion matrices (raw and base-1 normalized over total frames)
            - example label/prediction sequences (first n_examples utterances)
    """
    vad_model.eval()
    softmax = nn.Softmax(dim=1)
    
    examples_shown = 0
    example_data = []
    
    with torch.no_grad():
        targets = []
        outputs = []
        
        # Label mapping for binary mode
        if binary_mode:
            # Map: 0 (NS) -> 0, 1 (NTSS) -> 0, 2 (TSS) -> 1
            label_map = {0: 0, 1: 0, 2: 1}
            num_classes = 2
            class_names = ['Non-Target (NS+NTSS)', 'Target (TSS)']
        else:
            label_map = None
            num_classes = 3
            class_names = ['NS (silence)', 'NTSS (other)', 'TSS (target)']

        collect_examples = n_examples > 0 and (show_examples or return_details)
        
        for x_padded, y_padded, x_lens, y_lens, keys in test_loader:
            y_padded = y_padded.to(device)
            
            # Pass through VAD model
            out_padded, _ = vad_model(x_padded.to(device), x_lens, None)

            for j in range(out_padded.size(0)):
                p = softmax(out_padded[j][:y_lens[j]])
                y_true = y_padded[j][:y_lens[j]].cpu().numpy()
                p_np = p.cpu().numpy()
                
                if binary_mode:
                    # Map targets: 0,1 -> 0 (non-target), 2 -> 1 (target)
                    y_eval = np.array([label_map[y] for y in y_true])
                    # Combine output probabilities: P(non-target) = P(NS) + P(NTSS), P(target) = P(TSS)
                    p_eval = np.column_stack([p_np[:, 0] + p_np[:, 1], p_np[:, 2]])
                else:
                    y_eval = y_true
                    p_eval = p_np

                outputs.append(p_eval)
                targets.append(y_eval)

                # Collect examples for first n_examples utterances
                if collect_examples and examples_shown < n_examples:
                    example_data.append({
                        'key': keys[j],
                        'labels': y_eval,
                        'preds': np.argmax(p_eval, axis=1),
                        'probs': p_eval,
                        'length': y_lens[j]
                    })
                    examples_shown += 1
        
        # Compute average precision (matching evaluate_models.py)
        targets = np.concatenate(targets)
        outputs = np.concatenate(outputs)
        
        # Convert the target array to one hot
        targets_oh = np.eye(num_classes)[targets]
        
        # Run the AP
        out_AP = average_precision_score(targets_oh, outputs, average=None)
        mAP = average_precision_score(targets_oh, outputs, average='micro')
        
        # Compute class predictions
        classes = np.argmax(outputs, axis=1)
        
        # Compute the accuracy score
        acc = accuracy_score(classes, targets) * 100
        avg_precision = precision_score(targets, classes, average='micro', zero_division=0)
        avg_recall = recall_score(targets, classes, average='micro', zero_division=0)
        avg_f1 = f1_score(targets, classes, average='micro', zero_division=0)
        
        # Additional statistics for display
        all_targets = targets
        all_predictions = classes
        
        # Count class distribution
        class_indices = list(range(num_classes))
        label_counts = {i: np.sum(all_targets == i) for i in class_indices}
        pred_counts = {i: np.sum(all_predictions == i) for i in class_indices}
        
        # Compute per-class accuracy
        per_class_correct = {}
        per_class_total = {}
        
        for true_class in class_indices:
            mask = all_targets == true_class
            per_class_total[true_class] = np.sum(mask)
            per_class_correct[true_class] = np.sum((all_targets == true_class) & (all_predictions == true_class))
        
        # Compute precision and recall per class
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        for cls in class_indices:
            # Precision: of all frames predicted as this class, how many were correct?
            if pred_counts[cls] > 0:
                per_class_precision[cls] = per_class_correct[cls] / pred_counts[cls]
            else:
                per_class_precision[cls] = 0.0
            
            # Recall: of all frames that are this class, how many did we predict correctly?
            if per_class_total[cls] > 0:
                per_class_recall[cls] = per_class_correct[cls] / per_class_total[cls]
            else:
                per_class_recall[cls] = 0.0

            p_cls = per_class_precision[cls]
            r_cls = per_class_recall[cls]
            per_class_f1[cls] = 2 * p_cls * r_cls / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0.0
        
        class_stats = {
            'label_counts': label_counts,
            'pred_counts': pred_counts,
            'total_frames': len(all_targets),
            'per_class_correct': per_class_correct,
            'per_class_total': per_class_total,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'confusion_matrix': None,  # Raw frame counts
            'confusion_matrix_pct_total': None  # Base-1 fraction over total evaluated frames
        }
        
        # Compute raw confusion matrix for display
        raw_cm = np.zeros((num_classes, num_classes), dtype=int)
        for true_class in class_indices:
            for pred_class in class_indices:
                raw_cm[true_class, pred_class] = np.sum((all_targets == true_class) & (all_predictions == pred_class))
        class_stats['confusion_matrix'] = raw_cm

        # Match batch CSV logic: base-1 fraction over total evaluated frames
        total_frames = np.sum(raw_cm)
        if total_frames > 0:
            class_stats['confusion_matrix_pct_total'] = raw_cm / total_frames
        else:
            class_stats['confusion_matrix_pct_total'] = np.zeros((num_classes, num_classes), dtype=float)
    
    # Display examples
    if show_examples and example_data:
        print("\n" + "=" * 80)
        print(f"EXAMPLE PREDICTIONS (First {len(example_data)} utterances)")
        print("=" * 80)
        
        for idx, ex in enumerate([example_data[0]]):
            print(f"\n[{idx+1}] Utterance: {ex['key']}")
            print(f"    Length: {ex['length']} frames")
            
            # Show first 20 frames and last 10 frames if utterance is long
            n_show_start = min(100, ex['length'])
            n_show_end = min(0, ex['length'] - n_show_start)
            
            print(f"    First {n_show_start} frames:")
            if binary_mode:
                print(f"    {'Frame':<8} {'Label':<25} {'Pred':<25} {'Probabilities (NT/T)'}")
            else:
                print(f"    {'Frame':<8} {'Label':<18} {'Pred':<18} {'Probabilities (NS/NTSS/TSS)'}")
            print(f"    {'-'*70}")
            
            for i in range(n_show_start):
                label = ex['labels'][i]
                pred = ex['preds'][i]
                probs = ex['probs'][i]
                match = '✓' if label == pred else '✗'
                label_str = class_names[label] if label < len(class_names) else str(label)
                pred_str = class_names[pred] if pred < len(class_names) else str(pred)
                if binary_mode:
                    print(f"    {i:<8} {label_str:<25} {pred_str:<25} [{probs[0]:.3f}, {probs[1]:.3f}] {match}")
                else:
                    print(f"    {i:<8} {label_str:<18} {pred_str:<18} [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}] {match}")
            
            if n_show_end > 0 and ex['length'] > n_show_start + 5:
                print(f"    ... ({ex['length'] - n_show_start - n_show_end} frames omitted)")
                print(f"    Last {n_show_end} frames:")
                for i in range(ex['length'] - n_show_end, ex['length']):
                    label = ex['labels'][i]
                    pred = ex['preds'][i]
                    probs = ex['probs'][i]
                    match = '✓' if label == pred else '✗'
                    label_str = class_names[label] if label < len(class_names) else str(label)
                    pred_str = class_names[pred] if pred < len(class_names) else str(pred)
                    if binary_mode:
                        print(f"    {i:<8} {label_str:<25} {pred_str:<25} [{probs[0]:.3f}, {probs[1]:.3f}] {match}")
                    else:
                        print(f"    {i:<8} {label_str:<18} {pred_str:<18} [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}] {match}")
            
            # Per-utterance stats
            utt_correct = np.sum(ex['labels'] == ex['preds'])
            utt_acc = 100.0 * utt_correct / ex['length']
            print(f"    Utterance accuracy: {utt_acc:.2f}% ({utt_correct}/{ex['length']})")
    
    # Display class distribution statistics
    if show_class_stats:
        print("\n" + "=" * 80)
        print("CLASS DISTRIBUTION STATISTICS")
        print("=" * 80)
        print(f"\nTotal frames evaluated: {class_stats['total_frames']:,}")
        print(f"\n{'Class':<20} {'Label Count':<15} {'Label %':<12} {'Pred Count':<15} {'Pred %':<12}")
        print("-" * 80)
        
        for class_id in class_indices:
            class_name = class_names[class_id]
            label_count = class_stats['label_counts'][class_id]
            pred_count = class_stats['pred_counts'][class_id]
            label_pct = 100.0 * label_count / class_stats['total_frames']
            pred_pct = 100.0 * pred_count / class_stats['total_frames']
            
            print(f"{class_name:<20} {label_count:<15,} {label_pct:<12.2f} {pred_count:<15,} {pred_pct:<12.2f}")
        
        # Show difference
        print("\n" + "-" * 80)
        print("Distribution differences (Predicted - Label):")
        for class_id in class_indices:
            class_name = class_names[class_id]
            diff = class_stats['pred_counts'][class_id] - class_stats['label_counts'][class_id]
            diff_pct = 100.0 * diff / class_stats['total_frames']
            sign = "+" if diff > 0 else ""
            print(f"  {class_name:<20} {sign}{diff:,} frames ({sign}{diff_pct:.2f}%)")
        
        # Show per-class metrics
        print("\n" + "=" * 80)
        print("PER-CLASS PERFORMANCE METRICS")
        print("=" * 80)
        print(f"\n{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AP':<12} {'Accuracy':<12}")
        print("-" * 80)
        
        for class_id in class_indices:
            class_name = class_names[class_id]
            precision = class_stats['per_class_precision'][class_id]
            recall = class_stats['per_class_recall'][class_id]
            f1 = class_stats['per_class_f1'][class_id]
            ap = out_AP[class_id]
            class_acc = class_stats['per_class_correct'][class_id] / class_stats['per_class_total'][class_id] if class_stats['per_class_total'][class_id] > 0 else 0.0
            
            print(f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {ap:<12.4f} {class_acc:<12.4f}")
        
        # Confusion matrix (raw counts)
        print("\n" + "=" * 80)
        print("CONFUSION MATRIX (rows=true label, cols=predicted)")
        print("=" * 80)
        raw_cm = class_stats['confusion_matrix']
        header = "True\\Pred"
        if binary_mode:
            col_headers = [f"{class_names[i]:<20}" for i in class_indices]
            print(f"\n{header:<20} {' '.join(col_headers)}")
        else:
            print(f"\n{header:<20} {'NS':<15} {'NTSS':<15} {'TSS':<15}")
        print("-" * 65)
        for true_class in class_indices:
            class_name = class_names[true_class]
            if binary_mode:
                row_vals = [f"{raw_cm[true_class, i]:<20,}" for i in class_indices]
                print(f"{class_name:<20} {' '.join(row_vals)}")
            else:
                row = [raw_cm[true_class, pred_class] for pred_class in class_indices]
                print(f"{class_name:<20} {row[0]:<15,} {row[1]:<15,} {row[2]:<15,}")
        
        # Confusion matrix base-1 values over total evaluated frames (matches batch CSV)
        print("\n" + "=" * 80)
        print("CONFUSION MATRIX (base-1 over total frames)")
        print("=" * 80)
        norm_cm = class_stats['confusion_matrix_pct_total']
        if binary_mode:
            col_headers = [f"{class_names[i]:<20}" for i in class_indices]
            print(f"\n{header:<20} {' '.join(col_headers)}")
        else:
            print(f"\n{header:<20} {'NS':<15} {'NTSS':<15} {'TSS':<15}")
        print("-" * 65)
        for true_class in class_indices:
            class_name = class_names[true_class]
            if binary_mode:
                row_vals = [f"{norm_cm[true_class, i]:<20.4f}" for i in class_indices]
                print(f"{class_name:<20} {' '.join(row_vals)}")
            else:
                row = [norm_cm[true_class, pred_class] for pred_class in class_indices]
                print(f"{class_name:<20} {row[0]:<15.4f} {row[1]:<15.4f} {row[2]:<15.4f}")

    if return_details:
        details = {
            'class_names': class_names,
            'per_class_precision': class_stats['per_class_precision'],
            'per_class_recall': class_stats['per_class_recall'],
            'per_class_f1': class_stats['per_class_f1'],
            'confusion_matrix': class_stats['confusion_matrix'],
            'confusion_matrix_pct_total': class_stats['confusion_matrix_pct_total'],
            'example_sequences': [
                {
                    'key': ex['key'],
                    'true': np.asarray(ex['labels'], dtype=np.int32),
                    'pred': np.asarray(ex['preds'], dtype=np.int32),
                }
                for ex in example_data
            ],
        }
        return acc, mAP, out_AP, avg_precision, avg_recall, avg_f1, details

    return acc, mAP, out_AP, avg_precision, avg_recall, avg_f1


if __name__ == '__main__':
    parser = ap.ArgumentParser(
        description="Evaluate VAD SET-AE model with swappable autoencoder weights."
    )
    parser.add_argument('--vad_model', type=str, default=VAD_MODEL_PATH, required=False,
                        help=f'Path to trained VAD model (.pt file). Default: {VAD_MODEL_PATH}')
    parser.add_argument('--ae_model_path', type=str, default=None, required=False,
                        help=f'Path to autoencoder model directory to use for evaluation. If not provided and --use_full_dvec is not set, uses default: {AE_MODEL_PATH}')
    parser.add_argument('--test_dir', type=str, default=DATA_TEST,
                        help='Test data directory')
    parser.add_argument('--embed_path', type=str, default=EMBED_PATH,
                        help='Path to speaker embeddings directory')
    parser.add_argument('--score_type', type=int, default=SCORE_TYPE,
                        help='Scoring method: 0=baseline, 1=PC, 2=LI')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for evaluation')
    parser.add_argument('--use_full_dvec', action='store_true',
                        help='Use full 256-dim d-vectors instead of compressed')
    parser.add_argument('--use_ae_reconstruction', action='store_true',
                        help='Use full autoencoder reconstruction (enc+dec: 256->64->256) instead of bottleneck only (256->64)')
    parser.add_argument('--transform_enrolled_dvector', action='store_true',
                        help='In reconstruction mode, transform enrolled d-vectors for scoring anchor (default keeps enrolled vectors original).')
    parser.add_argument('--transform_enrolled_input_dvector', action='store_true',
                        help='In reconstruction mode, also feed transformed enrolled d-vectors into VAD input (can hurt precision).')
    parser.add_argument('--recompute_scores', action='store_true', default=RECOMPUTE_SCORES,
                        help='Recompute speaker verification scores using autoencoder-processed d-vectors instead of using pre-computed scores.scp')
    parser.add_argument('--audio_root', type=str, default=None,
                        help='Path to audio root directory (for wav.scp when recomputing scores)')
    parser.add_argument('--main_speaker', type=str, default=MAIN_SPEAKER_ID,
                        help='Only evaluate on files from this speaker ID (e.g., "84"). If not specified, evaluates on all speakers.')
    parser.add_argument('--binary_classification', action='store_true', default=BINARY_CLASSIFICATION,
                        help='Use binary classification mode: target speaker vs non-target (combines NS and NTSS into single class)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("VAD SET-AE EVALUATION WITH SWAPPABLE AUTOENCODER")
    print("=" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"   VAD Model: {args.vad_model}")
    print(f"   Test Dir: {args.test_dir}")
    print(f"   Score Type: {args.score_type}")
    if args.main_speaker:
        print(f"   Main Speaker: {args.main_speaker} (filtering enabled)")
    
    # Determine if using autoencoder
    # Command line --use_full_dvec overrides file flag USE_AE
    if args.use_full_dvec:
        use_autoencoder = False
    else:
        use_autoencoder = USE_AE
    
    # Command line --use_ae_reconstruction overrides file flag USE_AE_RECONSTRUCTION
    if args.use_ae_reconstruction:
        use_ae_reconstruction = True
    else:
        use_ae_reconstruction = USE_AE_RECONSTRUCTION

    # Command line --transform_enrolled_dvector overrides file flag
    if args.transform_enrolled_dvector:
        transform_enrolled_dvector = True
    else:
        transform_enrolled_dvector = TRANSFORM_ENROLLED_DVECTOR_FORSIMSCORE_IN_RECONSTRUCTION

    if args.transform_enrolled_input_dvector:
        transform_enrolled_input_dvector = True
    else:
        transform_enrolled_input_dvector = TRANSFORM_ENROLLED_VADINPUT_DVECTOR_IN_RECONSTRUCTION

    if not use_ae_reconstruction and transform_enrolled_dvector:
        print("\n⚠️  --transform_enrolled_dvector is only used with reconstruction mode. Ignoring flag.")
        transform_enrolled_dvector = False
    if not use_ae_reconstruction and transform_enrolled_input_dvector:
        print("\n⚠️  --transform_enrolled_input_dvector is only used with reconstruction mode. Ignoring flag.")
        transform_enrolled_input_dvector = False
    
    # Set default AE model path if not provided and using autoencoder
    if use_autoencoder and args.ae_model_path is None:
        args.ae_model_path = AE_MODEL_PATH
    
    # Load autoencoder
    if use_autoencoder:
        print(f"\n🔧 Loading autoencoder model...")
        print(f"   Requested path (args.ae_model_path): {args.ae_model_path}")
        
        # Resolve the path to show what will actually be loaded
        from pathlib import Path as PathLib
        model_dir_check = PathLib(args.ae_model_path)
        if not model_dir_check.is_absolute():
            script_dir = PathLib(__file__).parent.parent
            model_dir_check = script_dir / model_dir_check
        print(f"   Resolved absolute path: {model_dir_check}")
        print(f"   Config file: {model_dir_check / 'config.pkl'}")
        print(f"   Config exists: {(model_dir_check / 'config.pkl').exists()}")
        
        autoencoder, ae_config = load_autoencoder(args.ae_model_path, device)
        autoencoder.eval()
        encoded_dim = _get_ae_bottleneck_dim(ae_config)
        ae_model_type = ae_config.get('model_type', 'dvector_autoencoder')
        
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
        else:
            print(f"   Architecture: {ae_config['hidden_dims']}")
        print(f"   Bottleneck/output dim: {encoded_dim}-dim")
        if use_ae_reconstruction:
            print(f"   Mode: Full reconstruction (256→{encoded_dim}→256)")
            if transform_enrolled_dvector:
                print(f"   Enrolled scoring anchor: transformed (256→{encoded_dim}→256)")
            else:
                print(f"   Enrolled scoring anchor: original")
            if transform_enrolled_input_dvector:
                print(f"   Enrolled VAD input: transformed (256→{encoded_dim}→256)")
            else:
                print(f"   Enrolled VAD input: original")
        else:
            print(f"   Mode: Bottleneck only (256→{encoded_dim})")
        
        # Show training scheme if present (helps identify extraction vs identity AE)
        if 'training_scheme' in ae_config:
            print(f"   ⚠️  Training scheme: {ae_config['training_scheme']} (EXTRACTION model)")
        else:
            print(f"   ✓ Training scheme: Not present (likely IDENTITY model)")
        
        if 'main_speakers' in ae_config:
            print(f"   Main speakers: {ae_config['main_speakers']}")
        if 'best_val_loss' in ae_config:
            print(f"   Val loss: {ae_config['best_val_loss']:.6f}")
        if 'test_cosine_similarity_mean' in ae_config:
            pass
            # print(f"   Test cos sim: {ae_config['test_cosine_similarity_mean']:.4f}")
        
        if use_ae_reconstruction:
            input_dim = 297  # 40 fbanks + 256 enrolled + 1 score
        else:
            input_dim = 40 + encoded_dim + 1
    else:
        print(f"\n⚠️  Using full 256-dim d-vectors (no compression)")
        autoencoder = None
        use_ae_reconstruction = False
        input_dim = 297  # 40 fbanks + 256 full + 1 score
    
    # Load test dataset with the specified autoencoder
    print(f"\n📂 Loading test dataset...")
    print(f"   Test dir: {args.test_dir}")
    print(f"   Embeddings: {args.embed_path}")
    print(f"   Score type: {args.score_type}")
    if args.recompute_scores:
        print(f"   Score recomputation: ENABLED (using autoencoder-processed d-vectors)")
    else:
        print(f"   Score recomputation: DISABLED (using pre-computed scores.scp)")
    
    test_data = VadSETAEDataset(
        args.test_dir,
        args.embed_path,
        args.score_type,
        autoencoder=autoencoder,
        use_autoencoder=use_autoencoder,
        use_ae_reconstruction=use_ae_reconstruction,
        transform_enrolled_in_reconstruction=transform_enrolled_dvector,
        transform_enrolled_vadinput_in_reconstruction=transform_enrolled_input_dvector,
        recompute_scores=args.recompute_scores,
        audio_root=args.audio_root
    )
    
    total_samples = len(test_data)
    
    # Filter by main_speaker if specified
    if args.main_speaker:
        print(f"\n🔍 Filtering dataset for speaker: {args.main_speaker}")
        original_indices = []
        for idx in range(len(test_data)):
            # Get the key from the dataset
            # Keys are typically in format like "84-121123-0000" or similar
            key = test_data.keys[idx]
            # Extract speaker ID (first part before hyphen)
            speaker_id = key.split('-')[0]
            if speaker_id == args.main_speaker:
                original_indices.append(idx)
        
        if len(original_indices) == 0:
            print(f"\n❌ ERROR: No samples found for speaker '{args.main_speaker}' in test set!")
            print(f"   Total samples in dataset: {total_samples}")
            print(f"   Please check that the speaker ID is correct.")
            sys.exit(1)
        
        # Create filtered dataset
        from torch.utils.data import Subset
        test_data = Subset(test_data, original_indices)
        
        print(f"   Found {len(test_data)} samples for speaker {args.main_speaker} (out of {total_samples} total)")
        print(f"   Filtering ratio: {100*len(test_data)/total_samples:.1f}%")
    else:
        print(f"✅ Test samples: {total_samples} (all speakers)")
    
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate_with_metadata
    )
    
    # Load VAD model
    print(f"\n🏗️  Loading VAD model...")
    print(f"   Model path: {args.vad_model}")
    
    # Determine VAD model architecture
    hidden_dim = 64
    num_layers = 2
    out_dim = 3
    
    vad_model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=True, linear=False)
    
    # Load weights
    if device == torch.device('cuda') and torch.cuda.is_available():
        checkpoint = torch.load(args.vad_model)
    else:
        checkpoint = torch.load(args.vad_model, map_location='cpu')
    
    # Extract model state dict from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        vad_model.load_state_dict(state_dict)
        
        # Verify input dimension matches
        if 'lstm.weight_ih_l0' in state_dict:
            trained_input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
            if trained_input_dim != input_dim:
                print(f"\n⚠️  WARNING: Input dimension mismatch!")
                print(f"   Model was trained with: {trained_input_dim}-dim input")
                print(f"   Current configuration: {input_dim}-dim input")
                print(f"\n   This will cause poor performance!")
                if trained_input_dim == 105:
                    print(f"   → Model expects COMPRESSED d-vectors (bottleneck dim from AE config)")
                    print(f"   → Set USE_AE=True or use --ae_model_path")
                elif trained_input_dim == 297:
                    print(f"   → Model expects FULL d-vectors (256-dim)")
                    print(f"   → Set USE_AE=False or use --use_full_dvec")
                sys.exit(1)
    else:
        vad_model.load_state_dict(checkpoint)
    
    vad_model = vad_model.to(device)
    
    vad_params = sum(p.numel() for p in vad_model.parameters())
    print(f"✅ VAD model loaded successfully!")
    print(f"   Input: {input_dim}-dim")
    print(f"   Hidden: {hidden_dim}-dim")
    print(f"   Layers: {num_layers}")
    print(f"   Parameters: {vad_params:,}")
    
    # Count total parameters across all components
    print(f"\n📊 Total Model Parameters:")
    
    # 1. D-vector encoder (resemblyzer VoiceEncoder)
    print(f"\n   Loading VoiceEncoder to count parameters...")
    voice_encoder = VoiceEncoder(device=device)
    dvector_params = sum(p.numel() for p in voice_encoder.parameters())
    print(f"   1. D-Vector Encoder (resemblyzer): {dvector_params:,} parameters")
    
    # 2. Autoencoder (if used)
    ae_params = 0
    if use_autoencoder and autoencoder is not None:
        ae_params = sum(p.numel() for p in autoencoder.parameters())
        print(f"   2. Autoencoder: {ae_params:,} parameters")
    else:
        print(f"   2. Autoencoder: Not used (0 parameters)")
    
    # 3. VAD model
    print(f"   3. VAD Model (PersonalVAD): {vad_params:,} parameters")
    
    # Total
    total_all_params = dvector_params + ae_params + vad_params
    print(f"\n   🔢 TOTAL: {total_all_params:,} parameters")
    print(f"      Breakdown: {dvector_params:,} (d-vec) + {ae_params:,} (AE) + {vad_params:,} (VAD)")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING")
    if args.binary_classification:
        print("Mode: BINARY CLASSIFICATION (Target vs Non-Target)")
    else:
        print("Mode: 3-CLASS CLASSIFICATION (NS, NTSS, TSS)")
    print("=" * 80)
    
    acc, mAP, out_AP, avg_precision, avg_recall, avg_f1 = evaluate_vad_with_ae(
        vad_model,
        test_loader,
        device,
        show_examples=False,
        n_examples=3,
        binary_mode=args.binary_classification
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n📊 Accuracy: {acc:.2f}%")
    print(f"📊 mAP: {mAP:.4f}")
    print(f"📊 Precision (micro): {avg_precision:.4f}")
    print(f"📊 Recall (micro): {avg_recall:.4f}")
    print(f"📊 F1-score (micro): {avg_f1:.4f}")
    print(f"📊 Per-class AP: {out_AP}")
    
    if use_autoencoder:
        print(f"\n💡 Evaluated with autoencoder: {args.ae_model_path}")
        if use_ae_reconstruction:
            print(f"   Mode: Full reconstruction (256→{encoded_dim}→256)")
            if transform_enrolled_dvector:
                print(f"   Enrolled scoring anchor: transformed")
            else:
                print(f"   Enrolled scoring anchor: original")
            if transform_enrolled_input_dvector:
                print(f"   Enrolled VAD input: transformed")
            else:
                print(f"   Enrolled VAD input: original")
            print(f"   Purpose: Denoising and speaker extraction")
        else:
            print(f"   Mode: Bottleneck compression (256→{encoded_dim})")
            print(f"   Compression: {256/encoded_dim:.1f}x")
    else:
        print(f"\n💡 Evaluated with full 256-dim d-vectors")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
