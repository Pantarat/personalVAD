#!/usr/bin/env python3
"""
Loss weight sweep for greedy layer-wise d-vector autoencoder training.

Runs all combinations of MSE/Cosine/Negative weights in step increments
where the total weight sum is exactly 1.0.
"""

import builtins
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_dvector_autoencoder_greedy_layerwise_libri as base

# Sweep settings
SWEEP_STEP = 0.1
SWEEP_SAVE_ROOT = "test_outputs/models/greedy/dvector_ae_greedy_layerwise_weight_sweep_15_22-5-26"
PRINT_MODEL_SUMMARY = False
RESUME_FROM_LAST_COMPLETED = True
RESUME_MARKER_FILES = ("config.pkl", "final_model.pth")

_ORIGINAL_PRINT = builtins.print


def _timestamp_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", None)
    flush = kwargs.get("flush", False)
    text = sep.join("" if arg is None else str(arg) for arg in args)

    lines = text.split("\n")
    for idx, line in enumerate(lines):
        line_end = end if idx == len(lines) - 1 else "\n"
        if line == "":
            _ORIGINAL_PRINT(f"[{timestamp}]", end=line_end, file=file, flush=flush)
        else:
            _ORIGINAL_PRINT(f"[{timestamp}] {line}", end=line_end, file=file, flush=flush)


builtins.print = _timestamp_print


def _resolve_dir(path_value, script_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = script_dir / path
    return path


def _generate_loss_weight_combinations(step):
    step = float(step)
    if step <= 0.0:
        raise ValueError("SWEEP_STEP must be > 0")
    total_units = int(round(1.0 / step))
    if abs((total_units * step) - 1.0) > 1e-6:
        raise ValueError("SWEEP_STEP must evenly divide 1.0")

    combos = []
    for mse_units in range(total_units + 1):
        for cos_units in range(total_units + 1 - mse_units):
            neg_units = total_units - mse_units - cos_units
            combos.append((round(mse_units * step, 1), round(cos_units * step, 1), round(neg_units * step, 1)))
    return combos


def _format_weight_tag(mse_w, cos_w, neg_w):
    tag = f"mse{mse_w:.1f}_cos{cos_w:.1f}_neg{neg_w:.1f}"
    return tag.replace(".", "p")


def _run_is_complete(run_dir):
    for marker in RESUME_MARKER_FILES:
        if (run_dir / marker).exists():
            return True
    return False


def _find_resume_start_index(sweep_root, combos):
    last_completed_idx = -1
    for idx, (mse_w, cos_w, neg_w) in enumerate(combos):
        run_tag = _format_weight_tag(mse_w, cos_w, neg_w)
        run_dir = sweep_root / run_tag
        if _run_is_complete(run_dir):
            last_completed_idx = idx
    return last_completed_idx + 1, last_completed_idx


def _prepare_pair_data(script_dir):
    librispeech_root = _resolve_dir(base.LIBRISPEECH_ROOT, script_dir)
    musan_root = _resolve_dir(base.MUSAN_SPEECH_NOISE_ROOT, script_dir)

    cache_dir = _resolve_dir(base.DVECTOR_CACHE_DIR, script_dir)
    if base.DVECTOR_CACHE_ENABLED:
        cache_dir.mkdir(parents=True, exist_ok=True)

    extraction_cache_config = {
        "source": "deep_stacked_libri_noisy_clean_pairs",
        "librispeech_root": str(librispeech_root.resolve()),
        "librispeech_subsets": list(base.LIBRISPEECH_SUBSETS),
        "n_librispeech_utterances": int(base.N_LIBRI_UTTERANCES),
        "min_utterance_sec": float(base.MIN_UTTERANCE_SEC) if base.MIN_UTTERANCE_SEC is not None else None,
        "max_utterance_sec": float(base.MAX_UTTERANCE_SEC) if base.MAX_UTTERANCE_SEC is not None else None,
        "musan_speech_noise_root": str(musan_root.resolve()),
        "sample_rate": int(base.SAMPLE_RATE),
        "snr_range_db": list(base.MUSAN_BABBLE_SNR_RANGE_DB),
        "random_seed": int(base.RANDOM_SEED),
    }

    if base.INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS:
        extraction_cache_config["include_noise_only_silence_target_pairs"] = True

    checkpoint_dir = base._checkpoint_dir_for_config(
        cache_dir, base.EXTRACTION_CACHE_NAME, extraction_cache_config
    )

    def _extract_pairs():
        return base.extract_noisy_clean_dvector_pairs_from_libri(
            librispeech_root=librispeech_root,
            librispeech_subsets=base.LIBRISPEECH_SUBSETS,
            musan_speech_noise_root=musan_root,
            n_utterances=base.N_LIBRI_UTTERANCES,
            min_utt_sec=base.MIN_UTTERANCE_SEC,
            max_utt_sec=base.MAX_UTTERANCE_SEC,
            sample_rate=base.SAMPLE_RATE,
            device=base.DEVICE,
            snr_range_db=base.MUSAN_BABBLE_SNR_RANGE_DB,
            include_noise_only_zero_target_pairs=base.INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS,
            random_seed=base.RANDOM_SEED,
            preview_limit=0,
            checkpoint_dir=checkpoint_dir,
        )

    if base.DVECTOR_CACHE_ENABLED:
        pair_data = base._extract_with_cache(
            cache_dir=cache_dir,
            cache_name=base.EXTRACTION_CACHE_NAME,
            cache_config=extraction_cache_config,
            extractor_fn=_extract_pairs,
        )
    else:
        pair_data = _extract_pairs()

    return pair_data, librispeech_root, musan_root


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _run_training_with_weights(
    noisy_dvectors,
    clean_dvectors,
    librispeech_root,
    musan_root,
    mse_w,
    cos_w,
    neg_w,
    run_dir,
):
    weight_sum = mse_w + cos_w + neg_w
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weight sum must be 1.0, got {weight_sum:.4f}")

    base.LOSS_MSE_WEIGHT_START = float(mse_w)
    base.LOSS_MSE_WEIGHT_END = float(mse_w)
    base.LOSS_COSINE_WEIGHT_START = float(cos_w)
    base.LOSS_COSINE_WEIGHT_END = float(cos_w)
    base.NEGATIVE_CONTRASTIVE_WEIGHT = float(neg_w)
    base._set_seed(base.RANDOM_SEED)

    run_dir.mkdir(parents=True, exist_ok=True)
    input_dim = int(noisy_dvectors.shape[1])

    print(f"\n[data] Pairs: {len(noisy_dvectors)} | Dim: {input_dim}")
    if base.INCLUDE_CLEAN_IDENTITY_PAIRS:
        print(f"  Added clean->clean identity pairs: {len(clean_dvectors) // 2}")

    pretrained_layers = base.greedy_pretrain_layers(
        noisy_dvectors, clean_dvectors, input_dim, base.DEVICE, run_dir
    )

    print("\n[stack] Assembling full stacked autoencoder")
    stacked_model = base.StackedDenoisingAE(pretrained_layers).to(base.DEVICE)
    if PRINT_MODEL_SUMMARY:
        if base.DEVICE == "cuda":
            base.summary(stacked_model.cuda(), (input_dim,))
        else:
            base.summary(stacked_model, (input_dim,))

    train_idx, val_idx, test_idx = base.split_indices(
        n_items=len(noisy_dvectors),
        validation_split=base.VALIDATION_SPLIT,
        test_split=base.TEST_SPLIT,
        random_seed=base.RANDOM_SEED,
    )

    train_ds = base.NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, train_idx)
    val_ds = base.NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, val_idx)
    train_loader = DataLoader(train_ds, batch_size=base.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=base.BATCH_SIZE, shuffle=False)

    print("\n[finetune] Training full stacked model end-to-end")
    finetune_ckpt = run_dir / "stacked_finetune_best.pth"
    best_val = base._train_model(
        stacked_model,
        train_loader,
        val_loader,
        base.DEVICE,
        base.LEARNING_RATE_FINETUNE,
        base.NUM_EPOCHS_FINETUNE,
        finetune_ckpt,
    )

    final_model_path = run_dir / "final_model.pth"
    torch.save(stacked_model.state_dict(), final_model_path)

    config = {
        "model_type": "greedy_layerwise_stacked",
        "input_dim": input_dim,
        "greedy_hidden_dims": list(base.GREEDY_LAYER_HIDDEN_DIMS),
        "dropout_rate": float(base.DROPOUT_RATE),
        "norm_type": str(base.NORM_TYPE),
        "activation_type": str(base.ACTIVATION_TYPE),
        "output_normalization": base.OUTPUT_NORMALIZATION,
        "loss_mse_weight_start": float(base.LOSS_MSE_WEIGHT_START),
        "loss_mse_weight_end": float(base.LOSS_MSE_WEIGHT_END),
        "loss_cosine_weight_start": float(base.LOSS_COSINE_WEIGHT_START),
        "loss_cosine_weight_end": float(base.LOSS_COSINE_WEIGHT_END),
        "loss_negative_weight": float(base.NEGATIVE_CONTRASTIVE_WEIGHT),
        "loss_ramp_epochs": int(base.LOSS_RAMP_EPOCHS),
        "learning_rate_pretrain": float(base.LEARNING_RATE_PRETRAIN),
        "learning_rate_finetune": float(base.LEARNING_RATE_FINETUNE),
        "num_epochs_pretrain": int(base.NUM_EPOCHS_PRETRAIN),
        "num_epochs_finetune": int(base.NUM_EPOCHS_FINETUNE),
        "best_val_loss": float(best_val),
        "librispeech_root": str(librispeech_root),
        "librispeech_subsets": list(base.LIBRISPEECH_SUBSETS),
        "musan_speech_noise_root": str(musan_root),
        "musan_babble_snr_range_db": base.MUSAN_BABBLE_SNR_RANGE_DB,
        "include_noise_only_silence_target_pairs": bool(base.INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS),
        "include_clean_identity_pairs": bool(base.INCLUDE_CLEAN_IDENTITY_PAIRS),
        "sample_rate": base.SAMPLE_RATE,
        "batch_size": base.BATCH_SIZE,
        "validation_split": base.VALIDATION_SPLIT,
        "test_split": base.TEST_SPLIT,
        "random_seed": base.RANDOM_SEED,
    }

    config_path = run_dir / "config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    summary_path = run_dir / "config_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GREEDY LAYER-WISE D-VECTOR AE CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input dim: {input_dim}\n")
        f.write(f"Hidden dims: {base.GREEDY_LAYER_HIDDEN_DIMS}\n")
        f.write(f"Dropout rate: {base.DROPOUT_RATE}\n")
        f.write(f"Norm type: {base.NORM_TYPE}\n")
        f.write(f"Activation type: {base.ACTIVATION_TYPE}\n")
        f.write(f"Output normalization: {base.OUTPUT_NORMALIZATION}\n")
        f.write(f"Libri root: {librispeech_root}\n")
        f.write(f"Libri subsets: {base.LIBRISPEECH_SUBSETS}\n")
        f.write(f"MUSAN speech root: {musan_root}\n")
        f.write(f"SNR values (dB): {base.MUSAN_BABBLE_SNR_RANGE_DB}\n")
        f.write(f"Include noise-only silence-target pairs: {base.INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS}\n")
        f.write(f"Include clean identity pairs: {base.INCLUDE_CLEAN_IDENTITY_PAIRS}\n")
        f.write(f"Pretrain LR: {base.LEARNING_RATE_PRETRAIN}\n")
        f.write(f"Finetune LR: {base.LEARNING_RATE_FINETUNE}\n")
        f.write(f"Pretrain epochs: {base.NUM_EPOCHS_PRETRAIN}\n")
        f.write(f"Finetune epochs: {base.NUM_EPOCHS_FINETUNE}\n")
        f.write(
            "Loss schedule (mse, cos): "
            f"{base.LOSS_MSE_WEIGHT_START}->{base.LOSS_MSE_WEIGHT_END}, "
            f"{base.LOSS_COSINE_WEIGHT_START}->{base.LOSS_COSINE_WEIGHT_END} "
            f"over {base.LOSS_RAMP_EPOCHS} epochs\n"
        )
        f.write(f"Loss negative weight: {base.NEGATIVE_CONTRASTIVE_WEIGHT}\n")
        f.write(f"Loss weight sum: {weight_sum:.2f}\n")
        f.write(f"Best val loss: {best_val:.6f}\n")

    print("\nDone.")
    print(f"  Layer checkpoints: {run_dir}")
    print(f"  Best finetune checkpoint: {finetune_ckpt}")
    print(f"  Final model: {final_model_path}")
    print(f"  Config: {config_path}")


def main():
    script_dir = Path(__file__).parent

    print("=" * 90)
    print("GREEDY LAYER-WISE D-VECTOR DENOISING AUTOENCODER SWEEP")
    print("=" * 90)

    pair_data, librispeech_root, musan_root = _prepare_pair_data(script_dir)
    noisy_dvectors = pair_data["noisy_dvectors"]
    clean_dvectors = pair_data["clean_dvectors"]

    if base.INCLUDE_CLEAN_IDENTITY_PAIRS:
        noisy_dvectors = np.concatenate([noisy_dvectors, clean_dvectors], axis=0)
        clean_dvectors = np.concatenate([clean_dvectors, clean_dvectors], axis=0)

    sweep_root = _resolve_dir(SWEEP_SAVE_ROOT, script_dir)
    combos = _generate_loss_weight_combinations(SWEEP_STEP)
    print(f"\n[sweep] Total combos: {len(combos)} | Step: {SWEEP_STEP} | Combos: {combos}")

    start_idx = 0
    if RESUME_FROM_LAST_COMPLETED:
        start_idx, last_completed_idx = _find_resume_start_index(sweep_root, combos)
        if last_completed_idx >= 0:
            last_combo = combos[last_completed_idx]
            print(
                f"\n[sweep] Resuming after index {last_completed_idx} | "
                f"Last completed: {last_combo}"
            )
        if start_idx >= len(combos):
            print("\n[sweep] All combinations already completed.")
            return

    durations_sec = []
    total_runs = len(combos)

    for run_idx, (mse_w, cos_w, neg_w) in enumerate(combos[start_idx:], start=start_idx + 1):
        run_tag = _format_weight_tag(mse_w, cos_w, neg_w)
        run_dir = sweep_root / run_tag
        print(f"\n[sweep] Run {run_idx}/{len(combos)} | {run_tag}")

        run_start = time.time()
        _run_training_with_weights(
            noisy_dvectors,
            clean_dvectors,
            librispeech_root,
            musan_root,
            mse_w,
            cos_w,
            neg_w,
            run_dir,
        )
        run_elapsed = time.time() - run_start
        durations_sec.append(run_elapsed)

        avg_elapsed = sum(durations_sec) / len(durations_sec)
        remaining_runs = total_runs - run_idx
        eta_seconds = avg_elapsed * remaining_runs

        print(
            "\n[timing] "
            f"Run: {_format_duration(run_elapsed)} | "
            f"Avg: {_format_duration(avg_elapsed)} | "
            f"Remaining: {remaining_runs} | "
            f"ETA: {_format_duration(eta_seconds)}"
        )


if __name__ == "__main__":
    main()
