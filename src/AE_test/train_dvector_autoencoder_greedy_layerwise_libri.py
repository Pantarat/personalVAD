#!/usr/bin/env python3
"""
Greedy layer-wise pretraining + full-stack finetuning for denoising d-vector autoencoder.

Process:
1) Extract (noisy_dvector, clean_dvector) pairs from LibriSpeech + MUSAN babble.
2) Optionally greedy layer-wise pretrain shallow AEs:
    - Layer k is trained to map H_{k-1}(noisy) -> H_{k-1}(clean).
    - Hidden representation from encoder is used as input for next layer.
3) Assemble the stacked autoencoder.
4) Train the full stack end-to-end on noisy -> clean reconstruction.
"""

import copy
import json
import pickle
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from resemblyzer import VoiceEncoder

from train_dvector_autoencoder_deep_stacked_libri import (
    _checkpoint_dir_for_config,
    _extract_with_cache,
    discover_librispeech_audio_files,
    discover_musan_speech_noise_files,
    extract_noisy_clean_dvector_pairs_from_libri,
    load_noise_segment,
    mix_audio_with_snr,
    split_indices,
)

# Suppress FutureWarning from resemblyzer library (librosa positional args deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="resemblyzer")

# ============================================================================
# CONFIGURATION
# ============================================================================

LIBRISPEECH_ROOT = "../../data/LibriSpeech"
LIBRISPEECH_SUBSETS = [
    "train-other-500",
]
N_LIBRI_UTTERANCES = 40000
MIN_UTTERANCE_SEC = 1.6
MAX_UTTERANCE_SEC = None
CHUNK_DURATION_SEC = 1.6
MAX_CHUNKS_PER_UTTERANCE = 5

MUSAN_SPEECH_NOISE_ROOT = "../../kaldi/egs/pvad/musan/musan_speech_train"
MUSAN_BABBLE_SNR_RANGE_DB = (20.0, 17.0, 15.0, 13.0)
INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS = True  # If True, includes extra pairs where noise is mixed with silence and target is the clean silence d-vector (instead of zero vector)
INCLUDE_CLEAN_IDENTITY_PAIRS = False  # If True, add clean->clean identity pairs to training data

# MODEL_SAVE_DIR = "test_outputs/models/greedy/dvector_ae_greedy_layerwise_15_6-5-26"
MODEL_SAVE_DIR = "test_outputs/models/stacked/dvector_ae_stacked_1_17-7-26"

# Greedy layer-wise architecture (one hidden layer per stage)
GREEDY_LAYER_HIDDEN_DIMS = [192, 192]
ENABLE_GREEDY_LAYERWISE_PRETRAINING = False
DROPOUT_RATE = 0.1
NORM_TYPE = "layernorm"
ACTIVATION_TYPE = "tanh"  # tanh or relu
OUTPUT_NORMALIZATION = "none"  # "l2" or "none"

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE_PRETRAIN = 1e-4
LEARNING_RATE_FINETUNE = 5e-5
NUM_EPOCHS_PRETRAIN = 60
NUM_EPOCHS_FINETUNE = 120
VALIDATION_SPLIT = 0.05
TEST_SPLIT = 0.0
EARLY_STOPPING_PATIENCE = 20

# Mixed reconstruction loss schedule: start with cosine, ramp to MSE
LOSS_MSE_WEIGHT_START = 0.7
LOSS_MSE_WEIGHT_END = 0.7
LOSS_COSINE_WEIGHT_START = 0.3
LOSS_COSINE_WEIGHT_END = 0.3
LOSS_RAMP_EPOCHS = 40
LOSS_COSINE_EPS = 1e-8

# Contrastive negative term (push recon away from other speakers in batch)
NEGATIVE_CONTRASTIVE_WEIGHT = 0.2
NEGATIVE_CONTRASTIVE_MARGIN = 0.2

# Audio settings
SAMPLE_RATE = 16000

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed
RANDOM_SEED = 42

# D-vector extraction cache
DVECTOR_CACHE_ENABLED = True
DVECTOR_CACHE_DIR = "test_outputs/dvector_cache"
DVECTOR_CACHE_VERSION = 1
EXTRACTION_CACHE_NAME = "deep_stacked_libri_babble_pairs"


class NoisyCleanDvectorDataset(Dataset):
    """Dataset for mapping noisy d-vectors to clean d-vectors."""

    def __init__(self, noisy_dvectors, clean_dvectors, indices):
        self.noisy_dvectors = noisy_dvectors
        self.clean_dvectors = clean_dvectors
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        noisy = torch.from_numpy(self.noisy_dvectors[i]).float()
        clean = torch.from_numpy(self.clean_dvectors[i]).float()
        return noisy, clean


class ShallowDenoisingAE(nn.Module):
    """Single-hidden-layer denoising autoencoder used for greedy pretraining."""

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1, norm_type="batchnorm", activation_type="tanh"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.norm_type = str(norm_type).lower()
        self.activation_type = str(activation_type).lower()

        def _activation():
            if self.activation_type == "tanh":
                return nn.Tanh()
            if self.activation_type == "relu":
                return nn.ReLU()
            raise ValueError(f"Unsupported activation_type: {activation_type}")

        def _norm(dim):
            if self.norm_type == "batchnorm":
                return nn.BatchNorm1d(dim)
            if self.norm_type == "layernorm":
                return nn.LayerNorm(dim)
            return nn.Identity()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            _norm(self.hidden_dim),
            _activation(),
            nn.Dropout(self.dropout_rate),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return _normalize_output(recon)

    def encode(self, x):
        return self.encoder(x)


class StackedDenoisingAE(nn.Module):
    """Stacked autoencoder assembled from pretrained shallow AEs."""

    def __init__(self, pretrained_layers=None, input_dim=None, hidden_dims=None, dropout_rate=0.1, norm_type="batchnorm", activation_type="tanh"):
        super().__init__()
        if pretrained_layers is not None:
            self.encoders = nn.ModuleList([copy.deepcopy(layer.encoder) for layer in pretrained_layers])
            self.decoders = nn.ModuleList([copy.deepcopy(layer.decoder) for layer in pretrained_layers])
            return

        if input_dim is None or hidden_dims is None:
            raise ValueError("Either pretrained_layers or both input_dim and hidden_dims must be provided")

        hidden_dims = [int(dim) for dim in hidden_dims]
        encoder_layers = []
        decoder_layers = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layer = ShallowDenoisingAE(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                norm_type=norm_type,
                activation_type=activation_type,
            )
            encoder_layers.append(layer.encoder)
            decoder_layers.append(layer.decoder)
            current_dim = hidden_dim

        self.encoders = nn.ModuleList(encoder_layers)
        self.decoders = nn.ModuleList(decoder_layers)

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        for decoder in reversed(self.decoders):
            out = decoder(out)
        return _normalize_output(out)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_loss(pred, target):
    return 1.0 - F.cosine_similarity(pred, target, dim=1, eps=LOSS_COSINE_EPS).mean()


def _negative_contrastive_loss(pred, target, margin=0.2):
    """Penalize similarity to mismatched targets using a simple in-batch shuffle."""
    batch_size = pred.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=pred.device)

    shuffle = torch.randperm(batch_size, device=pred.device)
    neg_target = target[shuffle]
    cos_sim = F.cosine_similarity(pred, neg_target, dim=1, eps=LOSS_COSINE_EPS)
    return F.relu(cos_sim - margin).mean()


def _loss_weights_for_epoch(epoch_idx, total_epochs):
    ramp_epochs = max(1, min(int(LOSS_RAMP_EPOCHS), int(total_epochs)))
    t = min(1.0, float(epoch_idx) / float(ramp_epochs))
    mse_w = LOSS_MSE_WEIGHT_START + t * (LOSS_MSE_WEIGHT_END - LOSS_MSE_WEIGHT_START)
    cos_w = LOSS_COSINE_WEIGHT_START + t * (LOSS_COSINE_WEIGHT_END - LOSS_COSINE_WEIGHT_START)
    return float(mse_w), float(cos_w)


def _normalize_output(recon):
    mode = str(OUTPUT_NORMALIZATION).lower()
    if mode == "none":
        return recon
    if mode == "l2":
        return F.normalize(recon, p=2, dim=1, eps=LOSS_COSINE_EPS)
    raise ValueError(f"Unsupported OUTPUT_NORMALIZATION: {OUTPUT_NORMALIZATION}")


def _train_epoch(model, loader, optimizer, device, mse_weight, cos_weight):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        noisy = F.normalize(noisy, p=2, dim=1, eps=LOSS_COSINE_EPS)
        clean = F.normalize(clean, p=2, dim=1, eps=LOSS_COSINE_EPS)

        optimizer.zero_grad()
        recon = model(noisy)
        mse = F.mse_loss(recon, clean)
        cos = _cosine_loss(recon, clean)
        neg = _negative_contrastive_loss(
            recon, clean, margin=NEGATIVE_CONTRASTIVE_MARGIN
        )
        loss = (mse_weight * mse) + (cos_weight * cos) + (NEGATIVE_CONTRASTIVE_WEIGHT * neg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_cos += cos.item()

    n_batches = max(1, len(loader))
    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches


def _eval_epoch(model, loader, device, mse_weight, cos_weight):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            noisy = F.normalize(noisy, p=2, dim=1, eps=LOSS_COSINE_EPS)
            clean = F.normalize(clean, p=2, dim=1, eps=LOSS_COSINE_EPS)

            recon = model(noisy)
            mse = F.mse_loss(recon, clean)
            cos = _cosine_loss(recon, clean)
            neg = _negative_contrastive_loss(
                recon, clean, margin=NEGATIVE_CONTRASTIVE_MARGIN
            )
            loss = (mse_weight * mse) + (cos_weight * cos) + (NEGATIVE_CONTRASTIVE_WEIGHT * neg)

            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()

    n_batches = max(1, len(loader))
    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches


def _save_loss_plot(history, save_path, title):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss", linewidth=2)
    ax.plot(epochs, val_loss, label="val_loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _train_model(model, train_loader, val_loader, device, lr, num_epochs, save_path, plot_path=None, plot_title=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    patience = 0
    history = []

    for epoch in range(num_epochs):
        mse_w, cos_w = _loss_weights_for_epoch(epoch, num_epochs)
        train_loss, train_mse, train_cos = _train_epoch(
            model, train_loader, optimizer, device, mse_w, cos_w
        )
        val_loss, val_mse, val_cos = _eval_epoch(model, val_loader, device, mse_w, cos_w)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Train {train_loss:.6f} (mse {train_mse:.6f}, cos {train_cos:.6f}) | "
            f"Val {val_loss:.6f} (mse {val_mse:.6f}, cos {val_cos:.6f}) | "
            f"Weights mse={mse_w:.2f}, cos={cos_w:.2f}"
        )

        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "train_cos": float(train_cos),
                "val_cos": float(val_cos),
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss}, save_path)
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if plot_path is not None and history:
        _save_loss_plot(
            history=history,
            save_path=plot_path,
            title=plot_title or "Training vs Validation Loss",
        )

    return best_val


def _encode_dataset(model, data, batch_size, device):
    model.eval()
    encoded = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.from_numpy(data[i:i + batch_size]).float().to(device)
            batch = F.normalize(batch, p=2, dim=1, eps=LOSS_COSINE_EPS)
            z = model.encode(batch)
            encoded.append(z.cpu().numpy())
    return np.concatenate(encoded, axis=0)


def greedy_pretrain_layers(noisy_dvectors, clean_dvectors, input_dim, device, save_dir):
    pretrained_layers = []
    current_noisy = noisy_dvectors
    current_clean = clean_dvectors

    for idx, hidden_dim in enumerate(GREEDY_LAYER_HIDDEN_DIMS):
        print(f"\n[greedy] Pretraining layer {idx + 1}/{len(GREEDY_LAYER_HIDDEN_DIMS)}: {input_dim} -> {hidden_dim}")

        layer = ShallowDenoisingAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=DROPOUT_RATE,
            norm_type=NORM_TYPE,
            activation_type=ACTIVATION_TYPE,
        ).to(device)

        train_idx, val_idx, _ = split_indices(
            n_items=len(current_noisy),
            validation_split=VALIDATION_SPLIT,
            test_split=0.0,
            random_seed=RANDOM_SEED,
        )
        train_ds = NoisyCleanDvectorDataset(current_noisy, current_clean, train_idx)
        val_ds = NoisyCleanDvectorDataset(current_noisy, current_clean, val_idx)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        layer_ckpt = save_dir / f"greedy_layer_{idx + 1}.pth"
        layer_plot = save_dir / f"greedy_layer_{idx + 1}_loss.png"
        _train_model(
            layer,
            train_loader,
            val_loader,
            device,
            LEARNING_RATE_PRETRAIN,
            NUM_EPOCHS_PRETRAIN,
            layer_ckpt,
            plot_path=layer_plot,
            plot_title=f"Greedy Layer {idx + 1} Loss",
        )
        pretrained_layers.append(layer)

        current_noisy = _encode_dataset(layer, current_noisy, BATCH_SIZE, device)
        current_clean = _encode_dataset(layer, current_clean, BATCH_SIZE, device)
        input_dim = hidden_dim

    return pretrained_layers


def build_stacked_model(input_dim, device, pretrained_layers=None):
    if pretrained_layers is not None:
        return StackedDenoisingAE(pretrained_layers=pretrained_layers).to(device)

    return StackedDenoisingAE(
        input_dim=input_dim,
        hidden_dims=GREEDY_LAYER_HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
        norm_type=NORM_TYPE,
        activation_type=ACTIVATION_TYPE,
    ).to(device)


def main():
    _set_seed(RANDOM_SEED)
    script_dir = Path(__file__).parent

    librispeech_root = Path(LIBRISPEECH_ROOT)
    if not librispeech_root.is_absolute():
        librispeech_root = script_dir / librispeech_root

    musan_root = Path(MUSAN_SPEECH_NOISE_ROOT)
    if not musan_root.is_absolute():
        musan_root = script_dir / musan_root

    save_dir = Path(MODEL_SAVE_DIR)
    if not save_dir.is_absolute():
        save_dir = script_dir / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(DVECTOR_CACHE_DIR)
    if not cache_dir.is_absolute():
        cache_dir = script_dir / cache_dir
    if DVECTOR_CACHE_ENABLED:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("GREEDY LAYER-WISE D-VECTOR DENOISING AUTOENCODER")
    print("=" * 90)

    extraction_cache_config = {
        "source": "deep_stacked_libri_noisy_clean_pairs",
        "librispeech_root": str(librispeech_root.resolve()),
        "librispeech_subsets": list(LIBRISPEECH_SUBSETS),
        "n_librispeech_utterances": int(N_LIBRI_UTTERANCES),
        "min_utterance_sec": float(MIN_UTTERANCE_SEC) if MIN_UTTERANCE_SEC is not None else None,
        "max_utterance_sec": float(MAX_UTTERANCE_SEC) if MAX_UTTERANCE_SEC is not None else None,
        "musan_speech_noise_root": str(musan_root.resolve()),
        "sample_rate": int(SAMPLE_RATE),
        "snr_range_db": list(MUSAN_BABBLE_SNR_RANGE_DB),
        "random_seed": int(RANDOM_SEED),
    }

    if INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS:
        extraction_cache_config["include_noise_only_silence_target_pairs"] = True

    checkpoint_dir = _checkpoint_dir_for_config(cache_dir, EXTRACTION_CACHE_NAME, extraction_cache_config)

    def _extract_pairs():
        return extract_noisy_clean_dvector_pairs_from_libri(
            librispeech_root=librispeech_root,
            librispeech_subsets=LIBRISPEECH_SUBSETS,
            musan_speech_noise_root=musan_root,
            n_utterances=N_LIBRI_UTTERANCES,
            min_utt_sec=MIN_UTTERANCE_SEC,
            max_utt_sec=MAX_UTTERANCE_SEC,
            sample_rate=SAMPLE_RATE,
            device=DEVICE,
            snr_range_db=MUSAN_BABBLE_SNR_RANGE_DB,
            include_noise_only_zero_target_pairs=INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS,
            random_seed=RANDOM_SEED,
            preview_limit=0,
            checkpoint_dir=checkpoint_dir,
        )

    if DVECTOR_CACHE_ENABLED:
        pair_data = _extract_with_cache(
            cache_dir=cache_dir,
            cache_name=EXTRACTION_CACHE_NAME,
            cache_config=extraction_cache_config,
            extractor_fn=_extract_pairs,
        )
    else:
        pair_data = _extract_pairs()

    noisy_dvectors = pair_data["noisy_dvectors"]
    clean_dvectors = pair_data["clean_dvectors"]

    if INCLUDE_CLEAN_IDENTITY_PAIRS:
        # Duplicate clean targets as identity inputs
        noisy_dvectors = np.concatenate([noisy_dvectors, clean_dvectors], axis=0)
        clean_dvectors = np.concatenate([clean_dvectors, clean_dvectors], axis=0)
    input_dim = int(noisy_dvectors.shape[1])

    print(f"\n[data] Pairs: {len(noisy_dvectors)} | Dim: {input_dim}")
    if INCLUDE_CLEAN_IDENTITY_PAIRS:
        print(f"  Added clean->clean identity pairs: {len(clean_dvectors) // 2}")

    pretrained_layers = None
    if ENABLE_GREEDY_LAYERWISE_PRETRAINING:
        pretrained_layers = greedy_pretrain_layers(noisy_dvectors, clean_dvectors, input_dim, DEVICE, save_dir)

    if ENABLE_GREEDY_LAYERWISE_PRETRAINING:
        print("\n[stack] Assembling full stacked autoencoder from pretrained layers")
    else:
        print("\n[stack] Initializing full stacked autoencoder for end-to-end training")

    stacked_model = build_stacked_model(input_dim, DEVICE, pretrained_layers=pretrained_layers)
    if DEVICE == "cuda":
        summary(stacked_model.cuda(), (input_dim,))
    else:
        summary(stacked_model, (input_dim,))

    train_idx, val_idx, test_idx = split_indices(
        n_items=len(noisy_dvectors),
        validation_split=VALIDATION_SPLIT,
        test_split=TEST_SPLIT,
        random_seed=RANDOM_SEED,
    )

    train_ds = NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, train_idx)
    val_ds = NoisyCleanDvectorDataset(noisy_dvectors, clean_dvectors, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("\n[finetune] Training full stacked model end-to-end")
    finetune_ckpt = save_dir / "stacked_finetune_best.pth"
    best_val = _train_model(
        stacked_model,
        train_loader,
        val_loader,
        DEVICE,
        LEARNING_RATE_FINETUNE,
        NUM_EPOCHS_FINETUNE,
        finetune_ckpt,
        plot_path=save_dir / "stacked_finetune_loss.png",
        plot_title="Stacked AE Finetune Loss",
    )

    final_model_path = save_dir / "final_model.pth"
    torch.save(stacked_model.state_dict(), final_model_path)

    config = {
        "model_type": "greedy_layerwise_stacked" if ENABLE_GREEDY_LAYERWISE_PRETRAINING else "stacked_end_to_end",
        "input_dim": input_dim,
        "greedy_hidden_dims": list(GREEDY_LAYER_HIDDEN_DIMS),
        "enable_greedy_layerwise_pretraining": bool(ENABLE_GREEDY_LAYERWISE_PRETRAINING),
        "dropout_rate": float(DROPOUT_RATE),
        "norm_type": str(NORM_TYPE),
        "activation_type": str(ACTIVATION_TYPE),
        "output_normalization": OUTPUT_NORMALIZATION,
        "loss_mse_weight_start": float(LOSS_MSE_WEIGHT_START),
        "loss_mse_weight_end": float(LOSS_MSE_WEIGHT_END),
        "loss_cosine_weight_start": float(LOSS_COSINE_WEIGHT_START),
        "loss_cosine_weight_end": float(LOSS_COSINE_WEIGHT_END),
        "loss_ramp_epochs": int(LOSS_RAMP_EPOCHS),
        "learning_rate_pretrain": float(LEARNING_RATE_PRETRAIN),
        "learning_rate_finetune": float(LEARNING_RATE_FINETUNE),
        "num_epochs_pretrain": int(NUM_EPOCHS_PRETRAIN),
        "num_epochs_finetune": int(NUM_EPOCHS_FINETUNE),
        "best_val_loss": float(best_val),
        "librispeech_root": str(librispeech_root),
        "librispeech_subsets": list(LIBRISPEECH_SUBSETS),
        "musan_speech_noise_root": str(musan_root),
        "musan_babble_snr_range_db": MUSAN_BABBLE_SNR_RANGE_DB,
        "include_noise_only_silence_target_pairs": bool(INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS),
        "include_clean_identity_pairs": bool(INCLUDE_CLEAN_IDENTITY_PAIRS),
        "sample_rate": SAMPLE_RATE,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "test_split": TEST_SPLIT,
        "random_seed": RANDOM_SEED,
    }

    config_path = save_dir / "config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    summary_path = save_dir / "config_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GREEDY LAYER-WISE D-VECTOR AE CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input dim: {input_dim}\n")
        f.write(f"Hidden dims: {GREEDY_LAYER_HIDDEN_DIMS}\n")
        f.write(f"Enable greedy layer-wise pretraining: {ENABLE_GREEDY_LAYERWISE_PRETRAINING}\n")
        f.write(f"Dropout rate: {DROPOUT_RATE}\n")
        f.write(f"Norm type: {NORM_TYPE}\n")
        f.write(f"Activation type: {ACTIVATION_TYPE}\n")
        f.write(f"Output normalization: {OUTPUT_NORMALIZATION}\n")
        f.write(f"Libri root: {librispeech_root}\n")
        f.write(f"Libri subsets: {LIBRISPEECH_SUBSETS}\n")
        f.write(f"MUSAN speech root: {musan_root}\n")
        f.write(f"SNR values (dB): {MUSAN_BABBLE_SNR_RANGE_DB}\n")
        f.write(f"Include noise-only silence-target pairs: {INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS}\n")
        f.write(f"Include clean identity pairs: {INCLUDE_CLEAN_IDENTITY_PAIRS}\n")
        f.write(f"Pretrain LR: {LEARNING_RATE_PRETRAIN}\n")
        f.write(f"Finetune LR: {LEARNING_RATE_FINETUNE}\n")
        f.write(f"Pretrain epochs: {NUM_EPOCHS_PRETRAIN}\n")
        f.write(f"Finetune epochs: {NUM_EPOCHS_FINETUNE}\n")
        f.write(
            "Loss schedule (mse, cos): "
            f"{LOSS_MSE_WEIGHT_START}->{LOSS_MSE_WEIGHT_END}, "
            f"{LOSS_COSINE_WEIGHT_START}->{LOSS_COSINE_WEIGHT_END} "
            f"over {LOSS_RAMP_EPOCHS} epochs\n"
        )
        f.write(f"Best val loss: {best_val:.6f}\n")

    print("\nDone.")
    print(f"  Layer checkpoints: {save_dir}")
    print(f"  Best finetune checkpoint: {finetune_ckpt}")
    print(f"  Finetune loss graph: {save_dir / 'stacked_finetune_loss.png'}")
    print(f"  Final model: {final_model_path}")
    print(f"  Config: {config_path}")


if __name__ == "__main__":
    main()
