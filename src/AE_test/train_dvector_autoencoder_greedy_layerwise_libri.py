#!/usr/bin/env python3
"""
Greedy layer-wise pretraining + full-stack finetuning for denoising d-vector autoencoder.

Process:
1) Extract (noisy_dvector, clean_dvector) pairs from LibriSpeech + MUSAN babble.
2) Greedy layer-wise pretrain shallow AEs:
   - Layer k is trained to map H_{k-1}(noisy) -> H_{k-1}(clean).
   - Hidden representation from encoder is used as input for next layer.
3) Assemble stacked autoencoder from pretrained layers.
4) Finetune the full stack end-to-end on noisy -> clean reconstruction.
"""

import copy
import json
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from resemblyzer import VoiceEncoder

from train_dvector_autoencoder_deep_stacked_libri import (
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
INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS = True

MODEL_SAVE_DIR = "test_outputs/models/dvector_ae_greedy_layerwise_2-5-26"

# Greedy layer-wise architecture (one hidden layer per stage)
GREEDY_LAYER_HIDDEN_DIMS = [192, 128]
DROPOUT_RATE = 0.1
NORM_TYPE = "batchnorm"

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE_PRETRAIN = 1e-4
LEARNING_RATE_FINETUNE = 5e-5
NUM_EPOCHS_PRETRAIN = 60
NUM_EPOCHS_FINETUNE = 120
VALIDATION_SPLIT = 0.05
TEST_SPLIT = 0.0
EARLY_STOPPING_PATIENCE = 12

# Mixed reconstruction loss: total = MSE_WEIGHT * MSE + COSINE_WEIGHT * (1 - cosine)
LOSS_MSE_WEIGHT = 1.0
LOSS_COSINE_WEIGHT = 0.0
LOSS_COSINE_EPS = 1e-8

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
EXTRACTION_CACHE_NAME = "greedy_layerwise_libri_babble_pairs"


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

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1, norm_type="batchnorm"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.norm_type = str(norm_type).lower()

        def _norm(dim):
            if self.norm_type == "batchnorm":
                return nn.BatchNorm1d(dim)
            if self.norm_type == "layernorm":
                return nn.LayerNorm(dim)
            return nn.Identity()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            _norm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return F.normalize(recon, p=2, dim=-1, eps=1e-12)

    def encode(self, x):
        return self.encoder(x)


class StackedDenoisingAE(nn.Module):
    """Stacked autoencoder assembled from pretrained shallow AEs."""

    def __init__(self, pretrained_layers):
        super().__init__()
        self.encoders = nn.ModuleList([copy.deepcopy(layer.encoder) for layer in pretrained_layers])
        self.decoders = nn.ModuleList([copy.deepcopy(layer.decoder) for layer in pretrained_layers])

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        for decoder in reversed(self.decoders):
            out = decoder(out)
        return F.normalize(out, p=2, dim=-1, eps=1e-12)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_loss(pred, target):
    return 1.0 - F.cosine_similarity(pred, target, dim=1, eps=LOSS_COSINE_EPS).mean()


def _train_epoch(model, loader, optimizer, device):
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
        loss = (LOSS_MSE_WEIGHT * mse) + (LOSS_COSINE_WEIGHT * cos)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_cos += cos.item()

    n_batches = max(1, len(loader))
    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches


def _eval_epoch(model, loader, device):
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
            loss = (LOSS_MSE_WEIGHT * mse) + (LOSS_COSINE_WEIGHT * cos)

            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()

    n_batches = max(1, len(loader))
    return total_loss / n_batches, total_mse / n_batches, total_cos / n_batches


def _train_model(model, train_loader, val_loader, device, lr, num_epochs, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    patience = 0

    for epoch in range(num_epochs):
        train_loss, train_mse, train_cos = _train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mse, val_cos = _eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Train {train_loss:.6f} (mse {train_mse:.6f}, cos {train_cos:.6f}) | "
            f"Val {val_loss:.6f} (mse {val_mse:.6f}, cos {val_cos:.6f})"
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
        _train_model(layer, train_loader, val_loader, device, LEARNING_RATE_PRETRAIN, NUM_EPOCHS_PRETRAIN, layer_ckpt)
        pretrained_layers.append(layer)

        current_noisy = _encode_dataset(layer, current_noisy, BATCH_SIZE, device)
        current_clean = _encode_dataset(layer, current_clean, BATCH_SIZE, device)
        input_dim = hidden_dim

    return pretrained_layers


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

    print("=" * 90)
    print("GREEDY LAYER-WISE D-VECTOR DENOISING AUTOENCODER")
    print("=" * 90)

    pair_data = extract_noisy_clean_dvector_pairs_from_libri(
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
        checkpoint_dir=None,
    )

    noisy_dvectors = pair_data["noisy_dvectors"]
    clean_dvectors = pair_data["clean_dvectors"]
    input_dim = int(noisy_dvectors.shape[1])

    print(f"\n[data] Pairs: {len(noisy_dvectors)} | Dim: {input_dim}")

    pretrained_layers = greedy_pretrain_layers(noisy_dvectors, clean_dvectors, input_dim, DEVICE, save_dir)

    print("\n[stack] Assembling full stacked autoencoder")
    stacked_model = StackedDenoisingAE(pretrained_layers).to(DEVICE)

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
    )

    final_model_path = save_dir / "final_model.pth"
    torch.save(stacked_model.state_dict(), final_model_path)

    config = {
        "model_type": "greedy_layerwise_stacked",
        "input_dim": input_dim,
        "greedy_hidden_dims": list(GREEDY_LAYER_HIDDEN_DIMS),
        "dropout_rate": float(DROPOUT_RATE),
        "norm_type": str(NORM_TYPE),
        "loss_mse_weight": float(LOSS_MSE_WEIGHT),
        "loss_cosine_weight": float(LOSS_COSINE_WEIGHT),
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
        f.write(f"Dropout rate: {DROPOUT_RATE}\n")
        f.write(f"Norm type: {NORM_TYPE}\n")
        f.write(f"Libri root: {librispeech_root}\n")
        f.write(f"Libri subsets: {LIBRISPEECH_SUBSETS}\n")
        f.write(f"MUSAN speech root: {musan_root}\n")
        f.write(f"SNR values (dB): {MUSAN_BABBLE_SNR_RANGE_DB}\n")
        f.write(f"Include noise-only silence-target pairs: {INCLUDE_NOISE_ONLY_SILENCE_TARGET_PAIRS}\n")
        f.write(f"Pretrain LR: {LEARNING_RATE_PRETRAIN}\n")
        f.write(f"Finetune LR: {LEARNING_RATE_FINETUNE}\n")
        f.write(f"Pretrain epochs: {NUM_EPOCHS_PRETRAIN}\n")
        f.write(f"Finetune epochs: {NUM_EPOCHS_FINETUNE}\n")
        f.write(f"Best val loss: {best_val:.6f}\n")

    print("\nDone.")
    print(f"  Layer checkpoints: {save_dir}")
    print(f"  Best finetune checkpoint: {finetune_ckpt}")
    print(f"  Final model: {final_model_path}")
    print(f"  Config: {config_path}")


if __name__ == "__main__":
    main()
