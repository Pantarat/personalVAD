#!/usr/bin/env python3
"""
Fine-tune the d-vector autoencoder with reconstruction + InfoNCE contrastive loss.

This script reuses the full data preparation pipeline from train_dvector_autoencoder.py
and only overrides training/evaluation/save hooks to add a contrastive objective.

Usage:
    python train_dvector_autoencoder_infonce.py
"""

from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import train_dvector_autoencoder as ae


# ============================================================================
# CONTRASTIVE FINE-TUNING SETTINGS
# ============================================================================

# Keep MSE + cosine losses from the base script and add InfoNCE on top.
INFO_NCE_ENABLED = True
INFO_NCE_WEIGHT = 0.15
INFO_NCE_TEMPERATURE = 0.1
INFO_NCE_BIDIRECTIONAL = True
INFO_NCE_MIN_BATCH_SIZE = 2

# Asymmetric objective:
# - Keep target speaker manifold anchored (target reconstruction + distribution lock)
# - Push non-target samples away from frozen target anchor
ASYMMETRIC_OBJECTIVE_ENABLED = True
TARGET_RECON_WEIGHT = 1.0
TARGET_STAT_LOCK_WEIGHT = 0.015
TARGET_STAT_COV_WEIGHT = 0.005
NON_TARGET_REPEL_WEIGHT = 0.35
NON_TARGET_REPEL_MARGIN = 0.0
NON_TARGET_RECON_WEIGHT = 0.15
# Non-target reconstruction mix is intentionally separate from target reconstruction mix.
NON_TARGET_RECON_MSE_WEIGHT = 1.0
NON_TARGET_RECON_COS_WEIGHT = 0.0
REPEL_WARMUP_EPOCHS = 0
NON_TARGET_REPEL_HARD_TOPK_RATIO = 1.0
NON_TARGET_REPEL_SQUARED_HINGE = False
TARGET_OVERLAP_BOOST_WEIGHT = 0.3

# Save into a separate directory to avoid overwriting non-contrastive runs.
MODEL_SAVE_DIR_SUFFIX = "_infoNCE_asym23"


# Internal cache for exporting metrics into final config.
LAST_EVAL_INFONCE_LOSS = None
LAST_EVAL_TARGET_STAT_LOSS = None
LAST_EVAL_TARGET_OVERLAP_BOOST_LOSS = None
TARGET_REF_MEAN = None
TARGET_REF_COV = None


# Sample-kind IDs (must match DvectorPairDataset in train_dvector_autoencoder.py)
SAMPLE_KIND_TARGET_OVERLAP = 0
SAMPLE_KIND_TARGET_IDENTITY = 1
SAMPLE_KIND_NON_TARGET_OVERLAP = 2
SAMPLE_KIND_OTHER = 3


def _resolve_pretrained_dir(pretrained_model_path):
    """Resolve pretrained directory with fallback to AE script-relative path."""
    raw_path = Path(pretrained_model_path)
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.append(Path(ae.__file__).parent / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


def _align_architecture_with_pretrained():
    """
    Align architecture flags with pretrained config to avoid strict load mismatches.
    """
    if ae.PRETRAINED_MODEL_PATH is None:
        return

    pretrained_dir = _resolve_pretrained_dir(ae.PRETRAINED_MODEL_PATH)
    config_path = pretrained_dir / "config.pkl"

    if not config_path.exists():
        print(f"\n[InfoNCE] Pretrained config not found at: {config_path}")
        print("[InfoNCE] Using current architecture settings as-is")
        return

    with open(config_path, "rb") as f:
        pretrained_config = pickle.load(f)

    key_map = [
        ("HIDDEN_DIMS", "hidden_dims"),
        ("DROPOUT_RATE", "dropout_rate"),
        ("AE_NORM_TYPE", "norm_type"),
        ("AE_USE_RESIDUAL", "use_residual"),
        ("AE_RESIDUAL_SCALE_INIT", "residual_scale_init"),
    ]

    print("\n[InfoNCE] Aligning model architecture with pretrained checkpoint config")
    for module_attr, config_key in key_map:
        if config_key not in pretrained_config:
            continue

        old_value = getattr(ae, module_attr)
        new_value = pretrained_config[config_key]

        if old_value != new_value:
            setattr(ae, module_attr, new_value)
            print(f"  - {module_attr}: {old_value} -> {new_value}")


def _append_model_save_suffix():
    """Append output suffix once so contrastive runs are isolated."""
    if not MODEL_SAVE_DIR_SUFFIX:
        return

    current = str(ae.MODEL_SAVE_DIR)
    if not current.endswith(MODEL_SAVE_DIR_SUFFIX):
        ae.MODEL_SAVE_DIR = f"{current}{MODEL_SAVE_DIR_SUFFIX}"


def info_nce_loss(queries, positives, negatives, temperature=0.07, bidirectional=True, eps=1e-8):
    """
    Contrastive loss with explicit non-target negatives.

    Positive pair i is (queries[i], positives[i]).
    Negatives come only from the provided non-target pool, so target
    samples are never contrasted against other target samples.
    """
    batch_size = queries.size(0)

    if batch_size < INFO_NCE_MIN_BATCH_SIZE or negatives.size(0) == 0:
        return queries.new_tensor(0.0)

    q = F.normalize(queries, p=2, dim=1, eps=eps)
    p = F.normalize(positives, p=2, dim=1, eps=eps)
    n = F.normalize(negatives, p=2, dim=1, eps=eps)

    positive_logits = torch.sum(q * p, dim=1, keepdim=True) / temperature
    negative_logits = torch.matmul(q, n.transpose(0, 1)) / temperature
    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=queries.device)

    q_to_p = F.cross_entropy(logits, labels)
    if not bidirectional:
        return q_to_p

    positive_logits_rev = torch.sum(p * q, dim=1, keepdim=True) / temperature
    negative_logits_rev = torch.matmul(p, n.transpose(0, 1)) / temperature
    logits_rev = torch.cat([positive_logits_rev, negative_logits_rev], dim=1)
    p_to_q = F.cross_entropy(logits_rev, labels)
    return 0.5 * (q_to_p + p_to_q)


def _unpack_batch(batch, device):
    """Support legacy batches and sample-kind-aware batches with target mask."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 4:
        overlap, clean, is_target, sample_kind = batch
        target_mask = (is_target > 0.5).to(device)
        sample_kind = sample_kind.to(device)
    elif isinstance(batch, (tuple, list)) and len(batch) == 3:
        overlap, clean, is_target = batch
        target_mask = (is_target > 0.5).to(device)
        sample_kind = torch.where(
            target_mask,
            torch.full((overlap.size(0),), SAMPLE_KIND_TARGET_OVERLAP, dtype=torch.long, device=device),
            torch.full((overlap.size(0),), SAMPLE_KIND_OTHER, dtype=torch.long, device=device),
        )
    else:
        overlap, clean = batch
        target_mask = torch.ones(overlap.size(0), dtype=torch.bool, device=device)
        sample_kind = torch.full((overlap.size(0),), SAMPLE_KIND_TARGET_OVERLAP, dtype=torch.long, device=device)

    overlap = overlap.to(device)
    clean = clean.to(device)
    clean_norm = F.normalize(clean, p=2, dim=1, eps=ae.LOSS_COSINE_EPS)
    return overlap, clean_norm, target_mask, sample_kind


def _mean_and_cov(vectors):
    """Return mean and covariance of normalized vectors."""
    if vectors.size(0) == 0:
        return None, None

    mean_vec = vectors.mean(dim=0)
    centered = vectors - mean_vec

    if vectors.size(0) < 2:
        dim = vectors.size(1)
        cov = torch.zeros((dim, dim), dtype=vectors.dtype, device=vectors.device)
    else:
        cov = centered.transpose(0, 1).matmul(centered) / (vectors.size(0) - 1)

    return mean_vec, cov


def _compute_target_reference_stats(train_loader, device):
    """
    Freeze a target anchor from clean target vectors in the training set.
    This anchor is later used to lock target distribution and repel non-targets.
    """
    collected = []

    with torch.no_grad():
        for batch in train_loader:
            _, clean_norm, target_mask, _ = _unpack_batch(batch, device)
            if target_mask.any():
                collected.append(clean_norm[target_mask])

    if len(collected) == 0:
        return None, None

    target_vectors = torch.cat(collected, dim=0)
    mean_vec, cov = _mean_and_cov(target_vectors)
    return mean_vec.detach(), cov.detach()


def _combined_losses(reconstructed, clean_norm, target_mask, sample_kind, target_ref_mean=None, target_ref_cov=None, repel_scale=1.0):
    """Compute asymmetric losses: target lock + non-target repulsion."""
    if target_mask.dtype != torch.bool:
        target_mask = target_mask > 0.5

    if target_mask.numel() != reconstructed.size(0):
        target_mask = torch.ones(reconstructed.size(0), dtype=torch.bool, device=reconstructed.device)

    non_target_mask = ~target_mask
    target_overlap_mask = target_mask & (sample_kind == SAMPLE_KIND_TARGET_OVERLAP)

    target_mse_loss = reconstructed.new_tensor(0.0)
    target_cosine_loss = reconstructed.new_tensor(0.0)
    target_recon_cos_sim_mean = reconstructed.new_tensor(0.0)
    infonce_loss = reconstructed.new_tensor(0.0)

    if target_mask.any():
        recon_target = reconstructed[target_mask]
        clean_target = clean_norm[target_mask]

        target_mse_loss = F.mse_loss(recon_target, clean_target)
        target_recon_cos_sim_mean = F.cosine_similarity(
            recon_target,
            clean_target,
            dim=1,
            eps=ae.LOSS_COSINE_EPS,
        ).mean()
        target_cosine_loss = 1.0 - target_recon_cos_sim_mean

        if INFO_NCE_ENABLED and recon_target.size(0) >= INFO_NCE_MIN_BATCH_SIZE and non_target_mask.any():
            non_target_clean = clean_norm[non_target_mask]
            non_target_recon = reconstructed[non_target_mask]
            contrast_negatives = torch.cat([non_target_clean, non_target_recon], dim=0)
            infonce_loss = info_nce_loss(
                recon_target,
                clean_target,
                contrast_negatives,
                temperature=INFO_NCE_TEMPERATURE,
                bidirectional=INFO_NCE_BIDIRECTIONAL,
                eps=ae.LOSS_COSINE_EPS,
            )

    target_stat_mean_loss = reconstructed.new_tensor(0.0)
    target_stat_cov_loss = reconstructed.new_tensor(0.0)
    target_stat_lock_loss = reconstructed.new_tensor(0.0)

    if ASYMMETRIC_OBJECTIVE_ENABLED and target_ref_mean is not None and target_mask.any():
        recon_target = reconstructed[target_mask]
        batch_mean, batch_cov = _mean_and_cov(recon_target)
        target_stat_mean_loss = F.mse_loss(batch_mean, target_ref_mean)
        if target_ref_cov is not None and recon_target.size(0) >= 2:
            target_stat_cov_loss = F.mse_loss(batch_cov, target_ref_cov)
        target_stat_lock_loss = target_stat_mean_loss + (TARGET_STAT_COV_WEIGHT * target_stat_cov_loss)

    non_target_repel_loss = reconstructed.new_tensor(0.0)
    non_target_anchor_cos_mean = reconstructed.new_tensor(0.0)
    non_target_recon_cos_sim_mean = reconstructed.new_tensor(0.0)
    if ASYMMETRIC_OBJECTIVE_ENABLED and target_ref_mean is not None and non_target_mask.any():
        anchor = F.normalize(target_ref_mean.unsqueeze(0), p=2, dim=1, eps=ae.LOSS_COSINE_EPS)
        non_target_vectors = F.normalize(reconstructed[non_target_mask], p=2, dim=1, eps=ae.LOSS_COSINE_EPS)
        cosine_to_anchor = torch.matmul(non_target_vectors, anchor.transpose(0, 1)).squeeze(1)
        non_target_anchor_cos_mean = cosine_to_anchor.mean()
        hinge = F.relu(cosine_to_anchor - NON_TARGET_REPEL_MARGIN)

        if 0.0 < NON_TARGET_REPEL_HARD_TOPK_RATIO < 1.0 and hinge.numel() > 1:
            k = max(1, int(np.ceil(float(hinge.numel()) * NON_TARGET_REPEL_HARD_TOPK_RATIO)))
            hinge = torch.topk(hinge, k=k, largest=True).values

        if NON_TARGET_REPEL_SQUARED_HINGE:
            hinge = hinge.pow(2)

        non_target_repel_loss = hinge.mean()

    non_target_recon_loss = reconstructed.new_tensor(0.0)
    non_target_recon_mse_loss = reconstructed.new_tensor(0.0)
    non_target_recon_cosine_loss = reconstructed.new_tensor(0.0)
    if non_target_mask.any():
        non_target_recon_cos_sim_mean = F.cosine_similarity(
            reconstructed[non_target_mask],
            clean_norm[non_target_mask],
            dim=1,
            eps=ae.LOSS_COSINE_EPS,
        ).mean()
        non_target_recon_mse_loss = F.mse_loss(
            reconstructed[non_target_mask],
            clean_norm[non_target_mask],
        )
        non_target_recon_cosine_loss = 1.0 - non_target_recon_cos_sim_mean

    if NON_TARGET_RECON_WEIGHT > 0.0 and non_target_mask.any():
        non_target_recon_loss = (
            (NON_TARGET_RECON_MSE_WEIGHT * non_target_recon_mse_loss)
            + (NON_TARGET_RECON_COS_WEIGHT * non_target_recon_cosine_loss)
        )

    target_recon_loss = (ae.LOSS_MSE_WEIGHT * target_mse_loss) + (ae.LOSS_COSINE_WEIGHT * target_cosine_loss)
    target_overlap_boost_loss = reconstructed.new_tensor(0.0)
    if TARGET_OVERLAP_BOOST_WEIGHT > 0.0 and target_overlap_mask.any():
        recon_overlap = reconstructed[target_overlap_mask]
        clean_overlap = clean_norm[target_overlap_mask]
        overlap_mse = F.mse_loss(recon_overlap, clean_overlap)
        overlap_cos = 1.0 - F.cosine_similarity(
            recon_overlap,
            clean_overlap,
            dim=1,
            eps=ae.LOSS_COSINE_EPS,
        ).mean()
        target_overlap_boost_loss = (ae.LOSS_MSE_WEIGHT * overlap_mse) + (ae.LOSS_COSINE_WEIGHT * overlap_cos)

    if ASYMMETRIC_OBJECTIVE_ENABLED:
        total_loss = (
            (TARGET_RECON_WEIGHT * target_recon_loss)
            + (INFO_NCE_WEIGHT * infonce_loss)
            + (TARGET_STAT_LOCK_WEIGHT * target_stat_lock_loss)
            + ((NON_TARGET_REPEL_WEIGHT * repel_scale) * non_target_repel_loss)
            + (NON_TARGET_RECON_WEIGHT * non_target_recon_loss)
            + (TARGET_OVERLAP_BOOST_WEIGHT * target_overlap_boost_loss)
        )
    else:
        # Fallback to symmetric objective if needed.
        full_mse = F.mse_loss(reconstructed, clean_norm)
        full_cos = 1.0 - F.cosine_similarity(
            reconstructed,
            clean_norm,
            dim=1,
            eps=ae.LOSS_COSINE_EPS,
        ).mean()
        total_loss = (
            (ae.LOSS_MSE_WEIGHT * full_mse)
            + (ae.LOSS_COSINE_WEIGHT * full_cos)
            + (INFO_NCE_WEIGHT * infonce_loss)
        )

    return {
        "total_loss": total_loss,
        "target_mse_loss": target_mse_loss,
        "target_cosine_loss": target_cosine_loss,
        "target_recon_cos_sim_mean": target_recon_cos_sim_mean,
        "infonce_loss": infonce_loss,
        "target_stat_lock_loss": target_stat_lock_loss,
        "target_stat_mean_loss": target_stat_mean_loss,
        "target_stat_cov_loss": target_stat_cov_loss,
        "non_target_repel_loss": non_target_repel_loss,
        "non_target_recon_loss": non_target_recon_loss,
        "non_target_recon_mse_loss": non_target_recon_mse_loss,
        "non_target_recon_cosine_loss": non_target_recon_cosine_loss,
        "non_target_recon_cos_sim_mean": non_target_recon_cos_sim_mean,
        "non_target_anchor_cos_mean": non_target_anchor_cos_mean,
        "target_overlap_boost_loss": target_overlap_boost_loss,
        "n_target_overlap": int(target_overlap_mask.sum().item()),
        "n_target": int(target_mask.sum().item()),
        "n_non_target": int(non_target_mask.sum().item()),
    }


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir):
    """Train using base reconstruction losses + InfoNCE contrastive loss."""
    global TARGET_REF_MEAN, TARGET_REF_COV

    print("\n[InfoNCE] Training autoencoder with contrastive objective")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"  Loss weights -> MSE: {ae.LOSS_MSE_WEIGHT}, Cosine: {ae.LOSS_COSINE_WEIGHT}, InfoNCE: {INFO_NCE_WEIGHT}")
    print(f"  InfoNCE temperature: {INFO_NCE_TEMPERATURE}")
    print(f"  InfoNCE bidirectional: {INFO_NCE_BIDIRECTIONAL}")
    target_pull = TARGET_RECON_WEIGHT + INFO_NCE_WEIGHT + TARGET_STAT_LOCK_WEIGHT + TARGET_OVERLAP_BOOST_WEIGHT
    non_target_push = NON_TARGET_REPEL_WEIGHT + NON_TARGET_RECON_WEIGHT
    push_pull_ratio = (non_target_push / target_pull) if target_pull > 0 else 0.0
    print(f"  Weight balance -> target_pull: {target_pull:.3f}, non_target_push: {non_target_push:.3f}, push/pull: {push_pull_ratio:.3f}")
    if ASYMMETRIC_OBJECTIVE_ENABLED:
        print("  Asymmetric objective: ENABLED")
        print(f"    Target stat lock weight: {TARGET_STAT_LOCK_WEIGHT} (cov factor: {TARGET_STAT_COV_WEIGHT})")
        print(f"    Non-target repel weight: {NON_TARGET_REPEL_WEIGHT} (margin: {NON_TARGET_REPEL_MARGIN})")
        print(f"    Non-target recon weight: {NON_TARGET_RECON_WEIGHT}")
        print(f"    Non-target recon mix: MSE {NON_TARGET_RECON_MSE_WEIGHT}, Cos {NON_TARGET_RECON_COS_WEIGHT}")
        print(f"    Target overlap boost weight: {TARGET_OVERLAP_BOOST_WEIGHT}")
        print(f"    Repel warmup epochs: {REPEL_WARMUP_EPOCHS}")
        print(f"    Repel hard top-k ratio: {NON_TARGET_REPEL_HARD_TOPK_RATIO}")
        print(f"    Repel squared hinge: {NON_TARGET_REPEL_SQUARED_HINGE}")
    else:
        print("  Asymmetric objective: DISABLED")

    model = model.to(device)

    TARGET_REF_MEAN, TARGET_REF_COV = _compute_target_reference_stats(train_loader, device)
    if TARGET_REF_MEAN is None:
        print("\n[InfoNCE] Warning: no target samples found in train loader; target-stat lock and non-target repulsion are disabled")
    else:
        print(f"\n[InfoNCE] Target reference anchor prepared (dim={TARGET_REF_MEAN.shape[0]})")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    history = {
        "train_loss": [],
        "train_mse_loss": [],
        "train_cosine_loss": [],
        "train_infonce_loss": [],
        "train_target_stat_loss": [],
        "train_non_target_repel_loss": [],
        "train_non_target_recon_loss": [],
        "train_target_recon_cos_sim": [],
        "train_non_target_recon_cos_sim": [],
        "train_non_target_anchor_cos_mean": [],
        "train_target_overlap_boost_loss": [],
        "train_target_overlap_fraction": [],
        "train_target_fraction": [],
        "val_loss": [],
        "val_mse_loss": [],
        "val_cosine_loss": [],
        "val_infonce_loss": [],
        "val_target_stat_loss": [],
        "val_non_target_repel_loss": [],
        "val_non_target_recon_loss": [],
        "val_target_recon_cos_sim": [],
        "val_non_target_recon_cos_sim": [],
        "val_non_target_anchor_cos_mean": [],
        "val_target_overlap_boost_loss": [],
        "val_target_overlap_fraction": [],
        "val_target_fraction": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_cos = 0.0
        train_infonce = 0.0
        train_target_stat = 0.0
        train_non_target_repel = 0.0
        train_non_target_recon = 0.0
        train_target_recon_cos_sim = 0.0
        train_non_target_recon_cos_sim = 0.0
        train_non_target_anchor_cos = 0.0
        train_target_overlap_boost = 0.0
        train_target_overlap_count = 0
        train_target_count = 0
        train_total_count = 0

        if REPEL_WARMUP_EPOCHS > 0:
            repel_scale = min(1.0, float(epoch + 1) / float(REPEL_WARMUP_EPOCHS))
        else:
            repel_scale = 1.0

        for batch in train_loader:
            overlap, clean_norm, target_mask, sample_kind = _unpack_batch(batch, device)

            optimizer.zero_grad()
            reconstructed = model(overlap)

            losses = _combined_losses(
                reconstructed,
                clean_norm,
                target_mask,
                sample_kind,
                target_ref_mean=TARGET_REF_MEAN,
                target_ref_cov=TARGET_REF_COV,
                repel_scale=repel_scale,
            )

            losses["total_loss"].backward()
            optimizer.step()

            train_loss += losses["total_loss"].item()
            train_mse += losses["target_mse_loss"].item()
            train_cos += losses["target_cosine_loss"].item()
            train_infonce += losses["infonce_loss"].item()
            train_target_stat += losses["target_stat_lock_loss"].item()
            train_non_target_repel += losses["non_target_repel_loss"].item()
            train_non_target_recon += losses["non_target_recon_loss"].item()
            train_target_recon_cos_sim += losses["target_recon_cos_sim_mean"].item()
            train_non_target_recon_cos_sim += losses["non_target_recon_cos_sim_mean"].item()
            train_non_target_anchor_cos += losses["non_target_anchor_cos_mean"].item()
            train_target_overlap_boost += losses["target_overlap_boost_loss"].item()
            train_target_overlap_count += losses["n_target_overlap"]
            train_target_count += losses["n_target"]
            train_total_count += losses["n_target"] + losses["n_non_target"]

        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_cos /= len(train_loader)
        train_infonce /= len(train_loader)
        train_target_stat /= len(train_loader)
        train_non_target_repel /= len(train_loader)
        train_non_target_recon /= len(train_loader)
        train_target_recon_cos_sim /= len(train_loader)
        train_non_target_recon_cos_sim /= len(train_loader)
        train_non_target_anchor_cos /= len(train_loader)
        train_target_overlap_boost /= len(train_loader)
        train_target_fraction = (float(train_target_count) / float(train_total_count)) if train_total_count > 0 else 0.0
        train_target_overlap_fraction = (float(train_target_overlap_count) / float(train_total_count)) if train_total_count > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_cos = 0.0
        val_infonce = 0.0
        val_target_stat = 0.0
        val_non_target_repel = 0.0
        val_non_target_recon = 0.0
        val_target_recon_cos_sim = 0.0
        val_non_target_recon_cos_sim = 0.0
        val_non_target_anchor_cos = 0.0
        val_target_overlap_boost = 0.0
        val_target_overlap_count = 0
        val_target_count = 0
        val_total_count = 0

        with torch.no_grad():
            for batch in val_loader:
                overlap, clean_norm, target_mask, sample_kind = _unpack_batch(batch, device)

                reconstructed = model(overlap)

                losses = _combined_losses(
                    reconstructed,
                    clean_norm,
                    target_mask,
                    sample_kind,
                    target_ref_mean=TARGET_REF_MEAN,
                    target_ref_cov=TARGET_REF_COV,
                    repel_scale=1.0,
                )

                val_loss += losses["total_loss"].item()
                val_mse += losses["target_mse_loss"].item()
                val_cos += losses["target_cosine_loss"].item()
                val_infonce += losses["infonce_loss"].item()
                val_target_stat += losses["target_stat_lock_loss"].item()
                val_non_target_repel += losses["non_target_repel_loss"].item()
                val_non_target_recon += losses["non_target_recon_loss"].item()
                val_target_recon_cos_sim += losses["target_recon_cos_sim_mean"].item()
                val_non_target_recon_cos_sim += losses["non_target_recon_cos_sim_mean"].item()
                val_non_target_anchor_cos += losses["non_target_anchor_cos_mean"].item()
                val_target_overlap_boost += losses["target_overlap_boost_loss"].item()
                val_target_overlap_count += losses["n_target_overlap"]
                val_target_count += losses["n_target"]
                val_total_count += losses["n_target"] + losses["n_non_target"]

        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_cos /= len(val_loader)
        val_infonce /= len(val_loader)
        val_target_stat /= len(val_loader)
        val_non_target_repel /= len(val_loader)
        val_non_target_recon /= len(val_loader)
        val_target_recon_cos_sim /= len(val_loader)
        val_non_target_recon_cos_sim /= len(val_loader)
        val_non_target_anchor_cos /= len(val_loader)
        val_target_overlap_boost /= len(val_loader)
        val_target_fraction = (float(val_target_count) / float(val_total_count)) if val_total_count > 0 else 0.0
        val_target_overlap_fraction = (float(val_target_overlap_count) / float(val_total_count)) if val_total_count > 0 else 0.0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_mse_loss"].append(train_mse)
        history["train_cosine_loss"].append(train_cos)
        history["train_infonce_loss"].append(train_infonce)
        history["train_target_stat_loss"].append(train_target_stat)
        history["train_non_target_repel_loss"].append(train_non_target_repel)
        history["train_non_target_recon_loss"].append(train_non_target_recon)
        history["train_target_recon_cos_sim"].append(train_target_recon_cos_sim)
        history["train_non_target_recon_cos_sim"].append(train_non_target_recon_cos_sim)
        history["train_non_target_anchor_cos_mean"].append(train_non_target_anchor_cos)
        history["train_target_overlap_boost_loss"].append(train_target_overlap_boost)
        history["train_target_overlap_fraction"].append(train_target_overlap_fraction)
        history["train_target_fraction"].append(train_target_fraction)
        history["val_loss"].append(val_loss)
        history["val_mse_loss"].append(val_mse)
        history["val_cosine_loss"].append(val_cos)
        history["val_infonce_loss"].append(val_infonce)
        history["val_target_stat_loss"].append(val_target_stat)
        history["val_non_target_repel_loss"].append(val_non_target_repel)
        history["val_non_target_recon_loss"].append(val_non_target_recon)
        history["val_target_recon_cos_sim"].append(val_target_recon_cos_sim)
        history["val_non_target_recon_cos_sim"].append(val_non_target_recon_cos_sim)
        history["val_non_target_anchor_cos_mean"].append(val_non_target_anchor_cos)
        history["val_target_overlap_boost_loss"].append(val_target_overlap_boost)
        history["val_target_overlap_fraction"].append(val_target_overlap_fraction)
        history["val_target_fraction"].append(val_target_fraction)
        history["learning_rate"].append(current_lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch + 1}/{num_epochs}] - "
                f"Train Mixed: {train_loss:.6f} (MSE {train_mse:.6f}, Cos {train_cos:.6f}, NCE {train_infonce:.6f}, Stat {train_target_stat:.6f}, Repel {train_non_target_repel:.6f}, OvlpBoost {train_target_overlap_boost:.6f}, TrecCos {train_target_recon_cos_sim:.4f}, NTrecCos {train_non_target_recon_cos_sim:.4f}, NTanchorCos {train_non_target_anchor_cos:.4f}, T% {100.0 * train_target_fraction:.1f}, TO% {100.0 * train_target_overlap_fraction:.1f}), "
                f"Val Mixed: {val_loss:.6f} (MSE {val_mse:.6f}, Cos {val_cos:.6f}, NCE {val_infonce:.6f}, Stat {val_target_stat:.6f}, Repel {val_non_target_repel:.6f}, OvlpBoost {val_target_overlap_boost:.6f}, TrecCos {val_target_recon_cos_sim:.4f}, NTrecCos {val_non_target_recon_cos_sim:.4f}, NTanchorCos {val_non_target_anchor_cos:.4f}, T% {100.0 * val_target_fraction:.1f}, TO% {100.0 * val_target_overlap_fraction:.1f}), "
                f"LR: {current_lr:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_model_path = Path(save_dir) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mse_loss": train_mse,
                    "train_cosine_loss": train_cos,
                    "train_infonce_loss": train_infonce,
                    "train_target_stat_loss": train_target_stat,
                    "train_non_target_repel_loss": train_non_target_repel,
                    "train_non_target_recon_loss": train_non_target_recon,
                    "train_target_recon_cos_sim": train_target_recon_cos_sim,
                    "train_non_target_recon_cos_sim": train_non_target_recon_cos_sim,
                    "train_non_target_anchor_cos_mean": train_non_target_anchor_cos,
                    "train_target_overlap_boost_loss": train_target_overlap_boost,
                    "train_target_overlap_fraction": train_target_overlap_fraction,
                    "train_target_fraction": train_target_fraction,
                    "val_mse_loss": val_mse,
                    "val_cosine_loss": val_cos,
                    "val_infonce_loss": val_infonce,
                    "val_target_stat_loss": val_target_stat,
                    "val_non_target_repel_loss": val_non_target_repel,
                    "val_non_target_recon_loss": val_non_target_recon,
                    "val_target_recon_cos_sim": val_target_recon_cos_sim,
                    "val_non_target_recon_cos_sim": val_non_target_recon_cos_sim,
                    "val_non_target_anchor_cos_mean": val_non_target_anchor_cos,
                    "val_target_overlap_boost_loss": val_target_overlap_boost,
                    "val_target_overlap_fraction": val_target_overlap_fraction,
                    "val_target_fraction": val_target_fraction,
                    "loss_infonce_weight": INFO_NCE_WEIGHT,
                    "infonce_temperature": INFO_NCE_TEMPERATURE,
                    "infonce_bidirectional": INFO_NCE_BIDIRECTIONAL,
                    "asymmetric_objective_enabled": ASYMMETRIC_OBJECTIVE_ENABLED,
                    "target_recon_weight": TARGET_RECON_WEIGHT,
                    "target_stat_lock_weight": TARGET_STAT_LOCK_WEIGHT,
                    "target_stat_cov_weight": TARGET_STAT_COV_WEIGHT,
                    "non_target_repel_weight": NON_TARGET_REPEL_WEIGHT,
                    "non_target_repel_margin": NON_TARGET_REPEL_MARGIN,
                    "non_target_recon_weight": NON_TARGET_RECON_WEIGHT,
                    "non_target_recon_mse_weight": NON_TARGET_RECON_MSE_WEIGHT,
                    "non_target_recon_cos_weight": NON_TARGET_RECON_COS_WEIGHT,
                    "target_overlap_boost_weight": TARGET_OVERLAP_BOOST_WEIGHT,
                    "non_target_repel_hard_topk_ratio": NON_TARGET_REPEL_HARD_TOPK_RATIO,
                    "non_target_repel_squared_hinge": NON_TARGET_REPEL_SQUARED_HINGE,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= ae.EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch + 1}")
            break

    print("\n[InfoNCE] Training complete")
    print(f"  Best validation loss: {best_val_loss:.6f}")

    return history, best_val_loss


def evaluate_model(model, test_loader, device):
    """Evaluate with mixed reconstruction + InfoNCE metrics."""
    global LAST_EVAL_INFONCE_LOSS, LAST_EVAL_TARGET_STAT_LOSS, LAST_EVAL_TARGET_OVERLAP_BOOST_LOSS

    print("\n[InfoNCE] Evaluating model on test set")

    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0
    total_infonce = 0.0
    total_target_stat = 0.0
    total_non_target_repel = 0.0
    total_target_overlap_boost = 0.0

    all_overlap = []
    all_clean = []
    all_reconstructed = []
    all_target_masks = []

    with torch.no_grad():
        for batch in test_loader:
            overlap, clean_norm, target_mask, sample_kind = _unpack_batch(batch, device)

            reconstructed = model(overlap)
            losses = _combined_losses(
                reconstructed,
                clean_norm,
                target_mask,
                sample_kind,
                target_ref_mean=TARGET_REF_MEAN,
                target_ref_cov=TARGET_REF_COV,
                repel_scale=1.0,
            )

            total_loss += losses["total_loss"].item()
            total_mse += losses["target_mse_loss"].item()
            total_cos += losses["target_cosine_loss"].item()
            total_infonce += losses["infonce_loss"].item()
            total_target_stat += losses["target_stat_lock_loss"].item()
            total_non_target_repel += losses["non_target_repel_loss"].item()
            total_target_overlap_boost += losses["target_overlap_boost_loss"].item()

            all_overlap.append(overlap.cpu().numpy())
            all_clean.append(clean_norm.cpu().numpy())
            all_reconstructed.append(reconstructed.cpu().numpy())
            all_target_masks.append(target_mask.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    avg_mse = total_mse / len(test_loader)
    avg_cos = total_cos / len(test_loader)
    avg_infonce = total_infonce / len(test_loader)
    avg_target_stat = total_target_stat / len(test_loader)
    avg_non_target_repel = total_non_target_repel / len(test_loader)
    avg_target_overlap_boost = total_target_overlap_boost / len(test_loader)
    LAST_EVAL_INFONCE_LOSS = avg_infonce
    LAST_EVAL_TARGET_STAT_LOSS = avg_target_stat
    LAST_EVAL_TARGET_OVERLAP_BOOST_LOSS = avg_target_overlap_boost

    all_overlap = np.concatenate(all_overlap, axis=0)
    all_clean = np.concatenate(all_clean, axis=0)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    all_target_masks = np.concatenate(all_target_masks, axis=0).astype(bool)

    cosine_sims = []
    for i in range(len(all_clean)):
        clean_norm = all_clean[i] / np.linalg.norm(all_clean[i])
        recon_norm = all_reconstructed[i] / np.linalg.norm(all_reconstructed[i])
        cosine_sims.append(np.dot(clean_norm, recon_norm))

    cosine_sims = np.array(cosine_sims)
    target_cosine_sims = cosine_sims[all_target_masks]
    non_target_cosine_sims = cosine_sims[~all_target_masks]

    def _group_stats(values):
        if values.size == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    target_stats = _group_stats(target_cosine_sims)
    non_target_stats = _group_stats(non_target_cosine_sims)

    print("\n  Test Results:")
    print(f"    Mixed Loss: {avg_loss:.6f}")
    print(f"    MSE Loss: {avg_mse:.6f}")
    print(f"    Cosine Loss: {avg_cos:.6f}")
    print(f"    InfoNCE Loss: {avg_infonce:.6f}")
    print(f"    Target Stat Lock Loss: {avg_target_stat:.6f}")
    print(f"    Non-target Repel Loss: {avg_non_target_repel:.6f}")
    print(f"    Target Overlap Boost Loss: {avg_target_overlap_boost:.6f}")
    print(f"    Cosine Similarity: {cosine_sims.mean():.4f} +- {cosine_sims.std():.4f}")
    print(f"    Min Cosine Sim: {cosine_sims.min():.4f}")
    print(f"    Max Cosine Sim: {cosine_sims.max():.4f}")
    print(f"    Target Recon Cosine Sim: {target_stats['mean']:.4f} +- {target_stats['std']:.4f} (n={target_stats['count']})")
    print(f"    Target Recon Cosine Min/Max: {target_stats['min']:.4f} / {target_stats['max']:.4f}")
    print(f"    Non-target Recon Cosine Sim: {non_target_stats['mean']:.4f} +- {non_target_stats['std']:.4f} (n={non_target_stats['count']})")
    print(f"    Non-target Recon Cosine Min/Max: {non_target_stats['min']:.4f} / {non_target_stats['max']:.4f}")

    return {
        "mixed_loss": avg_loss,
        "mse_loss": avg_mse,
        "cosine_loss": avg_cos,
        "infonce_loss": avg_infonce,
        "target_stat_loss": avg_target_stat,
        "non_target_repel_loss": avg_non_target_repel,
        "target_overlap_boost_loss": avg_target_overlap_boost,
        "cosine_similarity": cosine_sims,
        "target_cosine_similarity": target_cosine_sims,
        "non_target_cosine_similarity": non_target_cosine_sims,
        "target_cosine_similarity_mean": target_stats["mean"],
        "target_cosine_similarity_std": target_stats["std"],
        "non_target_cosine_similarity_mean": non_target_stats["mean"],
        "non_target_cosine_similarity_std": non_target_stats["std"],
        "overlap": all_overlap,
        "clean": all_clean,
        "reconstructed": all_reconstructed,
    }


_ORIGINAL_SAVE_MODEL_AND_CONFIG = ae.save_model_and_config


def save_model_and_config(model, config, save_dir):
    """Augment config with InfoNCE metadata and delegate to base saver."""
    config = dict(config)
    config["info_nce_enabled"] = INFO_NCE_ENABLED
    config["loss_infonce_weight"] = INFO_NCE_WEIGHT if INFO_NCE_ENABLED else 0.0
    config["infonce_temperature"] = INFO_NCE_TEMPERATURE if INFO_NCE_ENABLED else None
    config["infonce_bidirectional"] = INFO_NCE_BIDIRECTIONAL if INFO_NCE_ENABLED else None
    config["asymmetric_objective_enabled"] = ASYMMETRIC_OBJECTIVE_ENABLED
    config["target_recon_weight"] = TARGET_RECON_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["target_stat_lock_weight"] = TARGET_STAT_LOCK_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["target_stat_cov_weight"] = TARGET_STAT_COV_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["non_target_repel_weight"] = NON_TARGET_REPEL_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["non_target_repel_margin"] = NON_TARGET_REPEL_MARGIN if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["non_target_recon_weight"] = NON_TARGET_RECON_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["non_target_recon_mse_weight"] = NON_TARGET_RECON_MSE_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["non_target_recon_cos_weight"] = NON_TARGET_RECON_COS_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["target_overlap_boost_weight"] = TARGET_OVERLAP_BOOST_WEIGHT if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["non_target_repel_hard_topk_ratio"] = NON_TARGET_REPEL_HARD_TOPK_RATIO if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["non_target_repel_squared_hinge"] = NON_TARGET_REPEL_SQUARED_HINGE if ASYMMETRIC_OBJECTIVE_ENABLED else None
    config["repel_warmup_epochs"] = REPEL_WARMUP_EPOCHS if ASYMMETRIC_OBJECTIVE_ENABLED else None
    if LAST_EVAL_INFONCE_LOSS is not None:
        config["test_infonce_loss"] = LAST_EVAL_INFONCE_LOSS
    if LAST_EVAL_TARGET_STAT_LOSS is not None:
        config["test_target_stat_loss"] = LAST_EVAL_TARGET_STAT_LOSS
    if LAST_EVAL_TARGET_OVERLAP_BOOST_LOSS is not None:
        config["test_target_overlap_boost_loss"] = LAST_EVAL_TARGET_OVERLAP_BOOST_LOSS

    _ORIGINAL_SAVE_MODEL_AND_CONFIG(model, config, save_dir)

    config_txt_path = Path(save_dir) / "config_summary.txt"
    with open(config_txt_path, "a") as f:
        f.write("\nCONTRASTIVE LEARNING\n")
        f.write("-" * 70 + "\n")
        f.write(f"InfoNCE enabled: {INFO_NCE_ENABLED}\n")
        f.write(f"InfoNCE weight: {INFO_NCE_WEIGHT}\n")
        f.write(f"InfoNCE temperature: {INFO_NCE_TEMPERATURE}\n")
        f.write(f"InfoNCE bidirectional: {INFO_NCE_BIDIRECTIONAL}\n")
        if LAST_EVAL_INFONCE_LOSS is not None:
            f.write(f"Test InfoNCE loss: {LAST_EVAL_INFONCE_LOSS:.6f}\n")
        f.write("\nASYMMETRIC OBJECTIVE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Enabled: {ASYMMETRIC_OBJECTIVE_ENABLED}\n")
        if ASYMMETRIC_OBJECTIVE_ENABLED:
            f.write(f"Target recon weight: {TARGET_RECON_WEIGHT}\n")
            f.write(f"Target stat lock weight: {TARGET_STAT_LOCK_WEIGHT}\n")
            f.write(f"Target stat cov weight: {TARGET_STAT_COV_WEIGHT}\n")
            f.write(f"Non-target repel weight: {NON_TARGET_REPEL_WEIGHT}\n")
            f.write(f"Non-target repel margin: {NON_TARGET_REPEL_MARGIN}\n")
            f.write(f"Non-target recon weight: {NON_TARGET_RECON_WEIGHT}\n")
            f.write(f"Non-target recon mse weight: {NON_TARGET_RECON_MSE_WEIGHT}\n")
            f.write(f"Non-target recon cos weight: {NON_TARGET_RECON_COS_WEIGHT}\n")
            f.write(f"Target overlap boost weight: {TARGET_OVERLAP_BOOST_WEIGHT}\n")
            f.write(f"Repel warmup epochs: {REPEL_WARMUP_EPOCHS}\n")
            f.write(f"Repel hard top-k ratio: {NON_TARGET_REPEL_HARD_TOPK_RATIO}\n")
            f.write(f"Repel squared hinge: {NON_TARGET_REPEL_SQUARED_HINGE}\n")
            if LAST_EVAL_TARGET_STAT_LOSS is not None:
                f.write(f"Test target stat loss: {LAST_EVAL_TARGET_STAT_LOSS:.6f}\n")
            if LAST_EVAL_TARGET_OVERLAP_BOOST_LOSS is not None:
                f.write(f"Test target overlap boost loss: {LAST_EVAL_TARGET_OVERLAP_BOOST_LOSS:.6f}\n")


def _patch_base_pipeline():
    """Swap base hooks with InfoNCE-aware versions."""
    ae.train_model = train_model
    ae.evaluate_model = evaluate_model
    ae.save_model_and_config = save_model_and_config


def main():
    _append_model_save_suffix()
    _align_architecture_with_pretrained()
    _patch_base_pipeline()

    print("\n" + "=" * 70)
    print("D-VECTOR AUTOENCODER FINETUNING WITH INFONCE")
    print("=" * 70)
    print(f"Model save dir: {ae.MODEL_SAVE_DIR}")
    print(f"Pretrained model path: {ae.PRETRAINED_MODEL_PATH}")
    print(f"Loss weights -> MSE: {ae.LOSS_MSE_WEIGHT}, Cosine: {ae.LOSS_COSINE_WEIGHT}, InfoNCE: {INFO_NCE_WEIGHT}")
    print(f"InfoNCE temperature: {INFO_NCE_TEMPERATURE}")
    print(
        f"Asymmetric objective: {ASYMMETRIC_OBJECTIVE_ENABLED} "
        f"(target-stat {TARGET_STAT_LOCK_WEIGHT}, non-target-repel {NON_TARGET_REPEL_WEIGHT}, overlap-boost {TARGET_OVERLAP_BOOST_WEIGHT}, margin {NON_TARGET_REPEL_MARGIN})"
    )
    print("Hyperparameter policy: global shared constants (no per-speaker overrides)")

    ae.main()


if __name__ == "__main__":
    main()
