#!/usr/bin/env python3
"""
Run train_dvector_autoencoder_infonce.py incrementally with finetuning.

For run n:
- SINGLE_SPEAKER_DATASETS uses train1..trainn
- OVERLAP_SAMPLES_DIRS points to the matching overlap dataset
- MODEL_SAVE_DIR changes per run

The InfoNCE script appends its own MODEL_SAVE_DIR_SUFFIX automatically.
"""

import train_dvector_autoencoder_infonce as infonce


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
SPEAKER_ID = "908"
START_TRAIN = 1
MAX_TRAIN = 25

# Must match your overlap generation runner output naming
OVERLAP_BASE = f"test_outputs/data/{SPEAKER_ID}"

# Single-speaker chunks base
SINGLE_SPEAKER_BASE = f"../../data/speaker_{SPEAKER_ID}"

# Where each incremental run will be saved (InfoNCE suffix is appended in infonce.main)
MODEL_BASE = f"test_outputs/models/{SPEAKER_ID}_wOV_finetune_asyminfoNCE_v4"

# Initial pretrained model used for all runs
# (set to None to train each run from scratch)
INITIAL_PRETRAINED_MODEL_PATH = infonce.ae.PRETRAINED_MODEL_PATH


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    # for n in range(START_TRAIN, MAX_TRAIN + 1):
    for n in [1,2,3,4,5,10,15,20,25]:  # Optional: only run specific increments
        # 1) train1..trainn for clean target data
        infonce.ae.SINGLE_SPEAKER_DATASETS = [
            f"{SINGLE_SPEAKER_BASE}/train{i}" for i in range(1, n + 1)
        ]

        # 2) matching overlap data directory (generated with same incremental scheme)
        overlap_dir = (
            f"{OVERLAP_BASE}/{SPEAKER_ID}_{4*n}pct_{3*n}utt-50spk+300Dev_5s_"
            f"100pctmainspk_100pctAmp_2000"
        )
        infonce.ae.OVERLAP_SAMPLES_DIRS = [overlap_dir]

        # 3) unique model output directory for this run
        model_dir = (
            f"{MODEL_BASE}/dvector_ae-{SPEAKER_ID}_{4*n}pct_{3*n}utt-50spk+300Dev_5s_"
            f"100pctmainspk_100pctAmp-{infonce.ae.TRAINING_SCHEME}_pretrain-2000_1e-5_100ep"
        )
        infonce.ae.MODEL_SAVE_DIR = model_dir

        # 4) finetune from the configured checkpoint
        infonce.ae.PRETRAINED_MODEL_PATH = INITIAL_PRETRAINED_MODEL_PATH

        print("\n" + "=" * 90)
        print(f"INFONCE FINETUNE RUN {n}/{MAX_TRAIN}")
        print(f"SINGLE_SPEAKER_DATASETS: train1 -> train{n}")
        print(f"OVERLAP_SAMPLES_DIRS: {infonce.ae.OVERLAP_SAMPLES_DIRS}")
        print(f"MODEL_SAVE_DIR (base): {infonce.ae.MODEL_SAVE_DIR}")
        print(f"MODEL_SAVE_SUFFIX: {infonce.MODEL_SAVE_DIR_SUFFIX}")
        print(f"PRETRAINED_MODEL_PATH: {infonce.ae.PRETRAINED_MODEL_PATH}")
        print(f"INFO_NCE_WEIGHT: {infonce.INFO_NCE_WEIGHT}")
        print(f"INFO_NCE_TEMPERATURE: {infonce.INFO_NCE_TEMPERATURE}")
        print("=" * 90)

        # Run one full training/finetuning cycle
        infonce.main()


if __name__ == "__main__":
    main()
