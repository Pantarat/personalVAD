#!/usr/bin/env python3
"""
Run train_dvector_autoencoder_finetune_greedy.py incrementally.

For run n:
- SINGLE_SPEAKER_DATASETS uses train1..trainn
- OVERLAP_TARGET_DIRS points to the matching overlap dataset
- MODEL_SAVE_DIR changes per run
"""

import train_dvector_autoencoder_finetune_greedy_v2 as ae


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
SPEAKER_ID ="6829"
START_TRAIN = 1
MAX_TRAIN = 25

# Must match your overlap generation runner output naming
OVERLAP_BASE = f"test_outputs/data/{SPEAKER_ID}"

# Single-speaker chunks base
SINGLE_SPEAKER_BASE = f"../../data/speaker_{SPEAKER_ID}"

# Where each finetuned model run will be saved
MODEL_BASE = f"test_outputs/models/greedy_finetunev7/layerwisePretrain/{SPEAKER_ID}"

# Initial pretrained model used for all runs
# (set to None to train each run from scratch)
INITIAL_PRETRAINED_MODEL_PATH = ae.PRETRAINED_MODEL_PATH


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    # for n in range(START_TRAIN, MAX_TRAIN + 1):
    for n in [1, 5, 10, 15, 20, 25]:  # Optional: only run specific increments
        # 1) train1..trainn for clean target data
        ae.SINGLE_SPEAKER_DATASETS = [
            f"{SINGLE_SPEAKER_BASE}/train{i}" for i in range(1, n + 1)
        ]

        # 2) matching target-overlap data directory
        overlap_dir = (
            f"{OVERLAP_BASE}/{SPEAKER_ID}_{4*n}pct_{3*n}utt-50spk+300Dev_5s_"
            f"100pctmainspk_100pctAmp_2000"
        )
        ae.OVERLAP_TARGET_DIRS = [overlap_dir]

        # 3) unique model output directory for this run
        model_dir = (
            f"{MODEL_BASE}/dvector_ae-{SPEAKER_ID}_{4*n}pct_{3*n}utt-50spk+300Dev_5s_"
            f"100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep"
        )
        ae.MODEL_SAVE_DIR = model_dir

        # 4) always finetune from the same initial checkpoint
        ae.PRETRAINED_MODEL_PATH = INITIAL_PRETRAINED_MODEL_PATH

        print("\n" + "=" * 90)
        print(f"GREEDY FINETUNE RUN {n}/{MAX_TRAIN}")
        print(f"SINGLE_SPEAKER_DATASETS: train1 -> train{n}")
        print(f"OVERLAP_TARGET_DIRS: {ae.OVERLAP_TARGET_DIRS}")
        print(f"MODEL_SAVE_DIR: {ae.MODEL_SAVE_DIR}")
        print(f"PRETRAINED_MODEL_PATH: {ae.PRETRAINED_MODEL_PATH}")
        print("=" * 90)

        # Run one full training/finetuning cycle
        ae.main()


if __name__ == "__main__":
    main()