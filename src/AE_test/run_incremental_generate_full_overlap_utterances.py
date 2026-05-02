#!/usr/bin/env python3
"""
Run generate_full_overlap_utterances.py multiple times while gradually adding
single-speaker datasets:
- run 1:  train1
- run 2:  train1, train2
- ...
- run 25: train1 ... train25
"""

from pathlib import Path

import generate_full_overlap_utterances as gen

# How many train folders to include incrementally
MAX_TRAIN = 25

SPEAKER_ID = "8463"  # Must match the speaker used in the single-speaker datasets and overlap generation config

# Base path used by generate_full_overlap_utterances.py
# (kept relative to keep behavior consistent with the original script)
SPEAKER_BASE = f'../../data/speaker_{SPEAKER_ID}'

# Put each run in its own output folder
OUTPUT_BASE = f'test_outputs/data/{SPEAKER_ID}/'

def main():
    for n in range(1, MAX_TRAIN + 1):
        datasets = [f'{SPEAKER_BASE}/train{i}' for i in range(1, n + 1)]

        # Update config in the imported generator module
        gen.SINGLE_SPEAKER_DATASETS = datasets
        gen.OUTPUT_DIR = f'{OUTPUT_BASE}/{SPEAKER_ID}_{4*n}pct_{3*n}utt-50spk+300Dev_5s_100pctmainspk_100pctAmp_2000'

        print('\n' + '=' * 80)
        print(f'RUN {n}/{MAX_TRAIN}')
        print(f'SINGLE_SPEAKER_DATASETS: train1 -> train{n}')
        print(f'OUTPUT_DIR: {gen.OUTPUT_DIR}')
        print('=' * 80)

        # Optional: keep run count deterministic and compact
        # gen.TOTAL_SAMPLES = 0  # use all main utterances

        # Execute one generation run
        gen.generate_dataset()


if __name__ == '__main__':
    main()
