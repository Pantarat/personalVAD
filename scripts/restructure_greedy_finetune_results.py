#!/usr/bin/env python3
"""Build tf-style percent/utterance folders from tt-style speaker folders."""

import re
import shutil
from pathlib import Path


# ============================== CONFIG ==============================

SOURCE_ROOT = Path("model_evaluation_results/greedy_finetunev5_ff/nonLayerwisePretrain/")
# Set OUTPUT_ROOT = SOURCE_ROOT if you want the mixed tf-style layout in place.
OUTPUT_ROOT = Path("model_evaluation_results/greedy_finetunev5_ff/nonLayerwisePretrain/grouped_by_percent/")

SPEAKER_DIR_PATTERN = re.compile(r"^\d+$")
PERCENT_UTT_PATTERN = re.compile(r"_(\d+pct_\d+utt)-")

# ==================================================================
def main():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing source folder: {SOURCE_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    created_buckets = set()

    speaker_dirs = []
    for path in SOURCE_ROOT.iterdir():
        if path.is_dir() and SPEAKER_DIR_PATTERN.fullmatch(path.name):
            speaker_dirs.append(path)

    for speaker_dir in sorted(speaker_dirs):
        speaker_id = speaker_dir.name

        for csv_path in sorted(speaker_dir.glob("*.csv")):
            if csv_path.name == "test_NO_AE.csv":
                bucket_name = "NO_AE"
                bucket_dir = OUTPUT_ROOT / bucket_name
                bucket_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_path, bucket_dir / (speaker_id + "_NOAE.csv"))
                created_buckets.add(bucket_name)
                continue

            match = PERCENT_UTT_PATTERN.search(csv_path.name)
            if not match:
                continue

            bucket_name = match.group(1)
            bucket_dir = OUTPUT_ROOT / bucket_name
            bucket_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, bucket_dir / csv_path.name)
            created_buckets.add(bucket_name)

    print(f"Source: {SOURCE_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Created/updated {len(created_buckets)} folders:")
    for bucket_name in sorted(created_buckets):
        print(f"  {bucket_name}")


if __name__ == "__main__":
    main()
