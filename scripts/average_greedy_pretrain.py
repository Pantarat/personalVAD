#!/usr/bin/env python3
"""Average CSV metrics across multiple evaluation files.

Edit the CONFIG values below to choose input files and output location.
If FILES is empty, the script auto-discovers CSVs in INPUT_DIR and skips
the configured output file.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


# =============================== CONFIG ===============================

INPUT_DIR = "model_evaluation_results/greedy_finetunev5_tt/direct_finetune/grouped_by_percent/NO_AE/"
# FILES = [
#     'test_dvector_ae-61_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
#     'test_dvector_ae-121_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
#     'test_dvector_ae-174_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
#     'test_dvector_ae-260_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
#     'test_dvector_ae-908_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
#     'test_dvector_ae-1221_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
#     'test_dvector_ae-1462_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
#     'test_dvector_ae-6829_100pct_75utt-50spk_300Dev_5s_100pctmainspk_100pctAmp-greedy_finetune_pretrain-2000_1e-5_200ep.csv',
# ]
FILES = [
    '61_NOAE.csv',
    '121_NOAE.csv',
    '174_NOAE.csv',
    '260_NOAE.csv',
    '908_NOAE.csv',
    '1221_NOAE.csv',
    '1462_NOAE.csv',
    '6829_NOAE.csv',
]
FILE_GLOB = "*.csv"
OUTPUT_PATH = "model_evaluation_results/greedy_finetunev5_tt/direct_finetune/grouped_by_percent/NO_AE/average.csv"
GROUP_BY_COLUMNS = [
    "ae_model_name",
]
AUXILIARY_COLUMNS = [
    "dataset_name",
    "dataset_path",
    "n_samples",
    "ae_model_path",
]
AVERAGE_COLUMNS = [
    "mAP",
    "accuracy",
    "precision_TSS",
    "recall_TSS",
    "f1_TSS",
]
AUTO_INCLUDE_DATASET_KEYS = True
NORMALIZE_DATASET_KEYS = True
NORMALIZE_AE_GROUPS = True
REQUIRE_ALL = False
INCLUDE_COUNT = False

# =====================================================================


def format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {path}")
        return list(reader)


def normalize_dataset_value(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    normalized = re.sub(r"(^|/)\d+_", r"\1", normalized)
    normalized = re.sub(r"_main\d+", "_main", normalized)
    normalized = re.sub(r"_\d{1,2}-\d{1,2}-\d{4}$", "", normalized)
    return normalized


def normalize_ae_group_value(value: str) -> str:
    normalized = value.strip()
    if normalized.upper() == "NO_AE":
        return "NO_AE"
    return "AE"


def build_group_columns(header: Sequence[str]) -> List[str]:
    group_columns = list(GROUP_BY_COLUMNS)

    if AUTO_INCLUDE_DATASET_KEYS and "dataset_name" in header:
        dataset_columns = ["dataset_name"]
        if "dataset_path" in header:
            dataset_columns.append("dataset_path")

        group_columns = dataset_columns + [
            column for column in group_columns if column not in dataset_columns
        ]

    missing = [column for column in group_columns if column not in header]
    if missing:
        raise ValueError(f"Missing group columns in header: {', '.join(missing)}")

    return group_columns


def get_group_key(row: Dict[str, str], group_columns: Sequence[str]) -> Tuple[str, ...]:
    values: List[str] = []
    for column in group_columns:
        value = row.get(column, "").strip()
        if NORMALIZE_DATASET_KEYS and column in {"dataset_name", "dataset_path"}:
            value = normalize_dataset_value(value)
        if NORMALIZE_AE_GROUPS and column == "ae_model_name":
            value = normalize_ae_group_value(value)
        values.append(value)
    return tuple(values)


def resolve_input_paths(input_dir: Path, output_path: Path) -> List[Path]:
    if FILES:
        return [input_dir / name for name in FILES]

    resolved_output = output_path.resolve()
    return sorted(
        path
        for path in input_dir.glob(FILE_GLOB)
        if path.is_file() and path.resolve() != resolved_output
    )


def main() -> int:
    input_dir = Path(INPUT_DIR)
    output_path = Path(OUTPUT_PATH)

    input_paths = resolve_input_paths(input_dir, output_path)
    if not input_paths:
        print("No input CSV files found. Update the CONFIG section.", file=sys.stderr)
        return 1

    missing_files = [str(path) for path in input_paths if not path.is_file()]
    if missing_files:
        print("Missing input files:", file=sys.stderr)
        for missing in missing_files:
            print(f"  {missing}", file=sys.stderr)
        return 1

    header: List[str] = []
    group_columns: List[str] = []
    numeric_columns: List[str] = []
    copied_columns: List[str] = []
    sums: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    copied_values: Dict[Tuple[str, ...], Dict[str, str]] = defaultdict(dict)

    global_groups: Set[Tuple[str, ...]] = set()
    file_groups_map: Dict[str, Set[Tuple[str, ...]]] = {}
    copy_conflicts: Dict[str, int] = defaultdict(int)

    for path in input_paths:
        rows = read_csv_rows(path)
        if not rows:
            print(f"No rows found in {path}", file=sys.stderr)
            return 1

        if not header:
            header = list(rows[0].keys())
            group_columns = build_group_columns(header)
            missing_metrics = [col for col in AVERAGE_COLUMNS if col not in header]
            if missing_metrics:
                print(
                    "Missing average columns in header: "
                    + ", ".join(missing_metrics),
                    file=sys.stderr,
                )
                return 1
            numeric_columns = [col for col in AVERAGE_COLUMNS if col in header]
            copied_columns = [
                col for col in AUXILIARY_COLUMNS if col in header and col not in group_columns
            ]
        else:
            if list(rows[0].keys()) != header:
                print(f"Header mismatch in {path}", file=sys.stderr)
                return 1

        file_groups: Set[Tuple[str, ...]] = set()
        for row in rows:
            group_key = get_group_key(row, group_columns)
            if not any(group_key):
                continue

            file_groups.add(group_key)
            global_groups.add(group_key)

            for column in copied_columns:
                value = row.get(column, "").strip()
                existing = copied_values[group_key].get(column, "")
                if not existing:
                    copied_values[group_key][column] = value
                elif value and existing != value:
                    copied_values[group_key][column] = ""
                    copy_conflicts[column] += 1

            counts[group_key] += 1
            for column in numeric_columns:
                value = row.get(column, "").strip()
                if value == "":
                    raise ValueError(f"Empty value for column '{column}' in {path}")
                try:
                    sums[group_key][column] += float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value '{value}' for column '{column}' in {path}"
                    ) from exc

        file_groups_map[str(path)] = file_groups

    if REQUIRE_ALL:
        for file_path, groups in file_groups_map.items():
            missing = global_groups - groups
            extra = groups - global_groups
            if missing or extra:
                print(f"Model mismatch in {file_path}", file=sys.stderr)
                if missing:
                    print(
                        "  Missing: "
                        + ", ".join("|".join(group) for group in sorted(missing)),
                        file=sys.stderr,
                    )
                if extra:
                    print(
                        "  Extra: "
                        + ", ".join("|".join(group) for group in sorted(extra)),
                        file=sys.stderr,
                    )
                return 1
    else:
        for file_path, groups in file_groups_map.items():
            missing = global_groups - groups
            if missing:
                print(
                    f"Warning: {file_path} missing {len(missing)} groups",
                    file=sys.stderr,
                )

    for column, conflict_count in sorted(copy_conflicts.items()):
        print(
            f"Warning: {conflict_count} conflicting values seen for '{column}'. "
            f"Leaving that column blank for affected rows.",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_header = group_columns + copied_columns
    output_header += numeric_columns
    if INCLUDE_COUNT:
        output_header.append("num_files")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_header)
        writer.writeheader()

        for group_key in sorted(global_groups):
            row: Dict[str, str] = dict(zip(group_columns, group_key))
            for column in copied_columns:
                row[column] = copied_values[group_key].get(column, "")
            for column in numeric_columns:
                count = counts[group_key]
                if count == 0:
                    row[column] = ""
                    continue
                row[column] = format_float(sums[group_key][column] / count)

            if INCLUDE_COUNT:
                row["num_files"] = str(counts[group_key])

            writer.writerow(row)

    print(f"Wrote averages for {len(global_groups)} groups to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
