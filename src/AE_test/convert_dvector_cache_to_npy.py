#!/usr/bin/env python3
"""Convert legacy pickle d-vector caches into readable folder-based .npy caches."""

import argparse
import pickle
from pathlib import Path

from readable_dvector_cache import cache_entry_dir, save_cache_entry


DEFAULT_CACHE_DIRS = [
    "test_outputs/dvector_cache",
    "test_outputs/dvector_cache2",
]


def _infer_cache_name(cache_file):
    stem = cache_file.stem
    if "_" not in stem:
        raise ValueError(f"Could not infer cache name from {cache_file.name}")
    return stem.rsplit("_", 1)[0]


def _iter_legacy_cache_files(cache_dir):
    for cache_file in sorted(Path(cache_dir).rglob("*.pkl")):
        if cache_file.name == "checkpoint.pkl":
            continue
        yield cache_file


def _convert_cache_file(cache_file, output_root, force=False):
    with open(cache_file, "rb") as handle:
        payload = pickle.load(handle)

    if not (isinstance(payload, dict) and "config" in payload and "data" in payload):
        raise ValueError(f"Unsupported legacy cache payload in {cache_file}")

    cache_name = _infer_cache_name(cache_file)
    cache_version = int(payload.get("cache_version", 1))
    cache_config = payload["config"]
    cache_data = payload["data"]

    entry_dir = cache_entry_dir(output_root, cache_name, cache_config, cache_version=cache_version)
    if entry_dir.exists() and not force:
        return cache_name, entry_dir, "skipped"

    save_cache_entry(
        cache_dir=output_root,
        cache_name=cache_name,
        cache_config=cache_config,
        data=cache_data,
        cache_version=cache_version,
    )
    return cache_name, entry_dir, "converted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_dirs",
        nargs="*",
        default=DEFAULT_CACHE_DIRS,
        help="Legacy cache directories to scan for .pkl cache payloads.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional destination root. Defaults to converting in place inside each cache dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing readable cache folders.",
    )
    args = parser.parse_args()

    converted = 0
    skipped = 0
    failed = 0

    for cache_dir_arg in args.cache_dirs:
        cache_dir = Path(cache_dir_arg)
        output_root = Path(args.output_root) if args.output_root else cache_dir

        if not cache_dir.exists():
            print(f"[skip] Cache dir not found: {cache_dir}")
            continue

        print(f"[scan] {cache_dir}")
        for cache_file in _iter_legacy_cache_files(cache_dir):
            try:
                cache_name, entry_dir, status = _convert_cache_file(
                    cache_file=cache_file,
                    output_root=output_root,
                    force=args.force,
                )
                print(f"[{status}] {cache_name}: {cache_file} -> {entry_dir}")
                if status == "converted":
                    converted += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                print(f"[failed] {cache_file}: {exc}")

    print(
        f"\nDone. converted={converted} skipped={skipped} failed={failed}"
    )


if __name__ == "__main__":
    main()
