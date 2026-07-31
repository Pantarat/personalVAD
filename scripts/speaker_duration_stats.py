#!/usr/bin/env python3
from pathlib import Path

import soundfile as sf

SPEAKER_IDS = ["61", "121", "174", "260", "908", "1221", "1462", "6829"]


def find_audio_files(root: Path, speaker_id: str):
    for subset_dir in root.iterdir():
        if not subset_dir.is_dir():
            continue
        speaker_dir = subset_dir / speaker_id
        if not speaker_dir.exists():
            continue
        for path in speaker_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".wav", ".flac"}:
                yield path


def main():
    root = Path("data/LibriSpeech")
    all_durations = []

    for speaker_id in SPEAKER_IDS:
        durations = []
        for path in find_audio_files(root, speaker_id):
            try:
                durations.append(sf.info(str(path)).duration)
            except Exception:
                pass

        if not durations:
            print(f"{speaker_id}: no utterances found")
            continue

        total = sum(durations)
        avg = total / len(durations)
        all_durations.extend(durations)
        print(f"{speaker_id}: {len(durations)} utterances, avg duration {avg:.3f}s, total duration {total:.3f}s")

    if all_durations:
        overall_avg = sum(all_durations) / len(all_durations)
        overall_total = sum(all_durations)
        print(f"OVERALL: {len(all_durations)} utterances, avg duration {overall_avg:.3f}s, total duration {overall_total:.3f}s")


if __name__ == "__main__":
    main()
