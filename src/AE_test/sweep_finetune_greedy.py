#!/usr/bin/env python3
"""
Loss weight sweep for greedy d-vector autoencoder finetuning.

Runs all combinations of MSE/Cosine/Negative weights in step increments
where the total weight sum is exactly 1.0.
"""

import builtins
import time
from datetime import datetime
from pathlib import Path

import train_dvector_autoencoder_finetune_greedy as base

# Sweep settings
SWEEP_STEP = 0.1
SWEEP_SAVE_ROOT = "test_outputs/models/greedy_finetune_v4/260"
RESUME_FROM_LAST_COMPLETED = True
RESUME_MARKER_FILES = ("config.pkl", "final_model.pth")

# Optional per-run debug output isolation
OVERRIDE_DEBUG_PAIR_DIR = True
DEBUG_PAIR_DIR_NAME = "debug_pairs"


def _timestamp_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", None)
    flush = kwargs.get("flush", False)
    text = sep.join("" if arg is None else str(arg) for arg in args)

    lines = text.split("\n")
    for idx, line in enumerate(lines):
        line_end = end if idx == len(lines) - 1 else "\n"
        if line == "":
            builtins.print(f"[{timestamp}]", end=line_end, file=file, flush=flush)
        else:
            builtins.print(f"[{timestamp}] {line}", end=line_end, file=file, flush=flush)


print = _timestamp_print


def _resolve_dir(path_value, script_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = script_dir / path
    return path


def _generate_loss_weight_combinations(step):
    step = float(step)
    if step <= 0.0:
        raise ValueError("SWEEP_STEP must be > 0")
    total_units = int(round(1.0 / step))
    if abs((total_units * step) - 1.0) > 1e-6:
        raise ValueError("SWEEP_STEP must evenly divide 1.0")

    combos = []
    for cos_units in range(total_units + 1):
        neg_units = total_units - cos_units
        combos.append(
            (0, round(cos_units * step, 1), round(neg_units * step, 1))
    # for mse_units in range(total_units + 1):
    #     for cos_units in range(total_units + 1 - mse_units):
    #         neg_units = total_units - mse_units - cos_units
    #         combos.append(
    #             (round(mse_units * step, 1), round(cos_units * step, 1), round(neg_units * step, 1))
            )
    return combos


def _format_weight_tag(mse_w, cos_w, neg_w):
    tag = f"mse{mse_w:.1f}_cos{cos_w:.1f}_neg{neg_w:.1f}"
    return tag.replace(".", "p")


def _run_is_complete(run_dir):
    for marker in RESUME_MARKER_FILES:
        if (run_dir / marker).exists():
            return True
    return False


def _find_resume_start_index(sweep_root, combos):
    last_completed_idx = -1
    for idx, (mse_w, cos_w, neg_w) in enumerate(combos):
        run_tag = _format_weight_tag(mse_w, cos_w, neg_w)
        run_dir = sweep_root / run_tag
        if _run_is_complete(run_dir):
            last_completed_idx = idx
    return last_completed_idx + 1, last_completed_idx


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _run_finetune_with_weights(mse_w, cos_w, neg_w, run_dir):
    weight_sum = mse_w + cos_w + neg_w
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weight sum must be 1.0, got {weight_sum:.4f}")

    base.LOSS_MSE_WEIGHT = float(mse_w)
    base.LOSS_COSINE_WEIGHT = float(cos_w)
    base.NEGATIVE_CONTRASTIVE_WEIGHT = float(neg_w)
    base.MODEL_SAVE_DIR = str(run_dir)

    if OVERRIDE_DEBUG_PAIR_DIR:
        base.DEBUG_PAIR_DIR = str(run_dir / DEBUG_PAIR_DIR_NAME)

    base.main()


def main():
    script_dir = Path(__file__).parent

    print("=" * 90)
    print("GREEDY FINETUNE D-VECTOR AE LOSS WEIGHT SWEEP")
    print("=" * 90)

    sweep_root = _resolve_dir(SWEEP_SAVE_ROOT, script_dir)
    combos = _generate_loss_weight_combinations(SWEEP_STEP)
    print(f"\n[sweep] Total combos: {len(combos)} | Step: {SWEEP_STEP} | Combos: {combos}")

    start_idx = 0
    if RESUME_FROM_LAST_COMPLETED:
        start_idx, last_completed_idx = _find_resume_start_index(sweep_root, combos)
        if last_completed_idx >= 0:
            last_combo = combos[last_completed_idx]
            print(
                f"\n[sweep] Resuming after index {last_completed_idx} | "
                f"Last completed: {last_combo}"
            )
        if start_idx >= len(combos):
            print("\n[sweep] All combinations already completed.")
            return

    durations_sec = []
    total_runs = len(combos)

    for run_idx, (mse_w, cos_w, neg_w) in enumerate(combos[start_idx:], start=start_idx + 1):
        run_tag = _format_weight_tag(mse_w, cos_w, neg_w)
        run_dir = sweep_root / run_tag
        print(f"\n[sweep] Run {run_idx}/{len(combos)} | {run_tag}")

        run_start = time.time()
        _run_finetune_with_weights(mse_w, cos_w, neg_w, run_dir)
        run_elapsed = time.time() - run_start
        durations_sec.append(run_elapsed)

        avg_elapsed = sum(durations_sec) / len(durations_sec)
        remaining_runs = total_runs - run_idx
        eta_seconds = avg_elapsed * remaining_runs

        print(
            "\n[timing] "
            f"Run: {_format_duration(run_elapsed)} | "
            f"Avg: {_format_duration(avg_elapsed)} | "
            f"Remaining: {remaining_runs} | "
            f"ETA: {_format_duration(eta_seconds)}"
        )


if __name__ == "__main__":
    main()
