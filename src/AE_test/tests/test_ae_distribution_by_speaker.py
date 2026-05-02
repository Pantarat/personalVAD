#!/usr/bin/env python3
"""Simple spread check for one speaker before and after autoencoder.

This version recomputes d-vectors directly from LibriSpeech recordings
(it does NOT use precomputed dvectors.scp).

Edit settings below, then run from project root:
python src/AE_test/tests/test_ae_distribution_by_speaker.py
"""

import sys
import random
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
from resemblyzer import VoiceEncoder
from tqdm import tqdm


# Suppress FutureWarning triggered inside resemblyzer's librosa calls.
warnings.filterwarnings('ignore', category=FutureWarning, module=r'resemblyzer(\.|$)')


# Allow importing from src/AE_test
THIS_DIR = Path(__file__).resolve().parent
AE_TEST_DIR = THIS_DIR.parent
if str(AE_TEST_DIR) not in sys.path:
	sys.path.insert(0, str(AE_TEST_DIR))

from autoencoder_utils import load_autoencoder


# =====================
# SETTINGS (edit these)
# =====================
LIBRISPEECH_ROOT = 'data/LibriSpeech'
LIBRISPEECH_SUBSETS = [
	'dev-clean',
	'dev-other',
	'test-clean',
	'test-other',
	'train-clean-100',
	'train-clean-360',
	'train-other-500',
]
SPEAKER_ID = '84'
AE_MODEL_DIR = 'src/AE_test/test_outputs/dvector_ae_identity_1100x50Dev_noOV_1,6s_20-2-27'
USE_BOTTLENECK = False  # False = reconstruction, True = encoder bottleneck output
MAX_SPEAKER_VECTORS = None  # e.g. 5000 for quick test, or None for all
RANDOM_SEED = None  # e.g. 42 for reproducible random window start
WINDOW_SEC = 1.6
PAD_SHORT_AUDIO = True
MULTI_WINDOWS_PER_FILE = True  # True: use all non-overlapping 1.6s windows per file
BATCH_SIZE = 1024
FORCE_CPU = False


device = 'cpu' if FORCE_CPU else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# Load AE files
model_dir = Path(AE_MODEL_DIR)
config_path = model_dir / 'config.pkl'
final_model = model_dir / 'final_model.pth'
best_model = model_dir / 'best_model.pth'

if not config_path.exists():
	raise FileNotFoundError(f'Missing config file: {config_path}')

if final_model.exists():
	model_path = final_model
elif best_model.exists():
	model_path = best_model
else:
	raise FileNotFoundError(f'No model checkpoint found in: {model_dir}')


ae_model = load_autoencoder(str(model_path), str(config_path), device=device)


# Recompute d-vectors from recordings and keep only selected speaker
speaker_encoder = VoiceEncoder(device=torch.device(device))

speaker = str(SPEAKER_ID)
speaker_dvectors = []
window_refs = []
num_candidate_files = 0
num_processed_files = 0
num_too_short = 0
num_padded_short = 0
num_audio_errors = 0
num_missing_subset_dirs = 0
num_windows_used = 0
num_files_with_multiple_windows = 0

libri_root = Path(LIBRISPEECH_ROOT)
if not libri_root.exists():
	raise FileNotFoundError(f'LibriSpeech root not found: {libri_root}')

speaker_files = []
for subset in LIBRISPEECH_SUBSETS:
	subset_dir = libri_root / subset
	if not subset_dir.exists():
		num_missing_subset_dirs += 1
		continue

	speaker_dir = subset_dir / speaker
	if not speaker_dir.exists():
		continue

	files = sorted(speaker_dir.rglob('*.flac'))
	speaker_files.extend(files)

num_candidate_files = len(speaker_files)
if MAX_SPEAKER_VECTORS is not None and MAX_SPEAKER_VECTORS > 0:
	speaker_files = speaker_files[:MAX_SPEAKER_VECTORS]

if RANDOM_SEED is not None:
	random.seed(RANDOM_SEED)

window_samples = int(WINDOW_SEC * 16000)

for audio_path in tqdm(speaker_files, desc='Loading speaker audio and extracting d-vectors', unit='file'):
	if MAX_SPEAKER_VECTORS is not None and MAX_SPEAKER_VECTORS > 0 and len(speaker_dvectors) >= MAX_SPEAKER_VECTORS:
		break
	try:
		wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
		wav = np.asarray(wav, dtype=np.float32)
		window_segments = []
		window_infos = []

		if len(wav) < window_samples:
			if PAD_SHORT_AUDIO and len(wav) > 0:
				segment = np.pad(wav, (0, window_samples - len(wav)), mode='constant')
				window_segments.append(segment)
				window_infos.append((0.0, len(wav) / 16000.0))
				num_padded_short += 1
			else:
				num_too_short += 1
				continue
		else:
			if MULTI_WINDOWS_PER_FILE:
				n_full = len(wav) // window_samples
				for i in range(n_full):
					start_idx = i * window_samples
					end_idx = start_idx + window_samples
					window_segments.append(wav[start_idx:end_idx])
					window_infos.append((start_idx / 16000.0, end_idx / 16000.0))

				remainder = len(wav) - (n_full * window_samples)
				if PAD_SHORT_AUDIO and remainder > 0:
					tail_start = n_full * window_samples
					tail = np.pad(wav[tail_start:], (0, window_samples - remainder), mode='constant')
					window_segments.append(tail)
					window_infos.append((tail_start / 16000.0, len(wav) / 16000.0))
					num_padded_short += 1
			else:
				max_start = len(wav) - window_samples
				start_idx = random.randint(0, max_start) if max_start > 0 else 0
				end_idx = start_idx + window_samples
				window_segments.append(wav[start_idx:end_idx])
				window_infos.append((start_idx / 16000.0, min(end_idx, len(wav)) / 16000.0))

		if len(window_segments) > 1:
			num_files_with_multiple_windows += 1

		for segment_idx, segment in enumerate(window_segments):
			if MAX_SPEAKER_VECTORS is not None and MAX_SPEAKER_VECTORS > 0 and len(speaker_dvectors) >= MAX_SPEAKER_VECTORS:
				break
			with torch.no_grad():
				dvec = speaker_encoder.embed_utterance(segment)
			start_sec, end_sec = window_infos[segment_idx]
			speaker_dvectors.append(np.asarray(dvec, dtype=np.float32))
			window_refs.append(
				{
					'file': str(audio_path),
					'utterance': audio_path.stem,
					'start_sec': float(start_sec),
					'end_sec': float(end_sec),
				}
			)
			num_processed_files += 1
			num_windows_used += 1
	except Exception as e:
		num_audio_errors += 1
		print(f'Warning: failed to process {audio_path}: {e}')

if MAX_SPEAKER_VECTORS is not None and MAX_SPEAKER_VECTORS > 0:
	speaker_dvectors = speaker_dvectors[:MAX_SPEAKER_VECTORS]

if len(speaker_dvectors) == 0:
	raise ValueError(
		f"No recomputed d-vectors found for speaker '{speaker}' from LibriSpeech. "
		f"Check SPEAKER_ID and LIBRISPEECH_SUBSETS."
	)

x_before = np.stack(
	[np.asarray(v, dtype=np.float32).reshape(-1) for v in speaker_dvectors],
	axis=0,
)


# Run AE
outputs = []
with torch.no_grad():
	for start in tqdm(range(0, x_before.shape[0], BATCH_SIZE), desc='Running AE', unit='batch'):
		x_batch = torch.from_numpy(x_before[start:start + BATCH_SIZE]).to(device)
		if USE_BOTTLENECK:
			y_batch = ae_model.encode(x_batch)
		else:
			y_batch = ae_model(x_batch)
		outputs.append(y_batch.cpu().numpy())

x_after = np.concatenate(outputs, axis=0)


# Compute before/after comparison (both non-normalized and normalized)
if x_before.shape[1] == x_after.shape[1]:
	dot_sim = np.sum(x_before * x_after, axis=1)
	dot_mean = float(np.mean(dot_sim))
	dot_std = float(np.std(dot_sim))
	dot_min = float(np.min(dot_sim))
	dot_p50 = float(np.median(dot_sim))
	dot_max = float(np.max(dot_sim))
	dot_min_idx = int(np.argmin(dot_sim))
	dot_max_idx = int(np.argmax(dot_sim))

	x_before_norm = np.linalg.norm(x_before, axis=1)
	x_after_norm = np.linalg.norm(x_after, axis=1)
	cosine_sim = dot_sim / np.maximum(x_before_norm * x_after_norm, 1e-12)
	cos_mean = float(np.mean(cosine_sim))
	cos_std = float(np.std(cosine_sim))
	cos_min = float(np.min(cosine_sim))
	cos_p50 = float(np.median(cosine_sim))
	cos_max = float(np.max(cosine_sim))
	cos_min_idx = int(np.argmin(cosine_sim))
	cos_max_idx = int(np.argmax(cosine_sim))

	pair_l2 = np.linalg.norm(x_after - x_before, axis=1)
	pair_l2_mean = float(np.mean(pair_l2))
	pair_l2_std = float(np.std(pair_l2))
	pair_l2_min = float(np.min(pair_l2))
	pair_l2_max = float(np.max(pair_l2))
	pair_l2_min_idx = int(np.argmin(pair_l2))
	pair_l2_max_idx = int(np.argmax(pair_l2))

	pair_l2_norm = pair_l2 / np.maximum(x_before_norm, 1e-12)
	pair_l2_norm_mean = float(np.mean(pair_l2_norm))
	pair_l2_norm_std = float(np.std(pair_l2_norm))
	pair_l2_norm_min = float(np.min(pair_l2_norm))
	pair_l2_norm_max = float(np.max(pair_l2_norm))
	pair_l2_norm_min_idx = int(np.argmin(pair_l2_norm))
	pair_l2_norm_max_idx = int(np.argmax(pair_l2_norm))
else:
	dot_mean = None
	dot_std = None
	dot_min = None
	dot_p50 = None
	dot_max = None
	dot_min_idx = None
	dot_max_idx = None
	cos_mean = None
	cos_std = None
	cos_min = None
	cos_p50 = None
	cos_max = None
	cos_min_idx = None
	cos_max_idx = None
	pair_l2_mean = None
	pair_l2_std = None
	pair_l2_min = None
	pair_l2_max = None
	pair_l2_min_idx = None
	pair_l2_max_idx = None
	pair_l2_norm_mean = None
	pair_l2_norm_std = None
	pair_l2_norm_min = None
	pair_l2_norm_max = None
	pair_l2_norm_min_idx = None
	pair_l2_norm_max_idx = None


# Spread before AE (non-normalized)
centroid_before = np.mean(x_before, axis=0, keepdims=True)
dist_before = np.linalg.norm(x_before - centroid_before, axis=1)
mean_l2_before = float(np.mean(dist_before))
std_l2_before = float(np.std(dist_before))
mean_dim_std_before = float(np.mean(np.std(x_before, axis=0)))
global_std_before = float(np.std(x_before))
dist_before_min = float(np.min(dist_before))
dist_before_max = float(np.max(dist_before))
dist_before_min_idx = int(np.argmin(dist_before))
dist_before_max_idx = int(np.argmax(dist_before))


# Spread before AE (normalized with unit d-vectors)
x_before_unit = x_before / np.maximum(np.linalg.norm(x_before, axis=1, keepdims=True), 1e-12)
centroid_before_unit = np.mean(x_before_unit, axis=0, keepdims=True)
dist_before_unit = np.linalg.norm(x_before_unit - centroid_before_unit, axis=1)
mean_l2_before_norm = float(np.mean(dist_before_unit))
std_l2_before_norm = float(np.std(dist_before_unit))
mean_dim_std_before_norm = float(np.mean(np.std(x_before_unit, axis=0)))
global_std_before_norm = float(np.std(x_before_unit))
dist_before_norm_min = float(np.min(dist_before_unit))
dist_before_norm_max = float(np.max(dist_before_unit))
dist_before_norm_min_idx = int(np.argmin(dist_before_unit))
dist_before_norm_max_idx = int(np.argmax(dist_before_unit))


# Spread after AE (non-normalized)
centroid_after = np.mean(x_after, axis=0, keepdims=True)
dist_after = np.linalg.norm(x_after - centroid_after, axis=1)
mean_l2_after = float(np.mean(dist_after))
std_l2_after = float(np.std(dist_after))
mean_dim_std_after = float(np.mean(np.std(x_after, axis=0)))
global_std_after = float(np.std(x_after))
dist_after_min = float(np.min(dist_after))
dist_after_max = float(np.max(dist_after))
dist_after_min_idx = int(np.argmin(dist_after))
dist_after_max_idx = int(np.argmax(dist_after))


# Spread after AE (normalized with unit d-vectors)
x_after_unit = x_after / np.maximum(np.linalg.norm(x_after, axis=1, keepdims=True), 1e-12)
centroid_after_unit = np.mean(x_after_unit, axis=0, keepdims=True)
dist_after_unit = np.linalg.norm(x_after_unit - centroid_after_unit, axis=1)
mean_l2_after_norm = float(np.mean(dist_after_unit))
std_l2_after_norm = float(np.std(dist_after_unit))
mean_dim_std_after_norm = float(np.mean(np.std(x_after_unit, axis=0)))
global_std_after_norm = float(np.std(x_after_unit))
dist_after_norm_min = float(np.min(dist_after_unit))
dist_after_norm_max = float(np.max(dist_after_unit))
dist_after_norm_min_idx = int(np.argmin(dist_after_unit))
dist_after_norm_max_idx = int(np.argmax(dist_after_unit))


delta_mean_l2 = mean_l2_after - mean_l2_before
delta_mean_dim_std = mean_dim_std_after - mean_dim_std_before
delta_global_std = global_std_after - global_std_before
delta_mean_l2_norm = mean_l2_after_norm - mean_l2_before_norm
delta_mean_dim_std_norm = mean_dim_std_after_norm - mean_dim_std_before_norm
delta_global_std_norm = global_std_after_norm - global_std_before_norm

print('\nRun summary')
print(f'  speaker: {speaker}')
print(
	f'  files: candidates={num_candidate_files}, windows={num_windows_used}, '
	f'vectors={x_before.shape[0]}'
)
print(f'  mode: {"bottleneck" if USE_BOTTLENECK else "reconstruction"} | dim: {x_before.shape[1]} -> {x_after.shape[1]}')

if x_before.shape[0] == 1:
	print('  note: only 1 vector found; spread is 0.0000 by definition.')

print('\nComparison (before vs after)')
if dot_mean is None:
	print('  Skipped: input/output dimensions are different (bottleneck mode).')
else:
	dot_min_ref = window_refs[dot_min_idx]
	dot_max_ref = window_refs[dot_max_idx]
	cos_min_ref = window_refs[cos_min_idx]
	cos_max_ref = window_refs[cos_max_idx]

	print('  Non-normalized')
	print(f'    dot_mean:      {dot_mean:.4f}')
	print(f'    pair_l2_mean:  {pair_l2_mean:.4f}')
	print(f'    dot_min:       {dot_min:.4f}')
	print(f'    dot_max:       {dot_max:.4f}')
	print(
		'    dot_min_ref:   '
		f"{dot_min_ref['utterance']} | {dot_min_ref['start_sec']:.2f}s to {dot_min_ref['end_sec']:.2f}s | {dot_min_ref['file']}"
	)
	print(
		'    dot_max_ref:   '
		f"{dot_max_ref['utterance']} | {dot_max_ref['start_sec']:.2f}s to {dot_max_ref['end_sec']:.2f}s | {dot_max_ref['file']}"
	)

	print('  Normalized')
	print(f'    cos_mean:      {cos_mean:.4f}')
	print(f'    pair_l2n_mean: {pair_l2_norm_mean:.4f}')
	print(f'    cos_min:       {cos_min:.4f}')
	print(f'    cos_max:       {cos_max:.4f}')
	print(
		'    cos_min_ref:   '
		f"{cos_min_ref['utterance']} | {cos_min_ref['start_sec']:.2f}s to {cos_min_ref['end_sec']:.2f}s | {cos_min_ref['file']}"
	)
	print(
		'    cos_max_ref:   '
		f"{cos_max_ref['utterance']} | {cos_max_ref['start_sec']:.2f}s to {cos_max_ref['end_sec']:.2f}s | {cos_max_ref['file']}"
	)

print('\nSpread (non-normalized)')
print(
	f'  mean_l2_to_centroid: before={mean_l2_before:.4f}, '
	f'after={mean_l2_after:.4f}, delta={delta_mean_l2:.4f}'
)
print(
	f'  global_std:          before={global_std_before:.4f}, '
	f'after={global_std_after:.4f}, delta={delta_global_std:.4f}'
)

print('\nSpread (normalized to unit d-vectors)')
print(
	f'  mean_l2_to_centroid: before={mean_l2_before_norm:.4f}, '
	f'after={mean_l2_after_norm:.4f}, delta={delta_mean_l2_norm:.4f}'
)
print(
	f'  global_std:          before={global_std_before_norm:.4f}, '
	f'after={global_std_after_norm:.4f}, delta={delta_global_std_norm:.4f}'
)
