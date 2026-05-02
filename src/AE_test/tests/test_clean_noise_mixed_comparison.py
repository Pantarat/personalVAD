#!/usr/bin/env python3
"""Compare clean, noise, and mixed speaker embeddings (d-vectors).

Evaluates:
1. Single clean utterance d-vector
2. Single noise utterance d-vector
3. Mixed (clean+noise) d-vector at various SNR levels

Metrics computed:
- Cosine similarity between embeddings
- Euclidean distance between embeddings
- L2 norm variations
- Drift from clean/noise baseline at each SNR level
 
 X-vectors extracted via SpeechBrain's pretrained speaker recognition model.

Edit settings below, then run from project root:
python src/AE_test/tests/test_clean_noise_mixed_comparison.py
"""

import warnings
import random
from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from resemblyzer import VoiceEncoder
from speechbrain.inference.speaker import SpeakerRecognition

# # Suppress FutureWarning triggered inside resemblyzer's librosa calls.
# warnings.filterwarnings('ignore', category=FutureWarning, module=r'resemblyzer(\.|$)')


# # =====================
# # SETTINGS (edit these)
# # =====================

# # Audio settings
# SAMPLE_RATE = 16000
# DURATION_SEC = 5.0
# WINDOW_SEC = 1.6
# WINDOW_STRIDE_SEC = 0.4
# FORCE_CPU = False

# # SNR range for mixing (same as training)
# SNR_RANGE_DB = (20.0, 17.0, 15.0, 13.0)

# # Data sources
# LIBRISPEECH_ROOT = 'data/LibriSpeech'
# LIBRISPEECH_SUBSET = 'dev-clean'  # Source for clean speech
# MUSAN_NOISE_DIR = 'kaldi/egs/pvad/musan/noise/free-sound'  # Source for noise

# # X-vector model (SpeechBrain)
# XVECTOR_MODEL = 'speechbrain/spkrec-ecapa-tdnn'  # ECAPA-TDNN x-vector model

# # Plot settings
# PLOT_RESULTS = True
# OUTPUT_PLOT_PATH = 'src/AE_test/test_outputs/clean_noise_mixed_comparison.png'

# # =====================
# # DEVICE SETUP
# # =====================
# device = 'cpu' if FORCE_CPU else ('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Device: {device}')

# # Load VoiceEncoder for d-vector extraction
# print("Loading VoiceEncoder for d-vector extraction...")
# ve = VoiceEncoder(device=device)

# # Load SpeechBrain speaker recognition model for x-vector extraction
# print("Loading SpeechBrain speaker recognition model for x-vector extraction...")
# try:
#     speaker_model = SpeakerRecognition.from_hparams(source=XVECTOR_MODEL, run_opts={"device": device})
#     xvector_available = True
#     print("✅ X-vector model loaded successfully")
# except Exception as e:
#     print(f"⚠️  Failed to load x-vector model: {e}")
#     xvector_available = False


# # =====================
# # UTILITY FUNCTIONS
# # =====================

# def cosine_similarity(a, b):
#     """Compute cosine similarity between embeddings."""
#     a = np.asarray(a, dtype=np.float32).ravel()
#     b = np.asarray(b, dtype=np.float32).ravel()
    
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
    
#     if norm_a == 0 or norm_b == 0:
#         return 0.0
    
#     return np.dot(a, b) / (norm_a * norm_b)


# def euclidean_distance(a, b):
#     """Compute Euclidean distance between embeddings."""
#     a = np.asarray(a, dtype=np.float32).ravel()
#     b = np.asarray(b, dtype=np.float32).ravel()
#     return np.linalg.norm(a - b)


# def l2_norm(a):
#     """Compute L2 norm of an embedding."""
#     a = np.asarray(a, dtype=np.float32).ravel()
#     return np.linalg.norm(a)


# def load_audio_file(path, sr=SAMPLE_RATE, duration=None):
#     """Load audio and optionally trim to duration."""
#     try:
#         y, _ = librosa.load(str(path), sr=sr, mono=True)
#         if duration is not None:
#             max_len = int(sr * duration)
#             if len(y) > max_len:
#                 y = y[:max_len]
#             else:
#                 # Pad if too short
#                 y = np.pad(y, (0, max_len - len(y)), mode='constant')
#         return y
#     except Exception as e:
#         print(f"Error loading {path}: {e}")
#         return None


# def extract_dvectors_from_audio(audio_waveform, ve, sr=SAMPLE_RATE, window_sec=WINDOW_SEC):
#     """Extract d-vectors from audio using sliding windows.
    
#     Returns: (list of d-vectors as numpy arrays, average d-vector)
#     """
#     if len(audio_waveform) == 0:
#         return [], None
    
#     window_frames = int(window_sec * sr)
#     stride = int(WINDOW_STRIDE_SEC * sr)
    
#     dvectors = []
#     for start in range(0, len(audio_waveform) - window_frames + 1, stride):
#         chunk = audio_waveform[start:start + window_frames]
#         try:
#             dv = ve.embed_utterance(chunk)
#             dvectors.append(dv)
#         except Exception as e:
#             print(f"Warning: failed to extract d-vector for window starting at {start}: {e}")
    
#     if len(dvectors) == 0:
#         return [], None
    
#     # Return list of individual d-vectors and their average
#     avg_dvector = np.mean(dvectors, axis=0)
#     return dvectors, avg_dvector


# def extract_xvector_from_audio(audio_waveform, speaker_model, sr=SAMPLE_RATE):
#     """Extract x-vector from audio using SpeechBrain speaker model.
    
#     Returns: (x-vector as numpy array)
#     """
#     if len(audio_waveform) == 0 or speaker_model is None:
#         return None
    
#     try:
#         # Convert to tensor and move to device
#         audio_tensor = torch.from_numpy(audio_waveform).float().unsqueeze(0)
#         if audio_tensor.device.type != device:
#             audio_tensor = audio_tensor.to(device)
        
#         # Extract x-vector embedding
#         with torch.no_grad():
#             xvector = speaker_model.encode_batch(audio_tensor)
        
#         # Convert to numpy and squeeze
#         xvec = xvector.squeeze().cpu().numpy()
#         return xvec
    
#     except Exception as e:
#         print(f"Warning: failed to extract x-vector: {e}")
#         return None

# def mix_audio(clean, noise, snr_db=10.0):
#     """Mix clean and noise audio at specified SNR (dB).
    
#     Returns: (clean, noise_scaled, mixed)
#     """
#     # Ensure same length
#     min_len = min(len(clean), len(noise))
#     clean = clean[:min_len]
#     noise = noise[:min_len]
    
#     # Calculate scaling factor to achieve target SNR
#     clean_power = np.mean(clean ** 2)
#     noise_power = np.mean(noise ** 2)
    
#     if noise_power == 0:
#         return clean, noise * 0, clean
    
#     snr_linear = 10.0 ** (snr_db / 10.0)
#     noise_scale = np.sqrt(clean_power / (snr_linear * noise_power))
    
#     noise_scaled = noise * noise_scale
#     mixed = clean + noise_scaled
    
#     return clean, noise_scaled, mixed


# # =====================
# # DATA LOADING
# # =====================
# print("\nLoading sample audio files...")
# clean_files = list(Path(LIBRISPEECH_ROOT).rglob(f"{LIBRISPEECH_SUBSET}/**/*.flac"))
# noise_files = list(Path(MUSAN_NOISE_DIR).rglob("*.wav"))

# if len(clean_files) == 0:
#     raise FileNotFoundError(f"No audio files found in {LIBRISPEECH_ROOT}/{LIBRISPEECH_SUBSET}")
# if len(noise_files) == 0:
#     raise FileNotFoundError(f"No noise files found in {MUSAN_NOISE_DIR}")

# # Randomly pick one clean file and one noise file
# clean_file = random.choice(clean_files)
# noise_file = random.choice(noise_files)

# print(f"✅ Clean file: {clean_file.name}")
# print(f"✅ Noise file: {noise_file.name}")


# # =====================
# # D-VECTOR COMPARISON
# # =====================
# print("\n" + "=" * 80)
# print("D-VECTOR COMPARISON: CLEAN vs NOISE vs MIXED")
# print("=" * 80)

# # Load and process audio
# clean_audio = load_audio_file(clean_file, sr=SAMPLE_RATE, duration=DURATION_SEC)
# noise_audio = load_audio_file(noise_file, sr=SAMPLE_RATE, duration=DURATION_SEC)

# if clean_audio is None or noise_audio is None:
#     raise RuntimeError("Failed to load audio files")

# print(f"✅ Clean audio: {len(clean_audio) / SAMPLE_RATE:.2f}s")
# print(f"✅ Noise audio: {len(noise_audio) / SAMPLE_RATE:.2f}s")

# # Extract d-vectors from clean and noise
# print("\nExtracting embeddings...")
# clean_dvecs, clean_dvec_avg = extract_dvectors_from_audio(clean_audio, ve, sr=SAMPLE_RATE)
# noise_dvecs, noise_dvec_avg = extract_dvectors_from_audio(noise_audio, ve, sr=SAMPLE_RATE)

# if clean_dvec_avg is None or noise_dvec_avg is None:
#     raise RuntimeError("Failed to extract d-vectors")

# print(f"  Clean d-vectors: {len(clean_dvecs)}, L2 norm: {l2_norm(clean_dvec_avg):.4f}")
# print(f"  Noise d-vectors: {len(noise_dvecs)}, L2 norm: {l2_norm(noise_dvec_avg):.4f}")

# # Extract x-vectors if available
# clean_xvec = None
# noise_xvec = None
# if xvector_available:
#     clean_xvec = extract_xvector_from_audio(clean_audio, speaker_model, sr=SAMPLE_RATE)
#     noise_xvec = extract_xvector_from_audio(noise_audio, speaker_model, sr=SAMPLE_RATE)
    
#     if clean_xvec is not None:
#         print(f"  Clean x-vector: shape={clean_xvec.shape}, L2 norm: {l2_norm(clean_xvec):.4f}")
#     if noise_xvec is not None:
#         print(f"  Noise x-vector: shape={noise_xvec.shape}, L2 norm: {l2_norm(noise_xvec):.4f}")

# # Baseline comparison (clean vs noise)
# baseline_cos = cosine_similarity(clean_dvec_avg, noise_dvec_avg)
# baseline_eucl = euclidean_distance(clean_dvec_avg, noise_dvec_avg)
# print(f"\nBaseline Clean vs Noise:")
# print(f"  Cosine similarity: {baseline_cos:.6f}")
# print(f"  Euclidean distance: {baseline_eucl:.6f}")

# baseline_xvec_cos = None
# baseline_xvec_eucl = None
# if xvector_available and clean_xvec is not None and noise_xvec is not None:
#     baseline_xvec_cos = cosine_similarity(clean_xvec, noise_xvec)
#     baseline_xvec_eucl = euclidean_distance(clean_xvec, noise_xvec)
#     print(f"\nBaseline Clean vs Noise (x-vectors):")
#     print(f"  Cosine similarity: {baseline_xvec_cos:.6f}")
#     print(f"  Euclidean distance: {baseline_xvec_eucl:.6f}")

# # Process multiple SNR levels
# print(f"\nProcessing mixed audio at SNR levels: {SNR_RANGE_DB}")
# dvector_results = {}
# xvector_results = {}

# for snr_db in SNR_RANGE_DB:
#     clean_audio_snr, noise_audio_snr, mixed_audio = mix_audio(clean_audio, noise_audio, snr_db=snr_db)
    
#     # Extract d-vectors from mixed audio
#     mixed_dvecs, mixed_dvec_avg = extract_dvectors_from_audio(mixed_audio, ve, sr=SAMPLE_RATE)
    
#     if mixed_dvec_avg is None:
#         print(f"  SNR {snr_db:5.1f}dB: Failed to extract d-vectors")
#         continue
    
#     # Compute similarities and distances
#     mixed_vs_clean_cos = cosine_similarity(mixed_dvec_avg, clean_dvec_avg)
#     mixed_vs_noise_cos = cosine_similarity(mixed_dvec_avg, noise_dvec_avg)
    
#     mixed_vs_clean_eucl = euclidean_distance(mixed_dvec_avg, clean_dvec_avg)
#     mixed_vs_noise_eucl = euclidean_distance(mixed_dvec_avg, noise_dvec_avg)
    
#     mixed_l2 = l2_norm(mixed_dvec_avg)
    
#     dvector_results[snr_db] = {
#         'mixed_vs_clean_cos': mixed_vs_clean_cos,
#         'mixed_vs_noise_cos': mixed_vs_noise_cos,
#         'mixed_vs_clean_eucl': mixed_vs_clean_eucl,
#         'mixed_vs_noise_eucl': mixed_vs_noise_eucl,
#         'mixed_l2_norm': mixed_l2,
#         'clean_l2_norm': l2_norm(clean_dvec_avg),
#         'noise_l2_norm': l2_norm(noise_dvec_avg),
#     }
    
#     print(f"\n  SNR {snr_db:5.1f}dB:")
#     print(f"    Mixed vs Clean: cos={mixed_vs_clean_cos:.6f}, eucl={mixed_vs_clean_eucl:.4f}")
#     print(f"    Mixed vs Noise: cos={mixed_vs_noise_cos:.6f}, eucl={mixed_vs_noise_eucl:.4f}")
#     print(f"    L2 norms: mixed={mixed_l2:.4f}, clean={l2_norm(clean_dvec_avg):.4f}, noise={l2_norm(noise_dvec_avg):.4f}")

#     # Extract and compare x-vectors if available
#     if xvector_available and clean_xvec is not None and noise_xvec is not None:
#         mixed_xvec = extract_xvector_from_audio(mixed_audio, speaker_model, sr=SAMPLE_RATE)
    
#         if mixed_xvec is not None:
#             mixed_xvec_vs_clean_cos = cosine_similarity(mixed_xvec, clean_xvec)
#             mixed_xvec_vs_noise_cos = cosine_similarity(mixed_xvec, noise_xvec)
#             mixed_xvec_vs_clean_eucl = euclidean_distance(mixed_xvec, clean_xvec)
#             mixed_xvec_vs_noise_eucl = euclidean_distance(mixed_xvec, noise_xvec)
#             mixed_xvec_l2 = l2_norm(mixed_xvec)
        
#             xvector_results[snr_db] = {
#                 'mixed_vs_clean_cos': mixed_xvec_vs_clean_cos,
#                 'mixed_vs_noise_cos': mixed_xvec_vs_noise_cos,
#                 'mixed_vs_clean_eucl': mixed_xvec_vs_clean_eucl,
#                 'mixed_vs_noise_eucl': mixed_xvec_vs_noise_eucl,
#                 'mixed_l2_norm': mixed_xvec_l2,
#                 'clean_l2_norm': l2_norm(clean_xvec),
#                 'noise_l2_norm': l2_norm(noise_xvec),
#             }
        
#             print(f"    X-vector - Mixed vs Clean: cos={mixed_xvec_vs_clean_cos:.6f}, eucl={mixed_xvec_vs_clean_eucl:.4f}")
#             print(f"    X-vector - Mixed vs Noise: cos={mixed_xvec_vs_noise_cos:.6f}, eucl={mixed_xvec_vs_noise_eucl:.4f}")

# # =====================
# # PLOTTING RESULTS
# # =====================
# if PLOT_RESULTS:
#     print("\n" + "=" * 80)
#     print("PLOTTING RESULTS")
#     print("=" * 80)
    
#     snrs = sorted(dvector_results.keys(), reverse=True)
    
#     # Select one SNR level to show (middle of range)
#     selected_snr = SNR_RANGE_DB[len(SNR_RANGE_DB) // 2]
#     mixed_dvec_selected = extract_dvectors_from_audio(
#         mix_audio(clean_audio, noise_audio, snr_db=selected_snr)[2], ve, sr=SAMPLE_RATE)[1]
    
#     # Determine if we should show x-vectors as well
#     show_xvector = xvector_available and clean_xvec is not None and noise_xvec is not None
    
#     # Create figure for d-vector components
#     fig_dvec = plt.figure(figsize=(16, 6))
#     fig_dvec.suptitle(f'D-Vector Component Comparison (256 dimensions)', fontsize=14, fontweight='bold')
    
#     ax_dvec = fig_dvec.add_subplot(111)
#     dims = np.arange(len(clean_dvec_avg))
#     width = 0.25
    
#     ax_dvec.bar(dims - width, clean_dvec_avg, width=width, label='Clean', alpha=0.8, color='green')
#     ax_dvec.bar(dims, noise_dvec_avg, width=width, label='Noise', alpha=0.8, color='red')
#     ax_dvec.bar(dims + width, mixed_dvec_selected, width=width, label=f'Mixed (SNR={selected_snr}dB)', alpha=0.8, color='purple')
    
#     ax_dvec.set_xlabel('Dimension', fontsize=11)
#     ax_dvec.set_ylabel('Value', fontsize=11)
#     ax_dvec.set_title('D-Vector: Clean vs Noise vs Mixed Audio', fontsize=12, fontweight='bold')
#     ax_dvec.legend(loc='upper right', fontsize=10)
#     ax_dvec.grid(True, axis='y', alpha=0.3)
    
#     plt.tight_layout()
#     output_path_dvec = Path(OUTPUT_PLOT_PATH).parent / 'dvector_components.png'
#     output_path_dvec.parent.mkdir(parents=True, exist_ok=True)
#     fig_dvec.savefig(output_path_dvec, dpi=150, bbox_inches='tight')
#     print(f"✅ D-vector plot saved to: {output_path_dvec}")
#     plt.close(fig_dvec)
    
#     # Create figure for x-vector components if available
#     if show_xvector:
#         print("Extracting x-vector for selected SNR...")
#         _, _, mixed_audio_selected = mix_audio(clean_audio, noise_audio, snr_db=selected_snr)
#         mixed_xvec_selected = extract_xvector_from_audio(mixed_audio_selected, speaker_model, sr=SAMPLE_RATE)
        
#         if mixed_xvec_selected is not None:
#             fig_xvec = plt.figure(figsize=(18, 6))
#             fig_xvec.suptitle(f'X-Vector Component Comparison (512 dimensions)', fontsize=14, fontweight='bold')
            
#             ax_xvec = fig_xvec.add_subplot(111)
#             dims = np.arange(len(clean_xvec))
#             width = 0.25
            
#             ax_xvec.bar(dims - width, clean_xvec, width=width, label='Clean', alpha=0.8, color='green')
#             ax_xvec.bar(dims, noise_xvec, width=width, label='Noise', alpha=0.8, color='red')
#             ax_xvec.bar(dims + width, mixed_xvec_selected, width=width, label=f'Mixed (SNR={selected_snr}dB)', alpha=0.8, color='purple')
            
#             ax_xvec.set_xlabel('Dimension', fontsize=11)
#             ax_xvec.set_ylabel('Value', fontsize=11)
#             ax_xvec.set_title('X-Vector: Clean vs Noise vs Mixed Audio', fontsize=12, fontweight='bold')
#             ax_xvec.legend(loc='upper right', fontsize=10)
#             ax_xvec.grid(True, axis='y', alpha=0.3)
            
#             plt.tight_layout()
#             output_path_xvec = Path(OUTPUT_PLOT_PATH).parent / 'xvector_components.png'
#             output_path_xvec.parent.mkdir(parents=True, exist_ok=True)
#             fig_xvec.savefig(output_path_xvec, dpi=150, bbox_inches='tight')
#             print(f"✅ X-vector plot saved to: {output_path_xvec}")
#             plt.close(fig_xvec)

import tempfile
tmpdir = tempfile.TemporaryDirectory()
xvector_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=tmpdir.name)
xvector = xvector_model.encode_batch(np.zeros(16000))
plt.figure()
plt.bar(np.arange(len(xvector)), xvector, width=0.5,
        label='X-Vector', alpha=0.8, color='blue')

plt.ylim(0, 1)  # same thing here

plt.legend()
plt.show()