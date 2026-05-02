#!/usr/bin/env python3
"""Deep stacked AE reconstruction quality test.

Evaluates cosine similarity and MSE for:
1. Clean speech original - clean speech reconstructed
2. Noise only original - noise only reconstructed
3. Clean + noise original - clean + noise reconstructed
4. Clean original - clean + noise reconstructed (leakage check)
5. Noise original - clean + noise reconstructed (leakage check)

Edit settings below, then run from project root:
python src/AE_test/tests/test_deep_stacked_reconstruction.py
"""

import sys
import warnings
import random
from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
AE_MODEL_DIR = 'src/AE_test/test_outputs/models/dvector_ae_deep_stacked_libri_babble_2_29-4-26'

# Audio settings
SAMPLE_RATE = 16000
DURATION_SEC = 5.0
WINDOW_SEC = 1.6
WINDOW_STRIDE_SEC = 0.4
BATCH_SIZE = 512
FORCE_CPU = False

# Data sources
LIBRISPEECH_ROOT = 'data/LibriSpeech'
LIBRISPEECH_SUBSET = 'dev-clean'  # Source for clean speech
MUSAN_NOISE_DIR = 'kaldi/egs/pvad/musan/noise/free-sound'  # Source for noise
MUSAN_BABBLE_SNR_RANGE_DB = (20.0, 17.0, 15.0, 13.0)
N_SAMPLES_EACH = 50  # Number of clean utterances and noise clips to test

# =====================
# DEVICE & MODEL SETUP
# =====================
device = 'cpu' if FORCE_CPU else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load AE model
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
ae_model.eval()

# Load VoiceEncoder for d-vector extraction
print("Loading VoiceEncoder...")
ve = VoiceEncoder(device=device)


# =====================
# UTILITIES
# =====================
def cosine_similarity(a, b):
    """Compute mean cosine similarity between d-vectors (N, 256)."""
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()

    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    
    # Normalize to unit norm
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    
    # Pairwise cosine similarity
    sim = torch.mm(a_norm, b_norm.t())  # (N_a, N_b)
    
    if sim.shape[0] == sim.shape[1]:
        # Diagonal is most relevant (same utterance)
        return torch.diag(sim).mean().item()
    return sim.mean().item()


def mse(a, b):
    """Compute mean squared error between d-vectors."""
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    return F.mse_loss(a, b).item()


def scale_to_match(a, b):
    """Scale a vector to match the L2 norm of b."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= 0 or b_norm <= 0:
        return a
    return a * (b_norm / a_norm)


def load_audio_file(path, sr=SAMPLE_RATE, duration=None):
    """Load audio and optionally trim to duration."""
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        if duration is not None:
            max_len = int(sr * duration)
            if len(y) > max_len:
                y = y[:max_len]
            else:
                # Pad if too short
                y = np.pad(y, (0, max_len - len(y)), mode='constant')
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_dvectors_from_audio(audio_waveform, ve, sr=SAMPLE_RATE, window_sec=WINDOW_SEC):
    """Extract d-vectors from audio using sliding windows."""
    if len(audio_waveform) == 0:
        return torch.tensor([], device=device)
    
    window_frames = int(window_sec * sr)
    stride = int(WINDOW_STRIDE_SEC * sr)
    
    dvectors = []
    for start in range(0, len(audio_waveform) - window_frames + 1, stride):
        chunk = audio_waveform[start:start + window_frames]
        try:
            # VoiceEncoder expects `rate` to mean partial window rate, not sample rate.
            dv = ve.embed_utterance(chunk)
            dvectors.append(torch.from_numpy(dv).to(device))
        except Exception as e:
            print(f"Warning: failed to extract d-vector for window starting at {start}: {e}")
    
    if len(dvectors) == 0:
        return torch.tensor([], device=device)
    
    return torch.stack(dvectors)


def reconstruct_dvectors(dvectors_tensor):
    """Pass d-vectors through AE and return reconstructed outputs."""
    if dvectors_tensor.shape[0] == 0:
        return torch.tensor([], device=device)
    
    dvectors_tensor = dvectors_tensor.float().to(device)
    reconstructed = []
    
    with torch.no_grad():
        for i in range(0, len(dvectors_tensor), BATCH_SIZE):
            batch = dvectors_tensor[i:i + BATCH_SIZE]
            out = ae_model(batch)
            reconstructed.append(out.cpu())
    
    return torch.cat(reconstructed, dim=0)


def mix_audio(clean, noise, snr_db=10.0):
    """Mix clean and noise audio at specified SNR (dB)."""
    # Ensure same length
    min_len = min(len(clean), len(noise))
    clean = clean[:min_len]
    noise = noise[:min_len]
    
    # Calculate scaling factor to achieve target SNR
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return clean, noise, clean + noise
    
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_scale = np.sqrt(clean_power / (snr_linear * noise_power))
    
    noise_scaled = noise * noise_scale
    mixed = clean + noise_scaled
    
    return clean, noise_scaled, mixed


# =====================
# DATA COLLECTION
# =====================
print(f"\nCollecting {N_SAMPLES_EACH} clean utterances from LibriSpeech...")
clean_files = list(Path(LIBRISPEECH_ROOT).rglob(f"{LIBRISPEECH_SUBSET}/**/*.flac"))
clean_files = clean_files[:N_SAMPLES_EACH] if N_SAMPLES_EACH else clean_files

print(f"Collecting {N_SAMPLES_EACH} noise clips from MUSAN...")
noise_files = list(Path(MUSAN_NOISE_DIR).rglob("*.wav"))
noise_files = noise_files[:N_SAMPLES_EACH] if N_SAMPLES_EACH else noise_files

if len(clean_files) == 0:
    raise FileNotFoundError(f"No audio files found in {LIBRISPEECH_ROOT}/{LIBRISPEECH_SUBSET}")
if len(noise_files) == 0:
    raise FileNotFoundError(f"No noise files found in {MUSAN_NOISE_DIR}")

print(f"✅ Found {len(clean_files)} clean files and {len(noise_files)} noise files")


# =====================
# MAIN EVALUATION LOOP
# =====================
print("\n" + "=" * 80)
print("DEEP STACKED AE RECONSTRUCTION QUALITY TEST")
print("=" * 80)

metrics = {
    'clean_vs_clean_recon': {'cos_sim': [], 'mse': []},
    'noise_vs_noise_recon': {'cos_sim': [], 'mse': []},
    'mixed_vs_mixed_recon': {'cos_sim': [], 'mse': []},
    'clean_vs_mixed_recon': {'cos_sim': [], 'mse': []},
    'noise_vs_mixed_recon': {'cos_sim': [], 'mse': []},
}

# Capture up to 5 example pairs (orig, recon) per comparison to plot later
examples = {}
print(f"\nProcessing {min(len(clean_files), len(noise_files))} sample pairs...")
for idx in tqdm(range(min(len(clean_files), len(noise_files)))):
    # Load clean and noise
    clean_audio = load_audio_file(clean_files[idx], sr=SAMPLE_RATE, duration=DURATION_SEC)
    noise_audio = load_audio_file(noise_files[idx], sr=SAMPLE_RATE, duration=DURATION_SEC)
    
    if clean_audio is None or noise_audio is None:
        continue
    
    # Match the deep-stacked training noise level by sampling from the same SNR range.
    snr_db = random.choice(MUSAN_BABBLE_SNR_RANGE_DB)
    clean, noise, mixed = mix_audio(clean_audio, noise_audio, snr_db=snr_db)
    
    # Extract d-vectors
    clean_dvecs = extract_dvectors_from_audio(clean, ve, sr=SAMPLE_RATE)
    noise_dvecs = extract_dvectors_from_audio(noise, ve, sr=SAMPLE_RATE)
    mixed_dvecs = extract_dvectors_from_audio(mixed, ve, sr=SAMPLE_RATE)
    
    if len(clean_dvecs) == 0 or len(noise_dvecs) == 0 or len(mixed_dvecs) == 0:
        continue
    
    # Reconstruct
    clean_recon = reconstruct_dvectors(clean_dvecs)
    noise_recon = reconstruct_dvectors(noise_dvecs)
    mixed_recon = reconstruct_dvectors(mixed_dvecs)
    
    # 1. Clean original vs clean reconstructed
    if len(clean_dvecs) > 0:
        cos_sim_val = cosine_similarity(clean_dvecs, clean_recon)
        mse_val = mse(clean_dvecs, clean_recon)
        metrics['clean_vs_clean_recon']['cos_sim'].append(cos_sim_val)
        metrics['clean_vs_clean_recon']['mse'].append(mse_val)
        if len(examples.get('clean_vs_clean_recon', [])) < 5 and len(clean_recon) > 0:
            idx_choice = random.randint(0, len(clean_recon) - 1)
            examples.setdefault('clean_vs_clean_recon', []).append(
                (clean_dvecs[idx_choice].detach().cpu().numpy(), clean_recon[idx_choice].detach().cpu().numpy())
            )
    
    # 2. Noise original vs noise reconstructed
    if len(noise_dvecs) > 0:
        cos_sim_val = cosine_similarity(noise_dvecs, noise_recon)
        mse_val = mse(noise_dvecs, noise_recon)
        metrics['noise_vs_noise_recon']['cos_sim'].append(cos_sim_val)
        metrics['noise_vs_noise_recon']['mse'].append(mse_val)
        if len(examples.get('noise_vs_noise_recon', [])) < 5 and len(noise_recon) > 0:
            idx_choice = random.randint(0, len(noise_recon) - 1)
            orig_noise = noise_dvecs[idx_choice].detach().cpu().numpy()
            recon_noise = noise_recon[idx_choice].detach().cpu().numpy()
            examples.setdefault('noise_vs_noise_recon', []).append(
                (scale_to_match(orig_noise, recon_noise), recon_noise)
            )
    
    # 3. Mixed original vs mixed reconstructed
    if len(mixed_dvecs) > 0:
        cos_sim_val = cosine_similarity(mixed_dvecs, mixed_recon)
        mse_val = mse(mixed_dvecs, mixed_recon)
        metrics['mixed_vs_mixed_recon']['cos_sim'].append(cos_sim_val)
        metrics['mixed_vs_mixed_recon']['mse'].append(mse_val)
        if len(examples.get('mixed_vs_mixed_recon', [])) < 5 and len(mixed_recon) > 0:
            idx_choice = random.randint(0, len(mixed_recon) - 1)
            examples.setdefault('mixed_vs_mixed_recon', []).append(
                (mixed_dvecs[idx_choice].detach().cpu().numpy(), mixed_recon[idx_choice].detach().cpu().numpy())
            )
    
    # 4. Clean original vs mixed reconstructed (cross comparison)
    # Match by taking same number of vectors
    min_len = min(len(clean_dvecs), len(mixed_recon))
    if min_len > 0:
        cos_sim_val = cosine_similarity(clean_dvecs[:min_len], mixed_recon[:min_len])
        mse_val = mse(clean_dvecs[:min_len], mixed_recon[:min_len])
        metrics['clean_vs_mixed_recon']['cos_sim'].append(cos_sim_val)
        metrics['clean_vs_mixed_recon']['mse'].append(mse_val)
        if len(examples.get('clean_vs_mixed_recon', [])) < 5 and min_len > 0:
            idx_choice = random.randint(0, min_len - 1)
            examples.setdefault('clean_vs_mixed_recon', []).append(
                (clean_dvecs[:min_len][idx_choice].detach().cpu().numpy(), mixed_recon[:min_len][idx_choice].detach().cpu().numpy())
            )
    
    # 5. Noise original vs mixed reconstructed (cross comparison)
    min_len = min(len(noise_dvecs), len(mixed_recon))
    if min_len > 0:
        cos_sim_val = cosine_similarity(noise_dvecs[:min_len], mixed_recon[:min_len])
        mse_val = mse(noise_dvecs[:min_len], mixed_recon[:min_len])
        metrics['noise_vs_mixed_recon']['cos_sim'].append(cos_sim_val)
        metrics['noise_vs_mixed_recon']['mse'].append(mse_val)
        if len(examples.get('noise_vs_mixed_recon', [])) < 5 and min_len > 0:
            idx_choice = random.randint(0, min_len - 1)
            orig_noise = noise_dvecs[:min_len][idx_choice].detach().cpu().numpy()
            recon_noise = mixed_recon[:min_len][idx_choice].detach().cpu().numpy()
            examples.setdefault('noise_vs_mixed_recon', []).append(
                (scale_to_match(orig_noise, recon_noise), recon_noise)
            )


# =====================
# RESULTS SUMMARY
# =====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

results_summary = []

for test_name, values in metrics.items():
    if len(values['cos_sim']) == 0:
        continue
    
    cos_sim_mean = np.mean(values['cos_sim'])
    cos_sim_std = np.std(values['cos_sim'])
    mse_mean = np.mean(values['mse'])
    mse_std = np.std(values['mse'])
    n_samples = len(values['cos_sim'])
    
    results_summary.append({
        'Test': test_name,
        'N': n_samples,
        'Cosine Sim (mean±std)': f"{cos_sim_mean:.4f}±{cos_sim_std:.4f}",
        'MSE (mean±std)': f"{mse_mean:.6f}±{mse_std:.6f}",
    })
    
    print(f"\n{test_name}:")
    print(f"  Samples: {n_samples}")
    print(f"  Cosine Similarity: {cos_sim_mean:.4f} ± {cos_sim_std:.4f}")
    print(f"  MSE: {mse_mean:.6f} ± {mse_std:.6f}")

if not results_summary:
    print("\n⚠️  No results were produced. Check that:")
    print("   - the clean and noise directories contain readable audio files")
    print("   - VoiceEncoder can extract d-vectors from the selected clips")
    print("   - the AE checkpoint path is correct")

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("""
✅ High cosine similarity (>0.9) & low MSE indicate good reconstruction fidelity
✅ Clean vs clean recon should have highest similarity (direct reconstruction)
✅ Clean vs mixed recon should be LOW (indicates noise is being separated)
✅ Noise vs mixed recon should be LOW (indicates clean signal separation)
⚠️  If clean vs mixed recon is HIGH, AE may not be removing noise properly
⚠️  If noise vs noise recon is very LOW, noise representation may be unstable
""")

print("\n✅ Test complete!")

# =====================
# SAVE EXAMPLE BARGRAPHS
# =====================
out_dir = model_dir / 'reconstruction_bargraphs'
out_dir.mkdir(parents=True, exist_ok=True)

if len(examples) == 0:
    print("No example d-vector pairs were captured for plotting.")
else:
    for name, pairs in examples.items():
        for i, (orig, recon) in enumerate(pairs):
            try:
                orig = np.asarray(orig).ravel()
                recon = np.asarray(recon).ravel()
                dims = min(len(orig), len(recon))
                x = np.arange(dims)
                fig, ax = plt.subplots(figsize=(20, 6))
                width = 0.35
                ax.bar(x - width/2, orig[:dims], width, label='original')
                ax.bar(x + width/2, recon[:dims], width, label='recon')
                ax.set_title(f"{name} example {i+1}: component-wise comparison")
                ax.set_xlabel('d-vector dimension')
                ax.set_ylabel('value')
                ax.legend()
                plt.tight_layout()
                save_path = out_dir / f"{name}_{i+1}.png"
                plt.savefig(save_path, dpi=200)
                plt.close()
                print(f"Saved bargraph: {save_path}")
            except Exception as e:
                print(f"Failed to save bargraph for {name} #{i+1}: {e}")
