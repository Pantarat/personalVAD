#!/usr/bin/env python3
"""
Diagnostic script to verify identity autoencoder behavior.

This script checks:
1. Autoencoder reconstruction quality (should be near-perfect for identity AE)
2. Whether d-vectors are actually being transformed
3. Cosine similarity before/after autoencoder processing
4. If there's residual data from previous evaluations
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import only what we need to avoid loading all dependencies
from vad_set_ae import load_autoencoder
import kaldiio

# Configuration
IDENTITY_AE_PATH = 'src/AE_test/test_outputs/dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000'
EMBED_PATH = 'data/embeddings'
TEST_DIR = 'data/84_ov_test_ov50pct_main84_babble_100'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("=" * 80)
    print("IDENTITY AUTOENCODER DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Load and inspect autoencoder
    print(f"\n1️⃣  LOADING AUTOENCODER")
    print(f"   Path: {IDENTITY_AE_PATH}")
    
    ae_path = Path(IDENTITY_AE_PATH)
    if not ae_path.is_absolute():
        ae_path = Path(__file__).parent / ae_path
    
    print(f"   Absolute path: {ae_path}")
    print(f"   Exists: {ae_path.exists()}")
    
    if not ae_path.exists():
        print(f"\n❌ ERROR: Autoencoder path does not exist!")
        return
    
    # Load config
    config_path = ae_path / 'config.pkl'
    print(f"\n   Loading config from: {config_path}")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    print(f"\n   📋 Config keys: {list(config.keys())}")
    print(f"   Architecture: {config['hidden_dims']}")
    print(f"   Bottleneck dim: {config['hidden_dims'][len(config['hidden_dims'])//2]}")
    
    # Check for training_scheme
    if 'training_scheme' in config:
        print(f"   ⚠️  Training scheme: {config['training_scheme']} (EXTRACTION model)")
        print(f"\n❌ ERROR: This appears to be an EXTRACTION autoencoder, not IDENTITY!")
        print(f"   Identity models should NOT have 'training_scheme' in config.")
        return
    else:
        print(f"   ✓ Training scheme: Not present (IDENTITY model confirmed)")
    
    # Check for other identity-specific markers
    if 'include_noise_identity' in config:
        print(f"   ✓ include_noise_identity: {config['include_noise_identity']}")
    
    # Load autoencoder using the function from vad_set_ae
    print(f"\n   Loading autoencoder model...")
    autoencoder, ae_config = load_autoencoder(str(ae_path), device)
    autoencoder.eval()
    autoencoder = autoencoder.to(device)  # Ensure on correct device
    
    print(f"   ✅ Autoencoder loaded successfully!")
    print(f"   Device: {device}")
    
    # 2. Test reconstruction quality on enrolled d-vectors
    print(f"\n2️⃣  TESTING RECONSTRUCTION QUALITY ON ENROLLED D-VECTORS")
    print(f"   Loading enrolled d-vectors from: {EMBED_PATH}")
    
    dvectors = kaldiio.load_scp(f'{EMBED_PATH}/dvectors.scp')
    print(f"   Loaded {len(dvectors)} enrolled d-vectors")
    
    # Test on first 5 speakers
    test_speakers = list(dvectors.keys())[:5]
    print(f"\n   Testing on speakers: {test_speakers}")
    
    reconstruction_errors = []
    cosine_similarities = []
    
    with torch.no_grad():
        for speaker in test_speakers:
            original = dvectors[speaker]
            original_tensor = torch.FloatTensor(original).unsqueeze(0).to(device)
            
            # Full reconstruction
            reconstructed_tensor = autoencoder(original_tensor)
            reconstructed = reconstructed_tensor.cpu().numpy().squeeze()
            
            # Compute metrics
            mse = np.mean((original - reconstructed) ** 2)
            cos_sim = compute_cosine_similarity(original, reconstructed)
            
            reconstruction_errors.append(mse)
            cosine_similarities.append(cos_sim)
            
            print(f"\n   Speaker: {speaker}")
            print(f"      Original shape: {original.shape}")
            print(f"      Reconstructed shape: {reconstructed.shape}")
            print(f"      MSE: {mse:.8f}")
            print(f"      Cosine similarity: {cos_sim:.8f}")
            print(f"      Original norm: {np.linalg.norm(original):.4f}")
            print(f"      Reconstructed norm: {np.linalg.norm(reconstructed):.4f}")
    
    avg_mse = np.mean(reconstruction_errors)
    avg_cos_sim = np.mean(cosine_similarities)
    
    print(f"\n   📊 SUMMARY:")
    print(f"      Average MSE: {avg_mse:.8f}")
    print(f"      Average Cosine Similarity: {avg_cos_sim:.8f}")
    
    # Interpretation
    if avg_cos_sim > 0.999:
        print(f"      ✅ EXCELLENT: Near-perfect reconstruction (identity-like)")
    elif avg_cos_sim > 0.99:
        print(f"      ✓ GOOD: Very high similarity (identity-like)")
    elif avg_cos_sim > 0.95:
        print(f"      ⚠️  MODERATE: Some transformation is happening")
    else:
        print(f"      ❌ POOR: Significant transformation (NOT identity-like)")
    
    # 3. Test on stream d-vectors (if wav files available)
    print(f"\n3️⃣  TESTING ON STREAM D-VECTORS")
    
    wav_scp_path = f'{TEST_DIR}/wav.scp'
    if not Path(wav_scp_path).exists():
        print(f"   ⚠️  No wav.scp found at {wav_scp_path}")
        print(f"   Skipping stream d-vector test")
    else:
        print(f"   Loading audio files from: {wav_scp_path}")
        wavs = kaldiio.load_scp(wav_scp_path)
        
        # Load resemblyzer
        from resemblyzer_mod import VoiceEncoderMod
        import librosa
        
        print(f"   Initializing VoiceEncoderMod...")
        dvector_encoder = VoiceEncoderMod(device=str(device))
        dvector_encoder.eval()
        dvector_encoder = dvector_encoder.to(device)
        print(f"   VoiceEncoderMod device: {device}")
        
        # Test on first file
        test_keys = list(wavs.keys())[:2]
        print(f"\n   Testing on files: {test_keys}")
        
        for key in test_keys:
            print(f"\n   File: {key}")
            
            try:
                sr, audio = wavs[key]
                audio = audio.astype(np.float32) / 32768.0
                
                # Extract fbanks
                fbanks_audio = librosa.feature.melspectrogram(
                    y=audio, sr=16000, n_fft=400, hop_length=160, n_mels=40
                ).astype('float32').T[:-2]
                
                # Extract stream d-vectors
                fbanks_tensor = torch.unsqueeze(torch.from_numpy(fbanks_audio), 0).to(device)
                with torch.no_grad():
                    embeds_stream, _ = dvector_encoder.forward_stream(fbanks_tensor, None)
                    embeds_stream = embeds_stream.cpu().numpy().squeeze()
                
                print(f"      Extracted {len(embeds_stream)} frame-level d-vectors")
                
                # Test reconstruction on a few frames
                test_frames = [0, len(embeds_stream)//2, -1]
                frame_cos_sims = []
                
                for frame_idx in test_frames:
                    original_frame = embeds_stream[frame_idx]
                    original_frame_tensor = torch.FloatTensor(original_frame).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        reconstructed_frame_tensor = autoencoder(original_frame_tensor)
                        reconstructed_frame = reconstructed_frame_tensor.cpu().numpy().squeeze()
                    
                    cos_sim = compute_cosine_similarity(original_frame, reconstructed_frame)
                    frame_cos_sims.append(cos_sim)
                    
                    print(f"      Frame {frame_idx}: cos_sim = {cos_sim:.8f}")
                
                avg_frame_cos_sim = np.mean(frame_cos_sims)
                print(f"      Average frame cos_sim: {avg_frame_cos_sim:.8f}")
                
            except Exception as e:
                print(f"      ❌ Error processing file: {e}")
    
    # 4. Check for pre-computed scores
    print(f"\n4️⃣  CHECKING FOR RESIDUAL PRE-COMPUTED DATA")
    
    scores_path = f'{TEST_DIR}/scores.scp'
    if Path(scores_path).exists():
        print(f"   ✓ Found scores.scp at: {scores_path}")
        print(f"   ⚠️  If RECOMPUTE_SCORES=False, these pre-computed scores will be used!")
        print(f"   These scores were computed with a different (or no) autoencoder.")
        
        # Load and inspect a few scores
        scores = kaldiio.load_scp(scores_path)
        test_key = list(scores.keys())[0]
        score_array = scores[test_key]
        print(f"\n   Sample key: {test_key}")
        print(f"   Score array shape: {score_array.shape}")
        print(f"   Score types available: {score_array.shape[0]}")
        print(f"   Frames: {score_array.shape[1]}")
    else:
        print(f"   ✓ No scores.scp found (will be computed on-the-fly)")
    
    # 5. Final recommendations
    print(f"\n" + "=" * 80)
    print(f"📝 RECOMMENDATIONS:")
    print(f"=" * 80)
    
    if avg_cos_sim < 0.999:
        print(f"\n⚠️  The autoencoder is transforming d-vectors (avg cos_sim = {avg_cos_sim:.8f})")
        print(f"   For a true identity mapping, expect cos_sim > 0.999")
        print(f"   Possible causes:")
        print(f"   - Wrong model loaded (extraction instead of identity)")
        print(f"   - Identity training not fully converged")
        print(f"   - Model trained with noise may transform clean inputs differently")
    
    if Path(scores_path).exists():
        print(f"\n⚠️  Pre-computed scores.scp exists!")
        print(f"   If evaluating with RECOMPUTE_SCORES=False:")
        print(f"   - Results will use these old scores (computed without your AE)")
        print(f"   - Identity AE will appear to have no effect!")
        print(f"\n   ✓ Solution: Use RECOMPUTE_SCORES=True (already set as default)")
    
    print(f"\n✅ To verify identity AE is truly identity-like:")
    print(f"   1. Check avg cosine similarity > 0.999")
    print(f"   2. Use RECOMPUTE_SCORES=True")
    print(f"   3. Compare results with NO_AE baseline")
    print(f"   4. Results should be very similar (within 1-2% accuracy)")
    
    print(f"\n" + "=" * 80)


if __name__ == '__main__':
    main()
