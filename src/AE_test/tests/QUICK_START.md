# Quick Start Guide: Clean/Noise/Mixed Embedding Comparison

## 30-Second Setup

```bash
# 1. Run the main d-vector test (no setup required)
cd c:\Work\Coding\Diarization\PersonalVAD\personalVAD
python src/AE_test/tests/test_clean_noise_mixed_comparison.py

# 2. Check output in src/AE_test/test_outputs/clean_noise_mixed_comparison.png
```

## What Gets Tested

For **each SNR level** (20dB, 17dB, 15dB, 13dB):

```
Clean Audio File (e.g., LibriSpeech dev-clean speaker utterance)
    ↓
Extract d-vectors (ResemBlyzer)
    ↓ Reconstruct via AE ↓
Measure: Cosine Similarity ≈ 0.95+, MSE ≈ 0.05

Noise Audio File (e.g., MUSAN babble)
    ↓
Extract d-vectors
    ↓ Reconstruct via AE ↓
Measure: Cosine Similarity ≈ 0.90+, MSE ≈ 0.10

Clean + Noise (mixed at SNR)
    ↓
Extract d-vectors
    ↓ Reconstruct via AE ↓
Measure: Cosine Similarity ≈ 0.92+, MSE ≈ 0.07

LEAKAGE CHECK:
Mixed Reconstruction → Original Clean: Should be LOW (< 0.5)
Mixed Reconstruction → Original Noise: Should be LOW (< 0.5)
```

## Example Output

```
Device: cuda

Loading sample audio files...
✅ Clean file: 1455-127105-0029.flac
✅ Noise file: SNR_0_babble_08_003.wav
✅ Clean audio: 5.00s
✅ Noise audio: 5.00s

================================================================================
D-VECTOR COMPARISON: CLEAN vs NOISE vs MIXED
================================================================================

Extracting d-vectors from clean audio...
  Clean d-vectors extracted: 8, avg norm: 5.2341

Extracting d-vectors from noise audio...
  Noise d-vectors extracted: 8, avg norm: 4.1205

Processing mixed audio at SNR levels: (20.0, 17.0, 15.0, 13.0)

  SNR 20.0dB:
    Clean orig->recon:  cos=0.954230, mse=0.047823
    Noise orig->recon:  cos=0.893456, mse=0.102341
    Mixed orig->recon:  cos=0.924567, mse=0.068432
    Mixed recon->clean: cos=0.421234
    Mixed recon->noise: cos=0.312456

  SNR 17.0dB:
    Clean orig->recon:  cos=0.952341, mse=0.048932
    Noise orig->recon:  cos=0.891234, mse=0.104523
    Mixed orig->recon:  cos=0.921345, mse=0.071234
    Mixed recon->clean: cos=0.428901
    Mixed recon->noise: cos=0.319876

  [... more SNR levels ...]

================================================================================
PLOTTING RESULTS
================================================================================

✅ Plot saved to: src/AE_test/test_outputs/clean_noise_mixed_comparison.png
```

## Interpreting Results

### Table: What Each Metric Means

| Metric | Target Range | Interpretation |
|--------|--------------|-----------------|
| Clean orig→recon cosine | 0.92-0.98 | How well AE preserves clean d-vectors |
| Noise orig→recon cosine | 0.85-0.95 | How well AE preserves noise d-vectors |
| Mixed orig→recon cosine | 0.90-0.96 | How well AE preserves mixed d-vectors |
| Mixed recon→clean cosine | <0.5 | **Lower is better** - no leakage to clean |
| Mixed recon→noise cosine | <0.5 | **Lower is better** - no leakage to noise |

### What's Good:
✅ Clean reconstruction cosine > 0.95  
✅ Mixed reconstruction cosine > 0.90  
✅ Leakage (mixed→clean/noise) < 0.5  
✅ Results consistent across SNR levels  

### What's Bad:
❌ Clean/noise reconstruction < 0.85 (high reconstruction error)  
❌ Leakage scores > 0.7 (mixed recon too similar to originals)  
❌ Strong SNR dependency (e.g., quality drops drastically at lower SNR)  

## Plot Description

**4-subplot comparison figure:**

1. **Top-Left:** Original → Reconstructed Cosine Similarity
   - Higher is better
   - Three lines for clean, noise, mixed
   - Should be relatively flat across SNR levels

2. **Top-Right:** Leakage Detection
   - Mixed reconstruction similarity to original clean/noise
   - **Lower is better** (ideally <0.5)
   - Shows if AE is just reproducing input modality

3. **Bottom-Left:** Reconstruction Error (MSE)
   - Lower is better
   - Complements cosine similarity metric

4. **Bottom-Right:** Summary Table
   - File names, d-vector counts, L2 norms
   - Average metrics across all SNR levels

## Common Configurations

### Test with Different AE Models
```python
# In test script, change AE_MODEL_DIR to:
AE_MODEL_DIR = 'src/AE_test/test_outputs/models/model_name'

# Then run script
python src/AE_test/tests/test_clean_noise_mixed_comparison.py
```

### Test with Different Noise Types
```python
# In test script, change MUSAN_NOISE_DIR to:
MUSAN_NOISE_DIR = 'kaldi/egs/pvad/musan/music'        # Music noise
MUSAN_NOISE_DIR = 'kaldi/egs/pvad/musan/speech'      # Speech noise
MUSAN_NOISE_DIR = 'kaldi/egs/pvad/musan/nature'      # Nature noise
```

### Test with Longer/Shorter Audio
```python
# In test script:
DURATION_SEC = 10.0  # Longer
# or
DURATION_SEC = 2.0   # Shorter
```

## X-Vector Comparison (Advanced)

To also extract and compare x-vectors:

1. **Setup Kaldi x-vector model:**
   ```bash
   # Ensure you have a trained x-vector model, e.g.:
   # kaldi/egs/voxceleb/v2/exp/xvector_nnet_1a/
   ```

2. **Install kaldiio:**
   ```bash
   pip install kaldiio
   ```

3. **Edit test script:**
   ```python
   XVECTOR_MODEL_DIR = 'kaldi/egs/voxceleb/v2/exp/xvector_nnet_1a'
   KALDI_ROOT = 'kaldi'
   ```

4. **Run extended test:**
   ```bash
   python src/AE_test/tests/test_clean_noise_mixed_comparison_full.py
   ```

## Troubleshooting

### "No audio files found"
```
Error: No audio files found in data/LibriSpeech/dev-clean
→ Download LibriSpeech dataset or check paths
```

### "Missing config file"
```
Error: Missing config file: .../config.pkl
→ Check AE_MODEL_DIR points to valid model directory
```

### "CUDA out of memory"
```python
# In test script:
BATCH_SIZE = 256  # Reduce from 512
FORCE_CPU = True  # Use CPU instead
```

### "D-vectors extracted: 0"
```
→ Check WINDOW_SEC/WINDOW_STRIDE_SEC values
→ May be too large for 5-second audio
```

## Next Steps

1. **Run baseline test:** Execute script as-is to understand output format
2. **Compare models:** Run with different `AE_MODEL_DIR` settings
3. **Analyze trends:** Study how reconstruction quality varies with SNR
4. **Extend metrics:** Add custom similarity/distance measures to results dict
5. **X-vector integration:** Set up Kaldi and integrate x-vector extraction

## Files Reference

- **Main test:** `src/AE_test/tests/test_clean_noise_mixed_comparison.py`
- **Extended test:** `src/AE_test/tests/test_clean_noise_mixed_comparison_full.py`
- **X-vector utils:** `src/AE_test/xvector_utils.py`
- **Full docs:** `src/AE_test/tests/TEST_COMPARISON_README.md`
- **Output plots:** `src/AE_test/test_outputs/clean_noise_mixed_comparison.png`
