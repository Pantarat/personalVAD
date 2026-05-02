# Clean vs Noise vs Mixed Embedding Comparison Tests

This test suite compares speaker embeddings (d-vectors and optionally x-vectors) extracted from clean audio, noise audio, and mixed audio at various SNR levels. The tests evaluate reconstruction quality when passing embeddings through autoencoders.

## Scripts Overview

### 1. `test_clean_noise_mixed_comparison.py`
**Main test script for d-vector comparison with autoencoder reconstruction.**

Evaluates:
- Single clean utterance d-vector
- Single noise utterance d-vector  
- Mixed (clean+noise) d-vector at SNR levels: (20dB, 17dB, 15dB, 13dB)
- AE reconstruction quality for each via cosine similarity & MSE
- Leakage detection (how much mixed reconstruction resembles original clean/noise)

Outputs:
- Console metrics for each SNR level
- PNG comparison plots saved to `src/AE_test/test_outputs/clean_noise_mixed_comparison.png`

**Usage:**
```bash
python src/AE_test/tests/test_clean_noise_mixed_comparison.py
```

### 2. `test_clean_noise_mixed_comparison_full.py`
**Extended test script with optional x-vector support.**

Same as above, plus infrastructure for x-vector extraction via Kaldi (if available).

**Usage (d-vectors only):**
```bash
python src/AE_test/tests/test_clean_noise_mixed_comparison_full.py
```

**Usage (d-vectors + x-vectors):**
```bash
# First, edit the script to set:
# - XVECTOR_MODEL_DIR to your Kaldi x-vector model path
# - KALDI_ROOT if not in default location

python src/AE_test/tests/test_clean_noise_mixed_comparison_full.py --with-xvector
```

### 3. `xvector_utils.py`
**Utility module for x-vector extraction via Kaldi.**

Provides `KaldiXVectorExtractor` class for:
- Extracting x-vectors from WAV files using Kaldi nnet3-xvector-compute
- Batch x-vector extraction
- Cosine similarity computation

## Configuration

Edit the settings section at the top of each test script:

```python
# Autoencoder model directory
AE_MODEL_DIR = 'src/AE_test/test_outputs/models/dvector_ae_deep_stacked_libri_babble_2_29-4-26'

# SNR levels for mixing (matches training data)
SNR_RANGE_DB = (20.0, 17.0, 15.0, 13.0)

# Audio duration and extraction parameters
DURATION_SEC = 5.0
WINDOW_SEC = 1.6
WINDOW_STRIDE_SEC = 0.4

# Data paths
LIBRISPEECH_ROOT = 'data/LibriSpeech'
MUSAN_NOISE_DIR = 'kaldi/egs/pvad/musan/noise/free-sound'

# X-vector model (for full script)
XVECTOR_MODEL_DIR = None  # Set to enable x-vector extraction
```

## Output Metrics

### Per SNR Level:

1. **Reconstruction Similarity (Cosine):**
   - Clean orig → clean recon
   - Noise orig → noise recon
   - Mixed orig → mixed recon
   
2. **Reconstruction Error (MSE):**
   - Mean squared error for each comparison above

3. **Leakage Detection (Cosine):**
   - Mixed reconstruction → original clean (should be low)
   - Mixed reconstruction → original noise (should be low)

### Plot (4 subplots):

1. **Top-left:** Reconstruction cosine similarity vs SNR
2. **Top-right:** Leakage detection (mixed recon similarity to originals)
3. **Bottom-left:** Reconstruction MSE vs SNR
4. **Bottom-right:** Summary statistics table

## Interpretation Guide

### Good Reconstruction:
- Cosine similarity close to 1.0 (typically 0.95+)
- MSE close to 0 (depends on embedding dimension, typically 0.01-0.1)

### No Leakage:
- Mixed reconstruction has low similarity to original clean/noise (ideally <0.5)
- Indicates AE isn't just reproducing input modality

### SNR Impact:
- Results may vary with SNR due to noise characteristics
- Typically, lower SNR (more noise) → lower reconstruction quality

## Setting up X-Vector Support

### Prerequisites:
1. **Kaldi compiled:** Ensure `kaldi/` directory with compiled binaries
2. **X-vector model:** Pre-trained model directory with:
   - `final.nnet` (neural network model)
   - `cmvn.opts` (CMVN normalization options)
   - Other standard Kaldi model files

3. **Python dependencies:**
   ```bash
   pip install kaldiio  # For reading Kaldi binary formats
   ```

### Configuration:
Edit the script to set paths:
```python
KALDI_ROOT = 'kaldi'  # Path to Kaldi root
XVECTOR_MODEL_DIR = 'kaldi/egs/voxceleb/v2/exp/xvector_nnet_1a'  # Your model
```

### Known Limitations:
- X-vector extraction currently requires full Kaldi setup (not Python-only)
- The `xvector_utils.py` module provides framework; feature/x-vector computation integration requires Kaldi subprocess calls
- For production use, integrate with Kaldi via:
  - `compute-mfcc-feats` for feature extraction
  - `nnet3-xvector-compute` for x-vector extraction
  - `ivector-compute-mean` for averaging

## File Locations

```
src/AE_test/tests/
  ├── test_clean_noise_mixed_comparison.py       [Main d-vector test]
  ├── test_clean_noise_mixed_comparison_full.py  [Extended with x-vector support]
  └── (existing test files)

src/AE_test/
  ├── xvector_utils.py                           [X-vector utilities]
  ├── autoencoder_utils.py                       [AE loading]
  ├── dvector_utils.py                           [D-vector utilities]
  └── (other AE training/utility scripts)

src/AE_test/test_outputs/
  └── clean_noise_mixed_comparison.png           [Generated plots]
```

## Extending the Tests

### Add Custom Metrics:
Modify `dvector_results` dictionary structure to track additional metrics:
```python
dvector_results[snr_db] = {
    'clean_cos': ...,
    # Add custom metrics here
    'my_metric': compute_my_metric(...),
}
```

### Test Different Models:
Change `AE_MODEL_DIR` to test different autoencoder checkpoints:
```python
# Compare multiple models by running script multiple times with different dirs
AE_MODEL_DIR = 'src/AE_test/test_outputs/models/model_v1'
# vs
AE_MODEL_DIR = 'src/AE_test/test_outputs/models/model_v2'
```

### Use Different Audio Sources:
Change `LIBRISPEECH_SUBSET` or `MUSAN_NOISE_DIR` to test with different audio:
```python
LIBRISPEECH_SUBSET = 'test-other'  # More challenging test set
MUSAN_NOISE_DIR = 'kaldi/egs/pvad/musan/music'  # Different noise type
```

## Troubleshooting

### ImportError for xvector_utils:
- X-vector support is optional; d-vector tests work without it
- To enable x-vector: `pip install kaldiio`

### Audio loading errors:
- Ensure LibriSpeech and MUSAN paths exist and are readable
- Check audio file formats (FLAC for LibriSpeech, WAV for MUSAN)

### Model not found:
- Verify `AE_MODEL_DIR` path exists and contains `config.pkl` and model checkpoint
- Support checkpoint names: `final_model.pth`, `best_model.pth`

### Out of memory:
- Reduce `BATCH_SIZE` (default: 512)
- Reduce `N_SAMPLES_EACH` if applicable
- Enable `FORCE_CPU = True` if only CPU available

## References

- **D-vectors:** ResemBlyzer VoiceEncoder (256-dim speaker embeddings)
- **X-vectors:** Kaldi nnet3-based speaker embeddings
- **SNR calculation:** Standard power-based SNR = 10 * log10(P_signal / P_noise)
- **Autoencoder:** Deep stacked DAE for embedding denoising
