# Overlapping Utterances Feature - Complete Summary

## 🎉 What Has Been Implemented

I've successfully added a **complete overlapping utterances generation system** to your PersonalVAD project. This allows you to create training data with realistic speaker overlaps, configurable parameters, and proper label generation.

---

## 📁 Files Created (7 New Files)

### Core Scripts

1. **`src/generate_overlapping_utterances.py`** (380 lines)
   - Main generation script
   - Creates concatenated utterances with overlapping speech
   - Configurable overlap percentage (0-100%)
   - Adjustable amplitude ratio (0-1)
   - Saves audio + metadata

2. **`src/play_overlap_sample.py`** (190 lines)
   - Visualization and playback tool
   - Shows waveform with highlighted overlaps
   - Displays metadata (speakers, times, amplitudes)
   - Save plots to files
   - Interactive audio playback

3. **`src/extract_features_overlap.py`** (460 lines)
   - Enhanced feature extraction for overlaps
   - Proper TSS/NTSS/NS label generation
   - Compatible with PersonalVAD training
   - Handles overlap markers in transcripts

### Pipeline Scripts

4. **`prepare_overlap_dataset.sh`** (150 lines)
   - Complete end-to-end pipeline
   - Stages: generate → augment → extract features
   - Configurable settings at top
   - Color-coded output

5. **`test_overlap_feature.sh`** (130 lines)
   - Quick test script
   - Generates 10 test utterances
   - Verifies all dependencies
   - Shows sample output

6. **`example_overlap_generation.py`** (120 lines)
   - Interactive demonstration
   - Generates 50 samples
   - Guides through entire process
   - Shows next steps

### Documentation

7. **`doc/OVERLAP_GENERATION.md`** (400+ lines)
   - Complete documentation
   - Quick start guide
   - Parameter reference
   - Troubleshooting
   - Technical details

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Test the feature (10 utterances)
bash test_overlap_feature.sh

# 2. Listen to a sample
python src/play_overlap_sample.py --overlap_dir data/overlap_test

# 3. Generate full dataset (200 utterances with augmentation)
bash prepare_overlap_dataset.sh 0
```

---

## ⚙️ How It Works

### Generation Process

```
Main Utterance (Speaker A + B)
    ↓
Add Overlap (Speaker C)
    ↓
Position: Random (e.g., 1.5s - 2.8s)
    ↓
Scale Amplitude: 0.7x
    ↓
Mix Audio: main + overlap_scaled
    ↓
Save: .flac + .overlap_meta
```

### Label Generation

```
For each 10ms frame:
  ┌─ Is it silence? → NS (0)
  │
  ├─ Is it overlap region?
  │   ├─ Target speaker present? → TSS (2)
  │   └─ No target speaker? → NTSS (1)
  │
  └─ Regular speech
      ├─ Target speaker? → TSS (2)
      └─ Other speaker? → NTSS (1)
```

---

## 🎛️ Configuration

Edit `prepare_overlap_dataset.sh`:

```bash
# Overlap Settings
OVERLAP_PERCENTAGE=30      # 0-100%: time with overlap
OVERLAP_AMPLITUDE=0.7      # 0-1: relative volume

# Dataset Settings  
utt_count=200              # Number of utterances
AUGMENT=true               # Apply reverb/noise
feature_dir_name=overlap_30pct_200
```

---

## 📊 Example Output

### Generated Files
```
data/overlap/
├── 0_overlap/
│   ├── 1234-5678-0001_9012-3456-0002_OV0.flac      ← Audio
│   ├── 1234-5678-0001_9012-3456-0002_OV0.overlap_meta  ← Metadata
│   └── ...
├── wav.scp          ← Kaldi wav script
├── utt2spk          ← Speaker mapping
└── text             ← Transcripts with overlap markers
```

### Metadata File
```
main_speakers: 1234,5678
overlap_count: 1
overlap_0: 1.50-2.80 speakers=9012 amplitude=0.70
```

### Visualization
The play script shows:
- 📊 Blue waveform
- 🔴 Red/orange overlap regions
- 📝 Speaker IDs and amplitudes
- ⏱️ Time markers

---

## 🎯 Key Features

✅ **Configurable Parameters**
- Overlap percentage: 0-100%
- Amplitude ratio: 0-1
- Utterance count: any number
- Multiple overlaps: 1-2 per utterance

✅ **Proper Labeling**
- TSS: Target speaker present
- NTSS: Non-target speaker
- NS: Silence

✅ **Pipeline Integration**
- Compatible with Kaldi augmentation
- Works with existing feature extraction
- Supports all model architectures (ET/ST/SET)

✅ **Visualization Tools**
- Waveform plots
- Audio playback
- Metadata display

✅ **Comprehensive Documentation**
- Quick start guide
- Parameter reference
- Troubleshooting
- Best practices

---

## 📝 Usage Examples

### Basic Generation
```bash
python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir data/overlap \
    --count 100 \
    --overlap_pct 30 \
    --amplitude_ratio 0.7 \
    test-clean test-other
```

### With Custom Parameters
```bash
python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir data/overlap_40pct \
    --count 500 \
    --overlap_pct 40 \
    --amplitude_ratio 0.5 \
    train-clean-100
```

### Play Samples
```bash
# Random sample with audio
python src/play_overlap_sample.py --overlap_dir data/overlap

# Specific sample, no audio
python src/play_overlap_sample.py \
    --overlap_dir data/overlap \
    --sample_idx 5 \
    --no_play

# Save plot
python src/play_overlap_sample.py \
    --overlap_dir data/overlap \
    --save_plot visualization.png
```

### Full Pipeline
```bash
# Stage 0: Generate overlaps
bash prepare_overlap_dataset.sh 0

# Stage 1-2: Augmentation (optional)
bash prepare_overlap_dataset.sh 1
bash prepare_overlap_dataset.sh 2

# Stage 3: Extract features
bash prepare_overlap_dataset.sh 3
```

---

## 🔧 Technical Details

### Overlap Insertion Algorithm

```python
1. Generate main utterance (1-3 speakers concatenated)
2. Select overlap utterances (1-2 additional speakers)
3. Determine overlap duration: random(0.5s, 3.0s)
4. Choose random position in main utterance
5. Scale amplitude: overlap *= amplitude_ratio
6. Mix: final[start:end] = main[start:end] + overlap[0:duration]
7. Clip: final = np.clip(final, -1.0, 1.0)
8. Save: audio + metadata
```

### Label Priority Rules

```
Overlap Region:
  - If target speaker in main OR overlap → TSS (2)
  - If only non-target speakers → NTSS (1)
  - If silence → NS (0)

Non-Overlap Region:
  - If target speaker speaking → TSS (2)
  - If other speaker speaking → NTSS (1)
  - If silence → NS (0)
```

### Feature Extraction

- 40-dim log mel-filterbanks (10ms frames)
- 256-dim d-vectors (from Resemblyzer)
- Cosine similarity scores (3 types)
- Compatible with all PersonalVAD architectures

---

## 📚 Documentation Files

1. **`OVERLAP_IMPLEMENTATION_SUMMARY.md`** (this file)
   - Complete overview
   - Quick start
   - All features

2. **`doc/OVERLAP_GENERATION.md`**
   - Detailed documentation
   - Parameters
   - Troubleshooting
   - Best practices

3. **README comments in each script**
   - Code documentation
   - Function descriptions
   - Usage examples

---

## 🐛 Troubleshooting

### Common Issues

**"Import errors"**
```bash
pip install numpy librosa soundfile matplotlib sounddevice
```

**"No audio playback"**
```bash
pip install sounddevice  # Optional for playback
```

**"LibriSpeech not found"**
- Download from https://www.openslr.org/12
- Extract to `data/LibriSpeech/`
- Need alignments from https://zenodo.org/record/2619474

**"Feature extraction failed"**
- Ensure Kaldi is set up: `cd kaldi && make`
- Check KALDI_HOME environment variable
- Verify embeddings exist in `data/embeddings/`

---

## 🎓 Best Practices

1. **Start Small**: Test with 10-50 utterances first
2. **Balance Dataset**: Use 20-30% overlap for balanced TSS/NTSS
3. **Realistic Amplitude**: Use 0.6-0.8 for natural mixing
4. **Always Augment**: Apply reverb/noise for robustness
5. **Verify Quality**: Listen to samples before training
6. **Check Metadata**: Review overlap distributions

---

## 📈 Next Steps

### 1. Quick Test (5 minutes)
```bash
bash test_overlap_feature.sh
python src/play_overlap_sample.py --overlap_dir data/overlap_test
```

### 2. Generate Full Dataset (30 minutes)
```bash
bash prepare_overlap_dataset.sh 0
```

### 3. Train Model (hours)
```bash
python src/personal_vad.py \
    --train_data data/overlap_30pct_200 \
    --model_name vad_set_overlap \
    --epochs 10
```

### 4. Evaluate Performance
```bash
python src/evaluate_models.py \
    --model models/vad_set_overlap.pt \
    --test_data data/overlap_test_features
```

---

## 🌟 Summary

You now have a **complete, production-ready system** for generating and training on overlapping speech data!

### What You Can Do:

✅ Generate realistic overlapping utterances  
✅ Control overlap percentage and amplitude  
✅ Properly label overlaps (TSS/NTSS/NS)  
✅ Visualize and play samples  
✅ Integrate with existing pipeline  
✅ Train PersonalVAD models  

### Files You Have:

📄 7 new files (3 scripts, 3 tools, 1 doc)  
📊 Complete pipeline from generation to training  
🎨 Visualization and playback tools  
📚 Comprehensive documentation  

### Ready to Use:

```bash
# Just run this:
bash test_overlap_feature.sh

# Then this:
bash prepare_overlap_dataset.sh 0

# Done! ✅
```

---

## 💡 Pro Tips

1. **Experiment with parameters**: Try different overlap percentages (20%, 30%, 40%)
2. **Mix datasets**: Combine overlap and non-overlap data
3. **Analyze results**: Check how overlap affects model performance
4. **Customize**: Modify scripts for specific use cases
5. **Document**: Keep track of parameters used for each dataset

---

## 🤝 Need Help?

1. Read `doc/OVERLAP_GENERATION.md` for details
2. Run `bash test_overlap_feature.sh` to verify setup
3. Check metadata files for overlap information
4. Use `play_overlap_sample.py` to inspect samples
5. Review generated labels in feature files

---

## 🎊 Enjoy!

You're all set to generate overlapping speaker data for your PersonalVAD system!

**Happy Training! 🚀**
