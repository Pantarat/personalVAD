# Overlapping Utterances Feature - Implementation Summary

## Overview

I've successfully added the ability to generate overlapping utterances data to your PersonalVAD project. This feature allows you to:

1. **Generate overlapping speech**: Mix multiple speakers with configurable overlap percentage
2. **Control amplitude levels**: Adjust the relative loudness of overlapped speech
3. **Proper label generation**: Automatically label overlap regions as TSS (target speaker) or NTSS (non-target speaker)
4. **Visualize samples**: Play and visualize generated overlapping audio
5. **Integrate with pipeline**: Compatible with existing augmentation and feature extraction

## Files Added

### 1. **src/generate_overlapping_utterances.py** ⭐ Main Generation Script
- Generates concatenated utterances with overlapping speech
- Configurable parameters:
  - `--overlap_pct`: Percentage of overlap (0-100%, default: 30%)
  - `--amplitude_ratio`: Amplitude of overlapped speech (0-1, default: 0.7)
  - `--count`: Number of utterances to generate
- Saves metadata files with overlap information
- Compatible with existing Kaldi pipeline

### 2. **src/play_overlap_sample.py** 🎵 Visualization Tool
- Load and play overlapping audio samples
- Visualize waveform with highlighted overlap regions
- Display metadata (speakers, times, amplitudes)
- Save plots to file
- Interactive playback control

### 3. **prepare_overlap_dataset.sh** 🚀 Full Pipeline Script
- End-to-end dataset preparation
- Stages:
  - Stage 0: Generate overlapping concatenations
  - Stage 1-2: Apply augmentation (reverb, noise, music, babble)
  - Stage 3: Extract features
- Configurable settings at top of file
- Color-coded terminal output

### 4. **src/extract_features_overlap.py** 🔧 Enhanced Feature Extraction
- Extends existing feature extraction for overlaps
- Proper label handling:
  - TSS (2): Target speaker present in main OR overlap
  - NTSS (1): Speech present, but not target speaker
  - NS (0): Silence
- Parses overlap markers from transcripts
- Compatible with PersonalVAD training

### 5. **doc/OVERLAP_GENERATION.md** 📚 Complete Documentation
- Quick start guide
- Detailed parameter descriptions
- Usage examples
- Troubleshooting guide
- Technical details
- Best practices

### 6. **example_overlap_generation.py** 🎯 Quick Example
- Interactive demonstration script
- Generates 50 sample utterances
- Shows metadata
- Visualizes sample
- Guides next steps

## How It Works

### Generation Process

```
┌─────────────────┐
│ 1. Select Main  │  Pick 1-3 speakers, concatenate utterances
│    Utterance    │  Duration: 2-10 seconds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Select       │  Pick 1-2 additional utterances
│    Overlap      │  Duration: 0.5-3.0 seconds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Position     │  Randomly place overlap in main utterance
│    Overlap      │  Ensure minimum/maximum duration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Scale        │  Multiply overlap audio by amplitude_ratio
│    Amplitude    │  Default: 0.7 (70% of original volume)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Mix Audio    │  final[start:end] = main + overlap_scaled
│                 │  Clip to [-1.0, 1.0] to prevent distortion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Generate     │  Save: .flac audio + .overlap_meta metadata
│    Files        │  Update: wav.scp, utt2spk, text
└─────────────────┘
```

### Label Generation Logic

```python
For each 10ms frame:
    if silence:
        label = NS (0)
    elif overlap_region:
        if target_speaker_in_main OR target_speaker_in_overlap:
            label = TSS (2)
        else:
            label = NTSS (1)
    else:  # no overlap
        if target_speaker_speaking:
            label = TSS (2)
        else:
            label = NTSS (1)
```

## Quick Start

### 1. Generate Sample Dataset

```bash
# Generate 50 overlapping utterances (quick test)
python example_overlap_generation.py
```

### 2. Generate Full Dataset with Augmentation

```bash
# Generate 200 utterances with 30% overlap and 0.7x amplitude
bash prepare_overlap_dataset.sh 0
```

### 3. Listen to Samples

```bash
# Play random sample
python src/play_overlap_sample.py --overlap_dir data/overlap

# Save visualization
python src/play_overlap_sample.py --overlap_dir data/overlap --save_plot sample.png
```

### 4. Train Model

```bash
# Use extracted features for training
python src/personal_vad.py --train_data data/overlap_30pct_200 --epochs 10
```

## Configuration

Edit `prepare_overlap_dataset.sh`:

```bash
# Overlap settings
OVERLAP_PERCENTAGE=30      # 0-100: time with overlap
OVERLAP_AMPLITUDE=0.7      # 0-1: relative amplitude

# Standard settings
AUGMENT=true               # Apply augmentation
utt_count=200              # Number of utterances
feature_dir_name=overlap_30pct_200  # Output directory
```

## Example Output

### Generated Files

```
data/overlap/
├── 0_overlap/
│   ├── 1234-5678-0001_9012-3456-0002_OV0.flac
│   ├── 1234-5678-0001_9012-3456-0002_OV0.overlap_meta
│   └── ...
├── wav.scp
├── utt2spk
└── text
```

### Metadata File

```
main_speakers: 1234,5678
overlap_count: 1
overlap_0: 1.50-2.80 speakers=9012 amplitude=0.70
```

### Visualization

The play script shows:
- Blue waveform line
- Red/orange highlighted overlap regions
- Speaker IDs and amplitudes
- Time markers

## Key Features

✅ **Configurable overlap percentage** (0-100%)  
✅ **Adjustable amplitude ratio** (0-1)  
✅ **Multiple overlaps per utterance** (1-2)  
✅ **Random positioning**  
✅ **Proper TSS/NTSS labeling**  
✅ **Metadata tracking**  
✅ **Compatible with augmentation**  
✅ **Visualization tools**  
✅ **Audio playback**  

## Integration with Existing Pipeline

The overlap generation seamlessly integrates:

```
Standard Pipeline:          Overlap Pipeline:
─────────────────          ──────────────────

concatenate_utterances.py → generate_overlapping_utterances.py
         ↓                            ↓
   reverberate              →   reverberate
   augment                      augment
         ↓                            ↓
   extract_features         →   extract_features_overlap.py
         ↓                            ↓
   PersonalVAD training     →   PersonalVAD training
```

## Parameters Reference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `overlap_pct` | 0-100 | 30 | Percentage of time with overlap |
| `amplitude_ratio` | 0-1 | 0.7 | Relative amplitude of overlap |
| `utt_count` | >0 | 200 | Number of utterances |
| `min_overlap_dur` | >0 | 0.5s | Minimum overlap duration |
| `max_overlap_dur` | >0 | 3.0s | Maximum overlap duration |
| `n_overlaps` | 1-2 | random | Number of overlaps per utterance |

## Best Practices

1. **Start with 20-30% overlap** for balanced datasets
2. **Use 0.6-0.8 amplitude** for realistic mixing
3. **Generate 500+ utterances** for good coverage
4. **Always apply augmentation** for robustness
5. **Listen to samples** to verify quality
6. **Check metadata files** to understand overlap distribution

## Troubleshooting

### "No audio files found"
- Ensure LibriSpeech datasets are downloaded
- Check path in `--libri_root` argument

### "Feature extraction failed"
- Verify Kaldi is set up correctly
- Check `KALDI_HOME` environment variable
- Ensure embeddings exist in `data/embeddings/`

### "Import errors"
- Install dependencies: `pip install numpy librosa soundfile matplotlib`
- For audio playback: `pip install sounddevice`

### "Labels seem incorrect"
- Check overlap metadata files
- Verify target speaker is tracked correctly
- Review transcript with overlap markers

## Technical Details

### Overlap Insertion Algorithm

1. Generate main utterance (concatenate 1-3 speakers)
2. Select overlap utterances (1-2 additional speakers)
3. Determine overlap duration: `random.uniform(0.5, 3.0)` seconds
4. Choose random position: `random.randint(0, main_duration - overlap_duration)`
5. Scale amplitude: `overlap_audio *= amplitude_ratio`
6. Mix: `final[start:end] += overlap_audio[0:duration]`
7. Clip: `final = np.clip(final, -1.0, 1.0)`

### Feature Compatibility

- Same 40-dim log mel-filterbanks
- Same 256-dim d-vectors
- Same cosine similarity scores
- Same 10ms frame alignment
- Compatible with all model architectures (ET/ST/SET)

## Next Steps

1. **Test with small dataset**: Run `example_overlap_generation.py`
2. **Generate full dataset**: Run `prepare_overlap_dataset.sh 0`
3. **Visualize samples**: Use `play_overlap_sample.py`
4. **Train model**: Use extracted features
5. **Evaluate performance**: Compare with non-overlap baseline

## Support Files

- `doc/OVERLAP_GENERATION.md`: Full documentation
- `example_overlap_generation.py`: Quick demo
- `prepare_overlap_dataset.sh`: Full pipeline
- `src/generate_overlapping_utterances.py`: Core generation
- `src/play_overlap_sample.py`: Visualization
- `src/extract_features_overlap.py`: Feature extraction

## Summary

You now have a complete system for generating and training on overlapping speech data! The implementation:

✅ Generates realistic overlapping utterances  
✅ Provides configurable parameters  
✅ Labels data correctly (TSS/NTSS/NS)  
✅ Integrates with existing pipeline  
✅ Includes visualization tools  
✅ Has comprehensive documentation  
✅ Supports audio playback  

Start with the example script, then scale up to full datasets!
