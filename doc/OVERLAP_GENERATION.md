# Overlapping Utterances Generation

This extension adds the ability to generate and process overlapping speech utterances for the PersonalVAD system.

## Features

- **Configurable Overlap**: Control the percentage of overlapping speech (0-100%)
- **Amplitude Control**: Adjust the relative amplitude of overlapped utterances (0-1)
- **Proper Labeling**: Automatically labels overlaps as TSS (target speaker) or NTSS (non-target speaker)
- **Compatible Pipeline**: Works with existing augmentation and feature extraction
- **Visualization**: Built-in tools to visualize and play sample overlapping audio

## Quick Start

### 1. Generate Overlapping Dataset

```bash
# Generate 200 overlapping utterances with 30% overlap and 0.7x amplitude
bash prepare_overlap_dataset.sh 0
```

This will:
- Generate overlapping concatenated utterances
- Apply augmentation (reverb, noise, music, babble) if enabled
- Extract features compatible with PersonalVAD training

### 2. Configuration

Edit `prepare_overlap_dataset.sh` to customize:

```bash
# Overlap settings
OVERLAP_PERCENTAGE=30     # 0-100: amount of time with overlap
OVERLAP_AMPLITUDE=0.7     # 0-1: relative amplitude of overlapped speech

# Standard settings
AUGMENT=true              # Apply reverb/noise augmentation
utt_count=200             # Number of utterances to generate
feature_dir_name=overlap_30pct_200  # Output feature directory name
```

### 3. Play Sample Audio

```bash
# Play a random sample
python src/play_overlap_sample.py --overlap_dir data/overlap

# Play specific sample
python src/play_overlap_sample.py --overlap_dir data/overlap --sample_idx 5

# Just visualize, no audio
python src/play_overlap_sample.py --overlap_dir data/overlap --no_play

# Save plot
python src/play_overlap_sample.py --overlap_dir data/overlap --save_plot overlap_sample.png
```

## How It Works

### 1. Overlap Generation

The `generate_overlapping_utterances.py` script:

1. **Creates Main Utterance**: Concatenates 1-3 speaker utterances (like standard pipeline)
2. **Adds Overlaps**: Inserts 1-2 overlapping utterances at random positions
3. **Configurable Duration**: Overlap duration between 0.5-3.0 seconds
4. **Amplitude Scaling**: Scales overlapped audio by `OVERLAP_AMPLITUDE` ratio
5. **Audio Mixing**: Mixes overlapped segment with main audio (with clipping)
6. **Metadata Tracking**: Saves overlap regions, speakers, and parameters

### 2. Label Generation

Labels are generated based on speaker presence:

- **TSS (Class 2)**: Target speaker is present in either main OR overlap
- **NTSS (Class 1)**: Speech present but target speaker is NOT in overlap region
- **NS (Class 0)**: Silence (no speech)

Example:
```
Time:    0s        1s        2s        3s        4s
Main:    [Speaker A -------> silence --------]
Overlap:           [Speaker B ----]
Target:  A

Labels:  TSS       NTSS      NS
```

### 3. File Structure

Generated files include:

```
data/overlap/
├── 0_overlap/
│   ├── 1234-5678-0001_9012-3456-0002_OV0.flac      # Audio file
│   ├── 1234-5678-0001_9012-3456-0002_OV0.overlap_meta  # Metadata
│   └── ...
├── 1_overlap/
│   └── ...
├── wav.scp          # Kaldi wav script
├── utt2spk          # Utterance to speaker mapping
└── text             # Transcripts with overlap markers
```

**Metadata File Format:**
```
main_speakers: 1234,5678
overlap_count: 1
overlap_0: 1.50-2.80 speakers=9012 amplitude=0.70
```

## Advanced Usage

### Manual Generation

You can generate overlapping utterances manually:

```bash
python src/generate_overlapping_utterances.py \
    --libri_root data/LibriSpeech \
    --concat_dir /path/to/output \
    --count 500 \
    --overlap_pct 40 \
    --amplitude_ratio 0.5 \
    test-clean test-other
```

### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `--overlap_pct` | 0-100 | Percentage of time with overlapping speech |
| `--amplitude_ratio` | 0-1 | Relative amplitude of overlapped utterance |
| `--count` | >0 | Number of utterances to generate |
| `--scp_prefix` | path | Path prefix for wav.scp entries |

### Integration with Augmentation

Overlapping utterances can be augmented with reverb and noise just like standard utterances:

```bash
# Stage 0: Generate overlaps
bash prepare_overlap_dataset.sh 0

# Stage 1-2: Apply augmentation
bash prepare_overlap_dataset.sh 1
bash prepare_overlap_dataset.sh 2

# Stage 3: Extract features
bash prepare_overlap_dataset.sh 3
```

## Training with Overlap Data

After feature extraction, train your PersonalVAD model:

```bash
# Use the generated features
python src/personal_vad.py \
    --train_data data/overlap_30pct_200 \
    --model_name vad_set_overlap \
    --epochs 10
```

## Visualization Examples

The `play_overlap_sample.py` script generates waveform plots showing:

- **Blue waveform**: Full audio
- **Colored regions**: Overlap segments (with speaker info)
- **Annotations**: Overlap index and amplitude ratio

Example output:
```
Found 200 overlapping utterances
Randomly selected sample #42

Loading: data/overlap/0_overlap/1234-5678-0001_9012-3456-0002_OV0.flac
Duration: 4.32 seconds
Sample rate: 16000 Hz

📊 Metadata:
  Main speakers: 1234, 5678
  Overlap count: 1

  Overlap 1:
    Time: 1.50s - 2.80s (duration: 1.30s)
    Speakers: 9012
    Amplitude: 0.70x
```

## Tips & Best Practices

1. **Start Small**: Begin with 20-30% overlap to maintain label balance
2. **Amplitude Ratio**: Use 0.6-0.8 for realistic overlap (not too quiet)
3. **Dataset Size**: Generate at least 500 utterances for good coverage
4. **Augmentation**: Always use augmentation for robustness
5. **Validation**: Listen to samples to verify quality

## Troubleshooting

### Issue: No audio playback

**Solution**: Install sounddevice
```bash
pip install sounddevice
```

### Issue: Import errors

**Solution**: Ensure all dependencies are installed
```bash
pip install numpy librosa soundfile matplotlib
```

### Issue: Labels seem incorrect

**Solution**: Check that target speaker is properly tracked in overlap regions. The system defaults to NTSS for overlaps unless explicitly marked.

### Issue: Audio too quiet/loud

**Solution**: Adjust `OVERLAP_AMPLITUDE` in `prepare_overlap_dataset.sh`:
- Too quiet: Increase to 0.8-0.9
- Too loud: Decrease to 0.5-0.6

## Technical Details

### Overlap Insertion Algorithm

1. Generate main concatenated utterance (1-3 speakers)
2. Select 1-2 additional utterances for overlap
3. Choose random position in main utterance
4. Determine overlap duration (0.5-3.0s)
5. Scale overlap amplitude by ratio
6. Mix: `final[start:end] = main[start:end] + overlap[0:duration] * ratio`
7. Clip: `final = clip(final, -1.0, 1.0)`

### Label Priority Rules

When multiple speech sources overlap:
- Target speaker present (main OR overlap) → TSS
- Only non-target speakers → NTSS
- No speech → NS

### Feature Extraction

Uses standard PersonalVAD features:
- 40-dimensional log mel-filterbanks
- 256-dimensional d-vectors
- Cosine similarity scores
- Proper alignment with 10ms frames

## Future Enhancements

Potential improvements:
- [ ] Variable overlap amplitude per region
- [ ] Multiple overlaps with different speakers
- [ ] Overlap-specific data augmentation
- [ ] Configurable overlap position (start/middle/end)
- [ ] Support for 3+ speaker overlaps
- [ ] Overlap-aware training strategies

## References

- Original PersonalVAD: [Paper](https://arxiv.org/abs/1908.04284)
- LibriSpeech Dataset: [Website](https://www.openslr.org/12)
- Kaldi Toolkit: [Documentation](https://kaldi-asr.org/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review generated metadata files
3. Visualize samples with `play_overlap_sample.py`
4. Verify feature extraction completed successfully
