#!/usr/bin/env python3
"""
Generate Non-Target Overlapping Speech Utterances
Creates overlap samples WITHOUT the target speaker - only other speakers

This script:
1. Selects random pairs of NON-TARGET speakers from LibriSpeech
2. Creates overlapping speech by mixing their audio
3. Generates frame-level labels (all frames labeled as non-target)
4. Saves audio files and metadata for training/testing

NEW FEATURE: Base Speaker Mode
- Set BASE_SPEAKER to a speaker ID (e.g., '84') to have that speaker in ALL overlaps
- The base speaker will be the main speaker, with other speakers mixed on top
- Useful for testing how the model handles a specific speaker with various overlaps

Use case: Train model to distinguish target speaker from ANY overlap
"""

import numpy as np
import librosa
import soundfile as sf
import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# LibriSpeech paths (can specify multiple datasets)
# Default: train-other-500
LIBRISPEECH_PATHS = [
    '../../data/LibriSpeech/train-clean-100',
    '../../data/LibriSpeech/train-clean-360',
    # '../../data/LibriSpeech/train-other-500',
]

# Target speaker (NOT included in overlaps)
TARGET_SPEAKER = ''

# Base speaker (OPTIONAL: if specified, this speaker will be the main speaker in ALL overlaps)
# Set to None to use random speaker combinations (original behavior)
# Set to a speaker ID to have that speaker as the base in all overlaps
BASE_SPEAKER = None  # e.g., '84' to have speaker 84 as base for all overlaps

# Other speakers to use for overlap generation
# Set to None to auto-discover speakers from LibriSpeech paths (excluding target/base)
OTHER_SPEAKERS = None

# Optional: balance non-target speaker pool by sex (M/F) using LibriSpeech SPEAKERS.TXT
BALANCE_OTHER_SPEAKERS_BY_SEX = True
LIBRISPEECH_SPEAKERS_TXT = '../../data/LibriSpeech/SPEAKERS.TXT'

# Number of overlapping speakers per sample (2 = pair, 3 = trio, etc.)
# Note: If BASE_SPEAKER is set, this is the total including the base speaker
N_OVERLAPPING_SPEAKERS = 2

# Total number of random overlap samples to generate
TOTAL_SAMPLES = 100000

# Output directory
OUTPUT_DIR = 'test_outputs/data/train_other_mix_1,6s_100000_balanced_intermediate_18-4'

# Audio parameters
SAMPLE_RATE = 16000
MIN_DURATION = 1.6  # Fixed duration in seconds
MAX_DURATION = 1.6  # Fixed duration in seconds

# Overlap parameters
MIN_OVERLAP_RATIO = 1.0  # Full overlap
MAX_OVERLAP_RATIO = 0.5  # Full overlap

# Mixing parameters
MIXING_SNR_RANGE = (0, 5)  # SNR range in dB for mixing speakers

# Optional MUSAN speech-noise augmentation (applied to each generated mix when enabled)
ADD_MUSAN_SPEECH_NOISE = False
MUSAN_SPEECH_NOISE_ROOT = '../../kaldi/egs/pvad/musan/musan_speech_train'
MUSAN_SPEECH_NOISE_SNR_RANGE = (0, 15)  # SNR of speech_mix vs added MUSAN speech noise in dB

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================


def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def discover_available_speakers(librispeech_paths):
    """Discover speaker IDs from LibriSpeech subset roots."""
    all_speakers = set()

    for librispeech_path in librispeech_paths:
        if not Path(librispeech_path).exists():
            continue
        for item in Path(librispeech_path).iterdir():
            if item.is_dir() and item.name.isdigit():
                all_speakers.add(item.name)

    return sorted(list(all_speakers))


def get_speaker_utterances(librispeech_paths, speaker_id):
    """Get all audio files for a speaker from multiple LibriSpeech directories"""
    audio_files = []
    
    for librispeech_path in librispeech_paths:
        speaker_path = Path(librispeech_path) / speaker_id
        
        if speaker_path.exists():
            files = list(speaker_path.glob('**/*.flac'))
            audio_files.extend(files)
    
    return audio_files


def load_speaker_sex_map(speakers_txt_path):
    """Load speaker -> sex mapping from LibriSpeech SPEAKERS.TXT."""
    sex_map = {}

    speakers_txt_path = Path(speakers_txt_path)
    if not speakers_txt_path.exists():
        print(f"⚠️  Warning: SPEAKERS.TXT not found at: {speakers_txt_path}")
        return sex_map

    with open(speakers_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:
                continue

            speaker_id = parts[0]
            sex = parts[1].upper()

            if speaker_id.isdigit() and sex in {'M', 'F'}:
                sex_map[speaker_id] = sex

    return sex_map


def balance_speaker_pool_by_sex(speakers, speaker_sex_map):
    """
    Return a balanced speaker pool with equal male/female counts when possible.

    Speakers with unknown sex are excluded in balanced mode.
    """
    male_speakers = [s for s in speakers if speaker_sex_map.get(s) == 'M']
    female_speakers = [s for s in speakers if speaker_sex_map.get(s) == 'F']

    if len(male_speakers) == 0 or len(female_speakers) == 0:
        return speakers, {
            'male': len(male_speakers),
            'female': len(female_speakers),
            'unknown': len([s for s in speakers if s not in speaker_sex_map]),
            'balanced': False
        }

    n_per_sex = min(len(male_speakers), len(female_speakers))
    selected_males = random.sample(male_speakers, n_per_sex)
    selected_females = random.sample(female_speakers, n_per_sex)

    balanced_pool = sorted(selected_males + selected_females)
    stats = {
        'male': n_per_sex,
        'female': n_per_sex,
        'unknown': 0,
        'balanced': True
    }
    return balanced_pool, stats


def load_and_normalize_audio(audio_path, target_sr=16000):
    """Load audio and normalize to [-1, 1]"""
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Normalize
    if len(audio) > 0:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio, sr


def discover_musan_speech_noise_files(speech_noise_root):
    """Discover MUSAN speech-train noise files under the provided root."""
    speech_noise_root = Path(speech_noise_root)
    if not speech_noise_root.exists():
        return []

    speech_noise_files = set()

    # Collect all audio files under musan_speech_train.
    for pattern in ('**/*.wav', '**/*.flac'):
        for file_path in speech_noise_root.glob(pattern):
            if file_path.is_file():
                speech_noise_files.add(file_path.resolve())

    # Optional fallback: parse wav.scp manifests (supports Kaldi-style prep).
    for wav_scp_path in speech_noise_root.glob('**/wav.scp'):
        try:
            with open(wav_scp_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if '|' in line:
                        # Skip command-based wav.scp lines.
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    token = parts[1]

                    # MUSAN wav.scp entries are often relative to kaldi/egs/pvad,
                    # e.g. "musan/speech/us-gov/speech-us-gov-0102.wav".
                    # Build robust candidate bases and resolve against each one.
                    candidate_bases = [
                        speech_noise_root,
                        speech_noise_root.parent,
                        speech_noise_root.parent.parent,
                        wav_scp_path.parent,
                        wav_scp_path.parent.parent,
                    ]

                    # Add kaldi/egs/pvad ancestor if present.
                    for ancestor in (speech_noise_root, *speech_noise_root.parents):
                        if ancestor.name == 'pvad':
                            candidate_bases.append(ancestor)
                            break

                    candidate_paths = [Path(token)]
                    candidate_paths.extend(base / token for base in candidate_bases)

                    for candidate in candidate_paths:
                        if candidate.exists() and candidate.is_file():
                            speech_noise_files.add(candidate.resolve())
                            break
        except OSError:
            continue

    return sorted(str(p) for p in speech_noise_files)


def load_noise_segment(noise_path, target_samples, target_sr=16000):
    """Load a noise file and return a segment of exactly target_samples length."""
    noise, _ = librosa.load(noise_path, sr=target_sr)
    if len(noise) == 0:
        return None

    max_abs = np.max(np.abs(noise))
    if max_abs > 0:
        noise = noise / (max_abs + 1e-8)

    if len(noise) >= target_samples:
        start_idx = random.randint(0, len(noise) - target_samples)
        segment = noise[start_idx:start_idx + target_samples]
    else:
        n_repeats = int(np.ceil(target_samples / len(noise)))
        segment = np.tile(noise, n_repeats)[:target_samples]

    return segment


def mix_audio_with_snr(audio1, audio2, snr_db):
    """
    Mix two audio signals with specified SNR
    
    Args:
        audio1: First audio (signal)
        audio2: Second audio (noise)
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Mixed audio
    """
    # Calculate power
    power1 = np.mean(audio1 ** 2)
    power2 = np.mean(audio2 ** 2)
    
    # Calculate scaling factor for audio2
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(power1 / (power2 * snr_linear + 1e-8))
    
    # Mix
    mixed = audio1 + scale * audio2
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)
    
    return mixed


def create_overlap_sample(speaker_files_list, min_duration, max_duration,
                         overlap_ratio, mixing_snr_range, sample_rate=16000,
                         base_speaker_files=None, speech_noise_files=None,
                         speech_noise_snr_range=(0, 10), add_musan_speech_noise=False):
    """
    Create overlapping speech sample from multiple speakers
    
    Args:
        speaker_files_list: List of lists, each containing audio files for one speaker
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        overlap_ratio: Ratio of overlap (0.0 to 1.0)
        mixing_snr_range: Tuple of (min_snr, max_snr) in dB
        sample_rate: Audio sample rate
        base_speaker_files: Optional list of audio files for base speaker (will be in all samples)
        speech_noise_files: List of MUSAN speech-train noise files
        speech_noise_snr_range: Tuple (min_snr, max_snr) for adding speech noise
        add_musan_speech_noise: If True, add MUSAN speech-train noise to every mix
    
    Returns:
        dict with mixed audio, metadata, and labels
    """
    # If base speaker specified, use it as first speaker
    if base_speaker_files is not None:
        # Base speaker is always first
        all_speaker_files = [base_speaker_files] + speaker_files_list
        n_speakers = len(all_speaker_files)
    else:
        all_speaker_files = speaker_files_list
        n_speakers = len(speaker_files_list)
    
    if n_speakers < 2:
        raise ValueError("Need at least 2 speakers for overlap")
    
    # Sample one utterance per speaker
    selected_files = []
    selected_audios = []
    
    for speaker_files in all_speaker_files:
        if len(speaker_files) == 0:
            return None
        
        audio_file = random.choice(speaker_files)
        audio, sr = load_and_normalize_audio(audio_file, sample_rate)
        
        selected_files.append(audio_file)
        selected_audios.append(audio)
    
    # Determine target duration
    target_duration = random.uniform(min_duration, max_duration)
    target_samples = int(target_duration * sample_rate)
    
    # Trim or pad each audio to target duration
    processed_audios = []
    for audio in selected_audios:
        if len(audio) >= target_samples:
            # Trim: random start position
            start_idx = random.randint(0, len(audio) - target_samples)
            audio_segment = audio[start_idx:start_idx + target_samples]
        else:
            # Pad: repeat audio
            n_repeats = int(np.ceil(target_samples / len(audio)))
            audio_repeated = np.tile(audio, n_repeats)
            audio_segment = audio_repeated[:target_samples]
        
        processed_audios.append(audio_segment)
    
    # Scale all other speakers to match the first speaker's RMS amplitude
    # Calculate first speaker's RMS (reference loudness)
    first_speaker_rms = np.sqrt(np.mean(processed_audios[0] ** 2))
    
    # Scale all speakers to match first speaker's RMS
    matched_audios = []
    for i, audio in enumerate(processed_audios):
        if i == 0:
            # First speaker - keep as reference
            matched_audios.append(audio)
        else:
            # Scale to match first speaker's RMS
            audio_rms = np.sqrt(np.mean(audio ** 2))
            if audio_rms > 0:
                scaled_audio = audio * (first_speaker_rms / audio_rms)
            else:
                scaled_audio = audio
            matched_audios.append(scaled_audio)
    
    # Create overlap with specified ratio
    # For simplicity with multiple speakers, we'll do full overlap
    # overlap_ratio controls timing offset
    
    if overlap_ratio >= 1.0:
        # Full overlap - all speakers at same time
        offsets = [0] * n_speakers
    else:
        # Partial overlap - stagger start times
        max_offset = int(target_samples * (1.0 - overlap_ratio))
        offsets = [random.randint(0, max_offset) for _ in range(n_speakers)]
    
    # Calculate total length needed
    max_end = max(offset + len(audio) for offset, audio in zip(offsets, matched_audios))
    
    # Create mixed audio
    mixed_audio = np.zeros(max_end)
    
    # Mix all speakers with optional SNR variation
    # Set mixing_snr_range to (0, 0) for perfectly equal volumes
    for i, (audio, offset) in enumerate(zip(matched_audios, offsets)):
        if i == 0 or mixing_snr_range == (0, 0):
            # First speaker or equal volume mode - add directly
            mixed_audio[offset:offset + len(audio)] += audio
        else:
            # Other speakers with random SNR (if enabled)
            snr_db = random.uniform(*mixing_snr_range)
            if abs(snr_db) < 0.01:  # Effectively 0 dB
                mixed_audio[offset:offset + len(audio)] += audio
            else:
                # Apply SNR scaling
                signal = mixed_audio[offset:offset + len(audio)].copy()
                mixed_segment = mix_audio_with_snr(signal, audio, snr_db)
                mixed_audio[offset:offset + len(audio)] = mixed_segment

    speech_noise_added = False
    speech_noise_file_used = None
    speech_noise_snr_db_used = None

    # Optional MUSAN augmentation: add one speech-noise segment per generated mix.
    if add_musan_speech_noise and speech_noise_files:
        speech_noise_file_used = random.choice(speech_noise_files)
        speech_noise_segment = load_noise_segment(speech_noise_file_used, len(mixed_audio), sample_rate)

        if speech_noise_segment is not None:
            speech_noise_snr_db_used = random.uniform(*speech_noise_snr_range)
            mixed_audio = mix_audio_with_snr(mixed_audio, speech_noise_segment, speech_noise_snr_db_used)
            speech_noise_added = True
    
    # Normalize final mix
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio * (0.95 / max_val)
    
    # Create frame-level labels (all non-target, use -1)
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    frame_hop = int(0.010 * sample_rate)     # 10ms hop
    n_frames = int((len(mixed_audio) - frame_length) / frame_hop) + 1
    
    # All frames are non-target (no target speaker present)
    frame_labels = np.zeros(n_frames, dtype=np.int32) - 1  # -1 = non-target
    
    return {
        'audio': mixed_audio,
        'sample_rate': sample_rate,
        'duration': len(mixed_audio) / sample_rate,
        'n_speakers': n_speakers,
        'speaker_files': [str(f) for f in selected_files],
        'offsets': offsets,
        'frame_labels': frame_labels,
        'overlap_ratio': overlap_ratio,
        'speech_noise_added': speech_noise_added,
        'speech_noise_file': speech_noise_file_used,
        'speech_noise_snr_db': speech_noise_snr_db_used,
    }


def generate_overlap_samples(librispeech_paths, target_speaker, other_speakers,
                            n_overlapping_speakers, n_total_samples,
                            output_dir, config, base_speaker=None,
                            balance_other_speakers_by_sex=False,
                            speakers_txt_path=None,
                            add_musan_speech_noise=False,
                            musan_speech_noise_root=None,
                            speech_noise_snr_range=(0, 10)):
    """Generate non-target overlap samples"""
    
    print(f"\n{'='*70}")
    print(f"GENERATING NON-TARGET OVERLAP SAMPLES")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  LibriSpeech datasets ({len(librispeech_paths)}):")
    for path in librispeech_paths:
        print(f"    - {path}")
    print(f"  Target speaker (excluded): {target_speaker}")
    if base_speaker is not None:
        print(f"  Base speaker (in all overlaps): {base_speaker}")
    print(f"  Other speakers: {len(other_speakers) if other_speakers is not None else 'Auto-discover'}")
    print(f"  Balance non-target speakers by sex: {balance_other_speakers_by_sex}")
    print(f"  Add MUSAN speech noise to each mix: {add_musan_speech_noise}")
    if add_musan_speech_noise:
        print(f"  MUSAN speech-noise root: {musan_speech_noise_root}")
        print(f"  MUSAN speech-noise SNR range: {speech_noise_snr_range}")
    print(f"  Speakers per overlap: {n_overlapping_speakers}")
    print(f"  Total random samples: {n_total_samples}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directories
    output_path = Path(output_dir)
    audio_dir = output_path / 'audio'
    labels_dir = output_path / 'labels'
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    speech_noise_files = []
    if add_musan_speech_noise:
        if musan_speech_noise_root is None:
            print("\n❌ Error: MUSAN speech-noise is enabled but no root path was provided")
            return []

        speech_noise_files = discover_musan_speech_noise_files(musan_speech_noise_root)
        if len(speech_noise_files) == 0:
            print(f"\n❌ Error: No MUSAN speech-noise files found under: {musan_speech_noise_root}")
            print("   Expected .wav/.flac files or wav.scp entries in musan_speech_train")
            return []

        print(f"  ✓ Discovered {len(speech_noise_files)} MUSAN speech-noise files")
    
    # Determine speaker pool (auto-discover by default)
    if other_speakers is None:
        discovered = discover_available_speakers(librispeech_paths)
        other_speakers = [
            s for s in discovered
            if s != target_speaker and (base_speaker is None or s != base_speaker)
        ]
        print(f"  Auto-discovered speakers from LibriSpeech: {len(other_speakers)}")

    # Optionally balance other speaker pool by sex
    if balance_other_speakers_by_sex:
        if speakers_txt_path is None:
            print(f"  ⚠️  SPEAKERS.TXT path not provided; skipping sex balancing")
        else:
            speaker_sex_map = load_speaker_sex_map(speakers_txt_path)
            if not speaker_sex_map:
                print(f"  ⚠️  Could not load speaker sex mapping; skipping sex balancing")
            else:
                before_count = len(other_speakers)
                other_speakers, sex_stats = balance_speaker_pool_by_sex(other_speakers, speaker_sex_map)
                after_count = len(other_speakers)

                if sex_stats['balanced']:
                    print(f"  ✓ Applied sex-balanced speaker pool: {before_count} -> {after_count}")
                    print(f"    Distribution: M={sex_stats['male']}, F={sex_stats['female']}, Unknown={sex_stats['unknown']}")
                else:
                    print(f"  ⚠️  Could not fully balance by sex (M={sex_stats['male']}, F={sex_stats['female']}); using unbalanced pool")

    # Get utterances for all other speakers (excluding target)
    print(f"\n📂 Loading speaker utterances from {len(librispeech_paths)} dataset(s)...")
    speaker_utterances = {}
    
    # Load base speaker if specified
    base_speaker_utterances = None
    if base_speaker is not None:
        base_speaker_utterances = get_speaker_utterances(librispeech_paths, base_speaker)
        if len(base_speaker_utterances) == 0:
            print(f"\n❌ Error: Base speaker {base_speaker} has no utterances!")
            return []
        print(f"  Base speaker {base_speaker}: {len(base_speaker_utterances)} utterances")
    
    for speaker_id in other_speakers:
        # Never include target speaker in non-target mixtures
        if speaker_id == target_speaker:
            continue

        # Skip base speaker if it's in other_speakers list
        if base_speaker is not None and speaker_id == base_speaker:
            continue
            
        utterances = get_speaker_utterances(librispeech_paths, speaker_id)
        if len(utterances) > 0:
            speaker_utterances[speaker_id] = utterances
            print(f"  Speaker {speaker_id}: {len(utterances)} utterances")
        else:
            print(f"  ⚠️  Speaker {speaker_id}: No utterances found")
    
    available_speakers = list(speaker_utterances.keys())
    
    # Calculate required speakers based on whether base speaker is used
    required_speakers = n_overlapping_speakers if base_speaker is None else n_overlapping_speakers - 1
    
    if len(available_speakers) < required_speakers:
        print(f"\n❌ Error: Not enough speakers available!")
        print(f"   Need: {required_speakers}, Available: {len(available_speakers)}")
        return []
    
    print(f"\n✓ Loaded utterances for {len(available_speakers)} speakers")
    
    if base_speaker is not None:
        print(f"\n🔢 Base speaker mode: {base_speaker} will be in all overlaps")

    print(f"\n🔢 Randomly selecting speaker sets per utterance")
    print(f"   Total samples to generate: {n_total_samples}")
    
    # Generate samples
    metadata_list = []
    sample_id = 0
    failed_count = 0
    
    progress_bar = tqdm(total=n_total_samples,
                       desc="Generating overlaps", unit="sample")

    for i in range(n_total_samples):
        try:
            # Randomly pick speakers for this sample
            if base_speaker is not None:
                n_other_needed = n_overlapping_speakers - 1
                speaker_combo = tuple(random.sample(available_speakers, n_other_needed))
            else:
                speaker_combo = tuple(random.sample(available_speakers, n_overlapping_speakers))

            speaker_files_list = [speaker_utterances[spk] for spk in speaker_combo]

            # Random overlap ratio (fixed to 1.0 by current defaults)
            overlap_ratio = random.uniform(config['min_overlap_ratio'], config['max_overlap_ratio'])

            # Create overlap sample
            result = create_overlap_sample(
                speaker_files_list,
                config['min_duration'],
                config['max_duration'],
                overlap_ratio,
                config['mixing_snr_range'],
                config['sample_rate'],
                base_speaker_files=base_speaker_utterances,
                speech_noise_files=speech_noise_files,
                speech_noise_snr_range=speech_noise_snr_range,
                add_musan_speech_noise=add_musan_speech_noise,
            )

            if result is None:
                failed_count += 1
                progress_bar.update(1)
                continue

            # Create sample name
            if base_speaker is not None:
                speakers_str = base_speaker + '_' + '_'.join(speaker_combo)
            else:
                speakers_str = '_'.join(speaker_combo)
            sample_name = f"sample_{sample_id:06d}_speakers{speakers_str}"

            # Save audio
            audio_path = audio_dir / f"{sample_name}.wav"
            sf.write(audio_path, result['audio'], result['sample_rate'])

            # Save labels
            labels_path = labels_dir / f"{sample_name}.npy"
            np.save(labels_path, result['frame_labels'])

            # Save metadata
            if base_speaker is not None:
                overlapping_speakers = [base_speaker] + list(speaker_combo)
            else:
                overlapping_speakers = list(speaker_combo)

            metadata = {
                'sample_id': sample_name,
                'target_speaker': target_speaker,
                'base_speaker': base_speaker,
                'overlapping_speakers': overlapping_speakers,
                'n_speakers': result['n_speakers'],
                'duration': result['duration'],
                'overlap_ratio': result['overlap_ratio'],
                'speaker_files': result['speaker_files'],
                'audio_file': str(audio_path.relative_to(output_path)),
                'labels_file': str(labels_path.relative_to(output_path)),
                'output_name': sample_name,
                'is_target_present': False,  # Key difference: target NOT present
                'musan_speech_noise_added': result['speech_noise_added'],
                'musan_speech_noise_file': result['speech_noise_file'],
                'musan_speech_noise_snr_db': result['speech_noise_snr_db'],
            }

            metadata_list.append(metadata)
            sample_id += 1

        except Exception as e:
            print(f"\n⚠️  Error generating sample: {e}")
            failed_count += 1

        progress_bar.update(1)
    
    progress_bar.close()
    
    print(f"\n{'='*70}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Statistics:")
    print(f"  Successful samples: {len(metadata_list)}")
    print(f"  Failed samples: {failed_count}")
    print(f"  Success rate: {100*len(metadata_list)/(len(metadata_list)+failed_count):.1f}%")
    if add_musan_speech_noise and len(metadata_list) > 0:
        noise_added_count = sum(1 for m in metadata_list if m.get('musan_speech_noise_added', False))
        print(f"  Samples with MUSAN speech noise: {noise_added_count}/{len(metadata_list)}")
    
    # Save metadata
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n💾 Saved:")
    print(f"  Audio files: {audio_dir}")
    print(f"  Label files: {labels_dir}")
    print(f"  Metadata: {metadata_path}")
    
    # Print speaker distribution
    speaker_counts = defaultdict(int)
    for metadata in metadata_list:
        for speaker in metadata['overlapping_speakers']:
            speaker_counts[speaker] += 1
    
    print(f"\n📊 Speaker distribution in overlaps:")
    for speaker in sorted(speaker_counts.keys()):
        print(f"  Speaker {speaker}: {speaker_counts[speaker]} samples")
    
    return metadata_list


def main():
    parser = argparse.ArgumentParser(description='Generate non-target overlapping speech samples')
    parser.add_argument('--librispeech-paths', nargs='+', default=LIBRISPEECH_PATHS,
                       help='Paths to LibriSpeech datasets (can specify multiple)')
    parser.add_argument('--target-speaker', type=str, default=TARGET_SPEAKER,
                       help='Target speaker ID (will be EXCLUDED from overlaps)')
    parser.add_argument('--base-speaker', type=str, default=BASE_SPEAKER,
                       help='Base speaker ID (will be in ALL overlaps). Set to None for random combinations.')
    parser.add_argument('--other-speakers', nargs='+', default=OTHER_SPEAKERS,
                       help='Other speaker IDs to use for overlaps (omit to auto-discover from LibriSpeech paths)')
    parser.add_argument('--n-overlapping-speakers', type=int, default=N_OVERLAPPING_SPEAKERS,
                       help='Number of speakers per overlap sample (includes base speaker if specified)')
    parser.add_argument('--n-samples', type=int, default=TOTAL_SAMPLES,
                       help='Total number of random samples to generate')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    parser.add_argument('--balance-other-speakers-by-sex', action='store_true',
                       default=BALANCE_OTHER_SPEAKERS_BY_SEX,
                       help='Balance non-target speaker pool by sex (M/F) using SPEAKERS.TXT metadata')
    parser.add_argument('--speakers-txt-path', type=str, default=LIBRISPEECH_SPEAKERS_TXT,
                       help='Path to LibriSpeech SPEAKERS.TXT (used for sex balancing)')
    parser.add_argument('--add-musan-speech-noise', action='store_true', default=ADD_MUSAN_SPEECH_NOISE,
                       help='Add one MUSAN speech-noise segment to every generated mix')
    parser.add_argument('--musan-speech-noise-root', type=str, default=MUSAN_SPEECH_NOISE_ROOT,
                       help='Path to MUSAN speech-train root (e.g., musan_speech_train)')
    parser.add_argument('--musan-speech-noise-snr-min', type=float, default=MUSAN_SPEECH_NOISE_SNR_RANGE[0],
                       help='Minimum SNR (dB) of speech mix vs added MUSAN speech noise')
    parser.add_argument('--musan-speech-noise-snr-max', type=float, default=MUSAN_SPEECH_NOISE_SNR_RANGE[1],
                       help='Maximum SNR (dB) of speech mix vs added MUSAN speech noise')
    
    args = parser.parse_args()
    
    # Normalize string "None" from CLI
    if isinstance(args.base_speaker, str) and args.base_speaker.lower() == 'none':
        args.base_speaker = None

    # Set random seeds
    set_seeds(args.seed)
    
    # Resolve paths
    script_dir = Path(__file__).parent
    
    # Resolve all LibriSpeech paths
    librispeech_paths = []
    for path_str in args.librispeech_paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = script_dir / path
        librispeech_paths.append(path)
    
    # Filter out non-existent paths
    existing_paths = [p for p in librispeech_paths if p.exists()]
    if not existing_paths:
        print(f"❌ Error: No valid LibriSpeech directories found")
        for path in librispeech_paths:
            print(f"   Not found: {path}")
        return
    
    if len(existing_paths) < len(librispeech_paths):
        print(f"⚠️  Warning: {len(librispeech_paths) - len(existing_paths)} path(s) not found, using {len(existing_paths)} valid path(s)")
    
    librispeech_paths = existing_paths
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    speakers_txt_path = Path(args.speakers_txt_path)
    if not speakers_txt_path.is_absolute():
        speakers_txt_path = script_dir / speakers_txt_path

    musan_speech_noise_root = Path(args.musan_speech_noise_root)
    if not musan_speech_noise_root.is_absolute():
        musan_speech_noise_root = script_dir / musan_speech_noise_root

    speech_noise_snr_range = (
        min(args.musan_speech_noise_snr_min, args.musan_speech_noise_snr_max),
        max(args.musan_speech_noise_snr_min, args.musan_speech_noise_snr_max),
    )
    
    # Configuration
    config = {
        'min_duration': MIN_DURATION,
        'max_duration': MAX_DURATION,
        'min_overlap_ratio': MIN_OVERLAP_RATIO,
        'max_overlap_ratio': MAX_OVERLAP_RATIO,
        'mixing_snr_range': MIXING_SNR_RANGE,
        'sample_rate': SAMPLE_RATE
    }
    
    # Generate samples
    metadata_list = generate_overlap_samples(
        librispeech_paths=librispeech_paths,
        target_speaker=args.target_speaker,
        other_speakers=args.other_speakers,
        n_overlapping_speakers=args.n_overlapping_speakers,
        n_total_samples=args.n_samples,
        output_dir=output_dir,
        config=config,
        base_speaker=args.base_speaker,
        balance_other_speakers_by_sex=args.balance_other_speakers_by_sex,
        speakers_txt_path=speakers_txt_path,
        add_musan_speech_noise=args.add_musan_speech_noise,
        musan_speech_noise_root=musan_speech_noise_root,
        speech_noise_snr_range=speech_noise_snr_range,
    )
    
    if len(metadata_list) > 0:
        print(f"\n✅ Successfully generated {len(metadata_list)} non-target overlap samples!")
        print(f"   These samples contain NO target speaker {args.target_speaker}")
        print(f"   Use them to train model to recognize when target is absent")
    else:
        print(f"\n❌ Failed to generate any samples")


if __name__ == "__main__":
    main()
