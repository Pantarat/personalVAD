#!/usr/bin/env python3
"""
Generate Fully Overlapping Utterances for Speaker Separation Testing

Creates audio samples where:
1. One main speaker is ALWAYS present (target speaker)
2. Other speakers overlap for the ENTIRE duration of the output
3. All speakers talk simultaneously from start to end

This is useful for testing speaker separation in continuous overlap scenarios.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import glob
import random
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to LibriSpeech directories (can specify multiple datasets)
# Examples: 'dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360', etc.
LIBRISPEECH_PATHS = [
    '../../data/LibriSpeech/dev-clean',
    '../../data/LibriSpeech/dev-other',
    # # '../../data/LibriSpeech/test-clean',
    '../../data/LibriSpeech/train-clean-100',
    '../../data/LibriSpeech/train-clean-360',
    # '../../data/LibriSpeech/train-other-500',
]

# Main speakers configuration
# Option 1: Use single speaker datasets (HIGHEST PRIORITY)
# Set to a list of paths to single speaker datasets (from extract_single_speaker.py)
# This will use the pre-generated chunks as the target speaker(s)
# Can specify multiple datasets to combine speakers from different sources
SINGLE_SPEAKER_DATASETS = [
    '../../data/speaker_8463/train1',
    '../../data/speaker_8463/train2',
    '../../data/speaker_8463/train3',
    '../../data/speaker_8463/train4',
    '../../data/speaker_8463/train5',
    '../../data/speaker_8463/train6',
    '../../data/speaker_8463/train7',
    '../../data/speaker_8463/train8',
    '../../data/speaker_8463/train9',
    '../../data/speaker_8463/train10',
    '../../data/speaker_8463/train11',
    '../../data/speaker_8463/train12',
    '../../data/speaker_8463/train13',
    '../../data/speaker_8463/train14',
    '../../data/speaker_8463/train15',
    '../../data/speaker_8463/train16',
    '../../data/speaker_8463/train17',
    '../../data/speaker_8463/train18',
    '../../data/speaker_8463/train19',
    '../../data/speaker_8463/train20',
    '../../data/speaker_8463/train21',
    '../../data/speaker_8463/train22',
    '../../data/speaker_8463/train23',
    '../../data/speaker_8463/train24',
    '../../data/speaker_8463/train25',
]  # List of single speaker dataset paths
SINGLE_SPEAKER_UTT_COUNT = 0  # Number of utterances to use per dataset (0 = use all)

# Option 2: Specify one target speaker + randomly select additional main speakers
TARGET_SPEAKER = 8463  # Primary target speaker (always included)
N_ADDITIONAL_MAIN_SPEAKERS = 49  # Number of additional random main speakers

# Option 3: Explicitly list all main speakers (if SINGLE_SPEAKER_DATASET and TARGET_SPEAKER are None)
MAIN_SPEAKERS = ['8463']  # e.g., ['61', '174', '251'] - overrides TARGET_SPEAKER if set
# Option 4: No main speaker mode - generate random speaker mixtures without target
# When True, ignores all main speaker configurations and randomly mixes any speakers
NO_MAIN_SPEAKER = False  # Set to True to generate random mixtures without main speaker

# Other speakers configuration
# Note: In NO_MAIN_SPEAKER mode, these are the speakers to randomly mix
# Option 1: Explicitly list speakers
OTHER_SPEAKERS = None  # Set to list like ['174', '251', ...] to specify explicitly
# OTHER_SPEAKERS = ['61', '121', '237', '260', '672', '908', '1089', '1188', '1221', '1284', '1320', '1580', '1995', '2094', '2300', '2830', '2961', '3570', '3575', '3729', '4077', '4446', '4507', '4970', '4992', '5105', '5142', '5639', '6829', '6930'] # test noise

# Option 2: Automatically select N speakers (if OTHER_SPEAKERS is None)
N_OTHER_SPEAKERS = 300  # Number of other speakers to randomly select from available speakers
EXCLUDE_SPEAKERS = []  # Speaker IDs to exclude from automatic selection (e.g., ['84', '116'])
# Optional: balance automatically selected other speakers by sex using LibriSpeech SPEAKERS.TXT
BALANCE_OTHER_SPEAKERS_BY_SEX = False  # If True, tries to select ~50% male and ~50% female
LIBRISPEECH_SPEAKERS_TXT = '../../data/LibriSpeech/SPEAKERS.TXT'

# Number of overlapping speakers per output (always 1)
N_OVERLAPPING_SPEAKERS = 1  # Main speaker + 1 other = 2 total speakers

# Total number of output samples to generate
# If set to 0, will generate enough samples to use all main speaker utterances at least once
TOTAL_SAMPLES = 2000  # Total utterances to generate (0 = use all main speaker utterances)

# Percentage of samples with main speaker overlap (rest will be non-main speaker only)
MAIN_SPEAKER_OVERLAP_PERCENTAGE = 100  # Percentage of samples with main speaker (0-100)

# Output directory
OUTPUT_DIR = 'test_outputs/data/8463/8463_100pct_75utt-50spk+300Dev_5s_100pctmainspk_100pctAmp_2000'

# Target duration for output (seconds)
# If None, uses the main speaker's utterance duration
TARGET_DURATION = 5  # or set to e.g., 5.0 for 5 seconds

# Audio mixing parameters
MAIN_SPEAKER_VOLUME = 1.0  # Volume scaling for main speaker (1.0 = original)
OTHER_SPEAKER_VOLUME = 1.0  # Volume scaling for overlapping speakers

# Random seed for reproducibility
RANDOM_SEED = 42

# Sample rate
SAMPLE_RATE = 16000

# ============================================================================


def discover_available_speakers(librispeech_paths):
    """
    Discover all available speakers from LibriSpeech directories
    
    Args:
        librispeech_paths: List of paths to LibriSpeech directories
        
    Returns:
        list: List of speaker IDs (strings) found across all directories
    """
    all_speakers = set()
    
    for librispeech_path in librispeech_paths:
        if not librispeech_path.exists():
            continue
        # Find all speaker directories (directories with numeric names)
        for item in librispeech_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                all_speakers.add(item.name)
    
    return sorted(list(all_speakers))


def load_single_speaker_dataset(dataset_dir, max_utterances=None):
    """
    Load utterances from a single speaker dataset (created by extract_single_speaker.py).
    
    Args:
        dataset_dir: Path to the single speaker dataset directory
        max_utterances: Optional maximum number of utterances to load (randomly sampled)
        
    Returns:
        tuple: (speaker_id, list of audio file paths)
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"⚠️  Warning: Single speaker dataset not found: {dataset_dir}")
        return None, []
    
    # Extract speaker ID from utt2spk file
    utt2spk_path = dataset_path / 'utt2spk'
    speaker_id = None
    if utt2spk_path.exists():
        with open(utt2spk_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                speaker_id = first_line.split()[1]
    
    if not speaker_id:
        print(f"⚠️  Warning: Could not extract speaker ID from {dataset_dir}")
        return None, []
    
    # Load audio files from the audio directory
    audio_dir = dataset_path / 'audio'
    if not audio_dir.exists():
        print(f"⚠️  Warning: Audio directory not found: {audio_dir}")
        return speaker_id, []
    
    audio_files = sorted(list(audio_dir.glob('*.flac')))
    audio_files = [str(f) for f in audio_files]
    
    print(f"  Loaded {len(audio_files)} utterances from single speaker dataset (speaker {speaker_id})")
    
    # Randomly sample if max_utterances is specified
    if max_utterances and max_utterances > 0 and len(audio_files) > max_utterances:
        print(f"  Randomly sampling {max_utterances} utterances from {len(audio_files)} total")
        audio_files = random.sample(audio_files, max_utterances)
    
    return speaker_id, audio_files


def load_speaker_utterances(librispeech_paths, speaker_id):
    """
    Load all audio files for a given speaker from multiple LibriSpeech directories
    
    Args:
        librispeech_paths: List of paths to LibriSpeech directories
        speaker_id: Speaker ID to load
        
    Returns:
        list: List of audio file paths from all directories
    """
    audio_files = []
    
    for librispeech_path in librispeech_paths:
        speaker_pattern = str(Path(librispeech_path) / speaker_id / "*" / "*.flac")
        files = glob.glob(speaker_pattern)
        audio_files.extend(files)
    
    if not audio_files:
        print(f"⚠️  Warning: No audio files found for speaker {speaker_id} in any dataset")
    
    return audio_files


def load_speaker_sex_map(speakers_txt_path):
    """
    Load speaker -> sex mapping from LibriSpeech SPEAKERS.TXT.

    Args:
        speakers_txt_path: Path to SPEAKERS.TXT

    Returns:
        dict: {speaker_id (str): sex ('M' or 'F')}
    """
    sex_map = {}

    if not speakers_txt_path.exists():
        print(f"⚠️  Warning: SPEAKERS.TXT not found at {speakers_txt_path}")
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


def select_balanced_speakers_by_sex(candidates, n_speakers, speaker_sex_map):
    """
    Select speakers with approximately balanced male/female distribution.

    Args:
        candidates: List of candidate speaker IDs
        n_speakers: Number of speakers to select
        speaker_sex_map: Dict mapping speaker_id -> 'M'/'F'

    Returns:
        list: Selected speaker IDs
    """
    male_candidates = [s for s in candidates if speaker_sex_map.get(s) == 'M']
    female_candidates = [s for s in candidates if speaker_sex_map.get(s) == 'F']
    unknown_candidates = [s for s in candidates if s not in speaker_sex_map]

    # For odd n_speakers, allow one extra female by default (difference <= 1)
    target_female = n_speakers // 2
    target_male = n_speakers - target_female

    selected_males = random.sample(male_candidates, min(target_male, len(male_candidates)))
    selected_females = random.sample(female_candidates, min(target_female, len(female_candidates)))

    selected = selected_males + selected_females

    # Fill any remaining slots from all remaining candidates (including unknown sex)
    remaining_needed = n_speakers - len(selected)
    if remaining_needed > 0:
        remaining_pool = [
            s for s in (male_candidates + female_candidates + unknown_candidates)
            if s not in selected
        ]
        selected.extend(random.sample(remaining_pool, min(remaining_needed, len(remaining_pool))))

    return selected


def load_and_preprocess_audio(file_path, target_duration=None, sample_rate=16000):
    """
    Load audio file and optionally adjust to target duration
    
    Args:
        file_path: Path to audio file
        target_duration: Target duration in seconds (None = keep original)
        sample_rate: Sample rate
        
    Returns:
        numpy.ndarray: Audio samples
    """
    # Load audio
    audio, sr = librosa.load(file_path, sr=sample_rate)
    
    if target_duration is not None:
        target_samples = int(target_duration * sample_rate)
        current_samples = len(audio)
        
        if current_samples < target_samples:
            # Pad with silence
            audio = np.pad(audio, (0, target_samples - current_samples), mode='constant')
        elif current_samples > target_samples:
            # Truncate
            audio = audio[:target_samples]
    
    return audio


def adjust_audio_to_duration(audio, target_samples, sample_rate=16000):
    """
    Adjust audio length to match target duration
    
    Args:
        audio: Audio samples
        target_samples: Target number of samples
        sample_rate: Sample rate
        
    Returns:
        numpy.ndarray: Adjusted audio
    """
    current_samples = len(audio)
    
    if current_samples < target_samples:
        # Loop audio to fill duration
        num_loops = int(np.ceil(target_samples / current_samples))
        audio = np.tile(audio, num_loops)[:target_samples]
    elif current_samples > target_samples:
        # Truncate
        audio = audio[:target_samples]
    
    return audio


def mix_utterances(main_audio, other_audios, main_volume=1.0, other_volume=0.7):
    """
    Mix main speaker audio with other speakers (all same length)
    Other speakers are scaled to match the main speaker's RMS amplitude.
    
    Args:
        main_audio: Main speaker audio samples
        other_audios: List of other speaker audio samples (all same length as main)
        main_volume: Volume scaling for main speaker (applied after measuring RMS)
        other_volume: Volume scaling for other speakers (applied after matching main speaker RMS)
        
    Returns:
        tuple: (mixed_audio, label_array)
            - mixed_audio: Combined audio
            - label_array: Frame-level labels (0=NS, 1=other speaker, 2=main speaker overlap)
    """
    # Calculate main speaker's RMS (loudness)
    main_rms = np.sqrt(np.mean(main_audio ** 2))
    
    # Apply volume scaling to main speaker
    mixed = main_audio * main_volume
    
    # Scale other speakers to match main speaker's RMS, then apply volume scaling
    for other_audio in other_audios:
        other_rms = np.sqrt(np.mean(other_audio ** 2))
        
        # Scale to match main speaker's RMS
        if other_rms > 0:
            other_scaled = other_audio * (main_rms / other_rms)
        else:
            other_scaled = other_audio
        
        # Apply additional volume scaling
        mixed += other_scaled * other_volume
    
    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95
    
    # Create labels: 2 for entire duration (main speaker always present with overlap)
    frame_duration = 0.01  # 10ms frames
    n_frames = int(len(mixed) / (SAMPLE_RATE * frame_duration))
    labels = np.full(n_frames, 2, dtype=np.int32)  # All frames = main speaker with overlap
    
    return mixed, labels


def generate_overlap_sample(main_utterance_file, other_utterance_files, 
                           main_speaker_id, target_duration=None, main_volume=1.0, other_volume=0.7):
    """
    Generate one fully overlapping sample
    
    Args:
        main_utterance_file: Path to main speaker audio
        other_utterance_files: List of paths to other speakers' audio
        main_speaker_id: ID of the main speaker
        target_duration: Target duration in seconds (None = use main speaker duration)
        main_volume: Volume scaling for main speaker
        other_volume: Volume scaling for other speakers
        
    Returns:
        tuple: (mixed_audio, labels, metadata)
    """
    # Load main speaker audio
    main_audio = load_and_preprocess_audio(main_utterance_file, sample_rate=SAMPLE_RATE)
    
    # Determine target duration
    if target_duration is None:
        target_samples = len(main_audio)
        actual_duration = len(main_audio) / SAMPLE_RATE
    else:
        target_samples = int(target_duration * SAMPLE_RATE)
        actual_duration = target_duration
        main_audio = adjust_audio_to_duration(main_audio, target_samples, SAMPLE_RATE)
    
    # Load and adjust other speakers' audio to match duration
    other_audios = []
    for other_file in other_utterance_files:
        other_audio = load_and_preprocess_audio(other_file, sample_rate=SAMPLE_RATE)
        other_audio = adjust_audio_to_duration(other_audio, target_samples, SAMPLE_RATE)
        other_audios.append(other_audio)
    
    # Mix all utterances
    mixed_audio, labels = mix_utterances(main_audio, other_audios, main_volume, other_volume)
    
    # Create metadata
    main_name = Path(main_utterance_file).stem
    other_names = [Path(f).stem for f in other_utterance_files]
    
    metadata = {
        'main_speaker': main_speaker_id,
        'main_utterance': main_name,
        'other_speakers': [Path(f).parts[-3] for f in other_utterance_files],  # Extract speaker IDs
        'other_utterances': other_names,
        'duration': actual_duration,
        'n_speakers': 1 + len(other_audios),
        'sample_rate': SAMPLE_RATE
    }
    
    return mixed_audio, labels, metadata


def generate_dataset():
    """
    Generate full dataset of overlapping utterances
    """
    print("=" * 70)
    print("FULL OVERLAP UTTERANCE GENERATOR")
    print("=" * 70)
    
    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Resolve paths
    script_dir = Path(__file__).parent
    
    # Resolve all LibriSpeech paths
    librispeech_paths = []
    for path_str in LIBRISPEECH_PATHS:
        path = Path(path_str)
        if not path.is_absolute():
            path = script_dir / path
        librispeech_paths.append(path)

    speakers_txt_path = Path(LIBRISPEECH_SPEAKERS_TXT)
    if not speakers_txt_path.is_absolute():
        speakers_txt_path = script_dir / speakers_txt_path
    
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    # Create output directories
    audio_dir = output_dir / 'audio'
    label_dir = output_dir / 'labels'
    audio_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize main speakers to list
    # Priority: NO_MAIN_SPEAKER > SINGLE_SPEAKER_DATASETS > MAIN_SPEAKERS > TARGET_SPEAKER
    main_speakers = None
    single_speaker_utterances_by_id = defaultdict(list)  # speaker_id -> list of utterance paths
    
    if NO_MAIN_SPEAKER:
        print(f"\n🎲 NO_MAIN_SPEAKER mode enabled: Generating random speaker mixtures")
        print(f"   No target speaker - all speakers treated equally")
        main_speakers = []  # Empty list to indicate no main speakers
    elif SINGLE_SPEAKER_DATASETS:
        # Use single speaker datasets (highest priority)
        print(f"\n🔍 Loading single speaker datasets...")
        all_speaker_ids = set()
        
        for dataset_path_str in SINGLE_SPEAKER_DATASETS:
            dataset_path = Path(dataset_path_str)
            if not dataset_path.is_absolute():
                dataset_path = script_dir / dataset_path
            
            print(f"  Loading: {dataset_path}")
            speaker_id, utterances = load_single_speaker_dataset(
                dataset_path, 
                max_utterances=SINGLE_SPEAKER_UTT_COUNT if SINGLE_SPEAKER_UTT_COUNT > 0 else None
            )
            
            if speaker_id and utterances:
                single_speaker_utterances_by_id[speaker_id].extend(utterances)
                all_speaker_ids.add(speaker_id)
        
        if all_speaker_ids:
            main_speakers = sorted(list(all_speaker_ids))
            total_utterances = sum(len(utts) for utts in single_speaker_utterances_by_id.values())
            print(f"  ✓ Loaded {len(main_speakers)} speaker(s) with {total_utterances} total utterances")
            for spk_id in main_speakers:
                print(f"    - Speaker {spk_id}: {len(single_speaker_utterances_by_id[spk_id])} utterances")
        else:
            print(f"  ⚠️  Failed to load any single speaker datasets, falling back to other options")
    
    if main_speakers is None:
        main_speakers = MAIN_SPEAKERS
    
    if main_speakers is None:
        # Use TARGET_SPEAKER + N_ADDITIONAL_MAIN_SPEAKERS
        if TARGET_SPEAKER is None:
            print(f"\n❌ Error: Either NO_MAIN_SPEAKER=True, SINGLE_SPEAKER_DATASET, MAIN_SPEAKERS, or TARGET_SPEAKER must be specified")
            return
        
        print(f"\n🔍 Building main speaker list...")
        print(f"  Target speaker: {TARGET_SPEAKER}")
        
        main_speakers = [TARGET_SPEAKER]
        
        if N_ADDITIONAL_MAIN_SPEAKERS > 0:
            print(f"  Randomly selecting {N_ADDITIONAL_MAIN_SPEAKERS} additional main speakers...")
            available_speakers = discover_available_speakers(librispeech_paths)
            candidates = [s for s in available_speakers 
                         if s not in EXCLUDE_SPEAKERS and s != TARGET_SPEAKER]
            
            if N_ADDITIONAL_MAIN_SPEAKERS > len(candidates):
                print(f"  ⚠️  Requested {N_ADDITIONAL_MAIN_SPEAKERS} speakers but only {len(candidates)} available")
                print(f"     Using all {len(candidates)} available speakers")
                main_speakers.extend(sorted(candidates))
            else:
                additional = sorted(random.sample(candidates, N_ADDITIONAL_MAIN_SPEAKERS))
                main_speakers.extend(additional)
            
            print(f"  ✓ Total main speakers: {len(main_speakers)} ({TARGET_SPEAKER} + {len(main_speakers)-1} random)")
    elif isinstance(main_speakers, str):
        main_speakers = [main_speakers]
    
    # Determine other speakers
    other_speakers = OTHER_SPEAKERS
    if other_speakers is None:
        print(f"\n🔍 Discovering available speakers from LibriSpeech...")
        available_speakers = discover_available_speakers(librispeech_paths)
        print(f"  Found {len(available_speakers)} total speakers")
        
        if NO_MAIN_SPEAKER:
            # In NO_MAIN_SPEAKER mode, select from all available speakers
            candidates = [s for s in available_speakers if s not in EXCLUDE_SPEAKERS]
            print(f"  Candidates (excluding excluded speakers): {len(candidates)}")
        else:
            # Filter out main speakers and excluded speakers
            candidates = [s for s in available_speakers if s not in main_speakers and s not in EXCLUDE_SPEAKERS]
            print(f"  Candidates (excluding main and excluded): {len(candidates)}")
        
        if len(candidates) < N_OTHER_SPEAKERS:
            print(f"  ⚠️  Requested {N_OTHER_SPEAKERS} speakers but only {len(candidates)} available")
            print(f"     Using all {len(candidates)} available speakers")
            other_speakers = candidates
        else:
            # Select N_OTHER_SPEAKERS (optionally balanced by sex)
            random.seed(RANDOM_SEED)
            if BALANCE_OTHER_SPEAKERS_BY_SEX:
                speaker_sex_map = load_speaker_sex_map(speakers_txt_path)
                if speaker_sex_map:
                    other_speakers = sorted(
                        select_balanced_speakers_by_sex(candidates, N_OTHER_SPEAKERS, speaker_sex_map)
                    )
                    male_count = sum(1 for s in other_speakers if speaker_sex_map.get(s) == 'M')
                    female_count = sum(1 for s in other_speakers if speaker_sex_map.get(s) == 'F')
                    unknown_count = len(other_speakers) - male_count - female_count
                    print(f"  ✓ Selected {len(other_speakers)} speakers with sex balancing enabled")
                    print(f"    Sex distribution: M={male_count}, F={female_count}, Unknown={unknown_count}")
                else:
                    print(f"  ⚠️  Could not load speaker sex mapping; falling back to random selection")
                    other_speakers = sorted(random.sample(candidates, N_OTHER_SPEAKERS))
                    print(f"  ✓ Selected {len(other_speakers)} random speakers")
            else:
                other_speakers = sorted(random.sample(candidates, N_OTHER_SPEAKERS))
                print(f"  ✓ Selected {len(other_speakers)} random speakers")
    
    # Validate separation: main speakers and other speakers must not intersect (except in NO_MAIN_SPEAKER mode)
    if not NO_MAIN_SPEAKER:
        main_speakers_set = set(main_speakers)
        other_speakers_set = set(other_speakers)
        intersection = main_speakers_set & other_speakers_set
        if intersection:
            raise ValueError(f"Main speakers and other speakers must not overlap. Intersection: {intersection}")
    
    print(f"\n📋 Configuration:")
    print(f"  LibriSpeech datasets ({len(librispeech_paths)}):")
    for path in librispeech_paths:
        print(f"    - {path}")
    
    if NO_MAIN_SPEAKER:
        print(f"  Mode: NO MAIN SPEAKER (random speaker mixtures)")
        print(f"  Speakers to mix ({len(other_speakers)}): {', '.join(other_speakers[:10])}{'...' if len(other_speakers) > 10 else ''}")
    else:
        if single_speaker_utterances_by_id:
            print(f"  Main speaker source: Single speaker dataset(s)")
            total_utts = sum(len(utts) for utts in single_speaker_utterances_by_id.values())
            print(f"  Main speakers ({len(single_speaker_utterances_by_id)}): {total_utts} total utterances")
            for spk_id in sorted(single_speaker_utterances_by_id.keys()):
                print(f"    - Speaker {spk_id}: {len(single_speaker_utterances_by_id[spk_id])} utterances")
            if SINGLE_SPEAKER_UTT_COUNT > 0:
                print(f"    (Limited to {SINGLE_SPEAKER_UTT_COUNT} utterances per dataset)")
        else:
            print(f"  Main speakers ({len(main_speakers)}): {main_speakers[0]} (target) + {len(main_speakers)-1} random")
        if len(main_speakers) <= 10:
            print(f"    Full list: {', '.join(main_speakers)}")
        else:
            print(f"    Sample: {', '.join(main_speakers[:10])}...")
        print(f"  Other speakers ({len(other_speakers)}): {', '.join(other_speakers[:10])}{'...' if len(other_speakers) > 10 else ''}")
        if BALANCE_OTHER_SPEAKERS_BY_SEX:
            speaker_sex_map = load_speaker_sex_map(speakers_txt_path)
            if speaker_sex_map:
                male_count = sum(1 for s in other_speakers if speaker_sex_map.get(s) == 'M')
                female_count = sum(1 for s in other_speakers if speaker_sex_map.get(s) == 'F')
                unknown_count = len(other_speakers) - male_count - female_count
                print(f"  Other speaker sex distribution: M={male_count}, F={female_count}, Unknown={unknown_count}")
        print(f"  ✓ Speaker separation validated: No intersection between main and other speakers")
    print(f"  Overlapping speakers per sample: {N_OVERLAPPING_SPEAKERS} (always 1)")
    print(f"  Total samples to generate: {TOTAL_SAMPLES}")
    print(f"  Target duration: {TARGET_DURATION if TARGET_DURATION else 'variable (main speaker duration)'}")
    print(f"  Output directory: {output_dir}")
    
    # Check LibriSpeech paths
    missing_paths = [p for p in librispeech_paths if not p.exists()]
    if missing_paths:
        print(f"\n⚠️  Warning: Some LibriSpeech directories not found:")
        for path in missing_paths:
            print(f"    - {path}")
        librispeech_paths = [p for p in librispeech_paths if p.exists()]
        if not librispeech_paths:
            print(f"\n❌ Error: No valid LibriSpeech directories found")
            return
        print(f"\n✓ Continuing with {len(librispeech_paths)} valid dataset(s)")
    
    # Load utterances for all speakers
    print(f"\n🎤 Loading speaker utterances from {len(librispeech_paths)} dataset(s)...")
    
    # Load utterances for all main speakers
    main_speakers_utterances = {}
    
    if not NO_MAIN_SPEAKER:
        # If using single speaker datasets, use those utterances directly
        if single_speaker_utterances_by_id:
            main_speakers_utterances.update(single_speaker_utterances_by_id)
            for spk_id, utts in single_speaker_utterances_by_id.items():
                print(f"  Main speaker {spk_id}: {len(utts)} utterances (from single speaker dataset(s))")
        
        # Load any additional main speakers from LibriSpeech (that aren't already loaded)
        for speaker_id in main_speakers:
            if speaker_id not in main_speakers_utterances:
                utterances = load_speaker_utterances(librispeech_paths, speaker_id)
                main_speakers_utterances[speaker_id] = utterances
                print(f"  Main speaker {speaker_id}: {len(utterances)} utterances")
        
        # Check if any main speaker has no utterances
        empty_main_speakers = [s for s, u in main_speakers_utterances.items() if len(u) == 0]
        if empty_main_speakers:
            print(f"\n❌ Error: No utterances found for main speaker(s): {', '.join(empty_main_speakers)}")
            return
    
    other_speakers_utterances = {}
    for speaker_id in other_speakers:
        utterances = load_speaker_utterances(librispeech_paths, speaker_id)
        other_speakers_utterances[speaker_id] = utterances
        print(f"  Speaker {speaker_id}: {len(utterances)} utterances")
    
    # Filter out speakers with no utterances
    available_other_speakers = [s for s, u in other_speakers_utterances.items() if len(u) > 0]
    
    if NO_MAIN_SPEAKER:
        # In NO_MAIN_SPEAKER mode, need enough speakers for mixing
        required_speakers = N_OVERLAPPING_SPEAKERS + 1
        if len(available_other_speakers) < required_speakers:
            print(f"\n❌ Error: Need at least {required_speakers} speakers for mixing (found {len(available_other_speakers)})")
            return
        print(f"\n✓ {len(available_other_speakers)} speakers available for random mixing")
    else:
        if len(available_other_speakers) < N_OVERLAPPING_SPEAKERS:
            print(f"\n❌ Error: Need at least {N_OVERLAPPING_SPEAKERS} other speakers with utterances")
            print(f"   Found only {len(available_other_speakers)}: {', '.join(available_other_speakers)}")
            return
        
        print(f"\n✓ {len(available_other_speakers)} other speakers available")
    
    # Calculate how many samples needed to use all main speaker utterances
    total_main_utterances = sum(len(utts) for utts in main_speakers_utterances.values()) if not NO_MAIN_SPEAKER else 0
    
    # Determine actual number of samples to generate
    if TOTAL_SAMPLES == 0:
        if NO_MAIN_SPEAKER:
            print(f"\n❌ Error: TOTAL_SAMPLES must be specified (> 0) when using NO_MAIN_SPEAKER mode")
            return
        # Generate enough to use all main speaker utterances at least once
        actual_total_samples = total_main_utterances
        print(f"\n🎵 Auto-generating samples to use all main speaker utterances...")
        print(f"   Total main speaker utterances: {total_main_utterances}")
    else:
        if NO_MAIN_SPEAKER:
            actual_total_samples = TOTAL_SAMPLES
            print(f"\n🎵 Generating {actual_total_samples} random mixture samples...")
        else:
            # Use specified TOTAL_SAMPLES, but ensure we use all main speaker utterances
            actual_total_samples = max(TOTAL_SAMPLES, total_main_utterances)
            if actual_total_samples > TOTAL_SAMPLES:
                print(f"\n🎵 Generating {actual_total_samples} samples (increased from {TOTAL_SAMPLES} to use all main speaker utterances)...")
            else:
                print(f"\n🎵 Generating {actual_total_samples} overlapping samples...")
    
    # Calculate how many samples should have main speaker overlap
    if NO_MAIN_SPEAKER:
        # In NO_MAIN_SPEAKER mode, all samples are random mixtures
        n_main_overlap_samples = 0
        n_non_main_samples = actual_total_samples
        print(f"   All {actual_total_samples} samples will be random speaker mixtures (no target speaker)")
    else:
        n_main_overlap_samples = int(actual_total_samples * (MAIN_SPEAKER_OVERLAP_PERCENTAGE / 100))
        n_non_main_samples = actual_total_samples - n_main_overlap_samples
        
        print(f"   Main speaker overlap samples: {n_main_overlap_samples} ({MAIN_SPEAKER_OVERLAP_PERCENTAGE}%)")
        print(f"   Non-main speaker only samples: {n_non_main_samples} ({100 - MAIN_SPEAKER_OVERLAP_PERCENTAGE}%)")
        print(f"   Total main speaker utterances: {total_main_utterances}")
        print(f"   Each main utterance will be used at least once")
    
    metadata_list = []
    sample_counter = 0
    used_main_speakers = set()
    main_speaker_sample_counts = defaultdict(int)
    main_utterance_usage = defaultdict(int)  # Track how many times each main utterance is used
    
    # Create a list of all main speaker utterances with their speaker IDs
    all_main_utterances = []
    for speaker_id, utterances in main_speakers_utterances.items():
        for utt_path in utterances:
            all_main_utterances.append((speaker_id, utt_path))
    
    # Shuffle to randomize order
    random.shuffle(all_main_utterances)
    
    # Generate main speaker overlap samples
    print(f"\n   Generating {n_main_overlap_samples} main speaker overlap samples...")
    main_utt_idx = 0
    for i in range(n_main_overlap_samples):
        # Cycle through main utterances, ensuring all are used
        main_speaker, main_file = all_main_utterances[main_utt_idx % len(all_main_utterances)]
        main_utt_idx += 1
        
        # Select random other speaker (different from main)
        available_others = [s for s in available_other_speakers if s != main_speaker]
        if not available_others:
            print(f"    ⚠️  No other speakers available (all are main speakers)")
            continue
        
        other_speaker = random.choice(available_others)
        other_utterances = other_speakers_utterances[other_speaker]
        
        if sample_counter % 50 == 0 and sample_counter > 0:
            print(f"    Progress: {sample_counter}/{actual_total_samples} samples...")
        
        # Select random utterance from the other speaker
        other_file = random.choice(other_utterances)
        other_files = [other_file]
        
        # Generate mixed sample
        try:
            mixed_audio, labels, metadata = generate_overlap_sample(
                main_file,
                other_files,
                main_speaker,
                target_duration=TARGET_DURATION,
                main_volume=MAIN_SPEAKER_VOLUME,
                other_volume=OTHER_SPEAKER_VOLUME
            )
            
            # Track used main speakers
            used_main_speakers.add(main_speaker)
            main_speaker_sample_counts[main_speaker] += 1
            main_utterance_usage[main_file] += 1
            
            # Create output filename
            sample_id = f"sample_{sample_counter:04d}"
            output_name = f"{sample_id}_main{main_speaker}_other{other_speaker}"
            sample_counter += 1
        
            # Save audio
            audio_path = audio_dir / f"{output_name}.wav"
            sf.write(audio_path, mixed_audio, SAMPLE_RATE)
            
            # Save labels
            label_path = label_dir / f"{output_name}_labels.npy"
            np.save(label_path, labels)
            
            # Store metadata
            metadata['sample_id'] = sample_id
            metadata['output_name'] = output_name
            metadata['has_main_speaker'] = True
            metadata_list.append(metadata)
                
        except Exception as e:
            print(f"    ⚠️  Error generating sample {sample_counter}: {e}")
            continue
    
    # Generate non-main speaker only samples
    if n_non_main_samples > 0:
        print(f"\n   Generating {n_non_main_samples} non-main speaker only samples...")
        for i in range(n_non_main_samples):
            # Select two random other speakers (both different from main speakers)
            if len(available_other_speakers) < 2:
                print(f"    ⚠️  Need at least 2 other speakers for non-main samples")
                break
            
            # Select two different other speakers
            selected_others = random.sample(available_other_speakers, 2)
            primary_other = selected_others[0]
            secondary_other = selected_others[1]
            
            primary_utterances = other_speakers_utterances[primary_other]
            secondary_utterances = other_speakers_utterances[secondary_other]
            
            if sample_counter % 50 == 0 and sample_counter > 0:
                print(f"    Progress: {sample_counter}/{actual_total_samples} samples...")
            
            # Select random utterances
            primary_file = random.choice(primary_utterances)
            secondary_file = random.choice(secondary_utterances)
            
            # Generate mixed sample (primary as "main", but it's not actually a main speaker)
            try:
                mixed_audio, labels, metadata = generate_overlap_sample(
                    primary_file,
                    [secondary_file],
                    primary_other,  # Use as "main" for mixing purposes
                    target_duration=TARGET_DURATION,
                    main_volume=MAIN_SPEAKER_VOLUME,
                    other_volume=OTHER_SPEAKER_VOLUME
                )
                
                # Override labels: since no main speaker is present, mark as non-target
                # 0=silence, 1=non-target speech, 2=target speech
                labels = np.where(labels == 2, 1, labels)  # Change target labels to non-target
                
                # Create output filename
                sample_id = f"sample_{sample_counter:04d}"
                output_name = f"{sample_id}_nonmain_{primary_other}_{secondary_other}"
                sample_counter += 1
            
                # Save audio
                audio_path = audio_dir / f"{output_name}.wav"
                sf.write(audio_path, mixed_audio, SAMPLE_RATE)
                
                # Save labels
                label_path = label_dir / f"{output_name}_labels.npy"
                np.save(label_path, labels)
                
                # Store metadata
                metadata['sample_id'] = sample_id
                metadata['output_name'] = output_name
                metadata['has_main_speaker'] = False
                metadata['main_speaker'] = None
                metadata['other_speakers'] = [primary_other, secondary_other]
                metadata_list.append(metadata)
                    
            except Exception as e:
                print(f"    ⚠️  Error generating non-main sample {sample_counter}: {e}")
                continue
    
    # Save metadata
    import json
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    # Save speaker configuration
    n_main_overlap = sum(1 for m in metadata_list if m.get('has_main_speaker', True))
    n_non_main = len(metadata_list) - n_main_overlap
    
    speaker_config = {
        'main_speakers': sorted(list(used_main_speakers)),
        'target_speaker': TARGET_SPEAKER if MAIN_SPEAKERS is None else None,
        'main_speaker': sorted(list(used_main_speakers))[0] if used_main_speakers else None,  # For backward compatibility
        'other_speakers': available_other_speakers,
        'n_main_speakers': len(used_main_speakers),
        'n_other_speakers': len(available_other_speakers),
        'total_speakers': len(used_main_speakers) + len(available_other_speakers),
        'main_selection_method': 'explicit' if MAIN_SPEAKERS is not None else 'target_plus_random',
        'other_selection_method': 'explicit' if OTHER_SPEAKERS is not None else 'automatic',
        'balance_other_speakers_by_sex': BALANCE_OTHER_SPEAKERS_BY_SEX,
        'speakers_txt_path': str(speakers_txt_path),
        'random_seed': RANDOM_SEED,
        'excluded_speakers': EXCLUDE_SPEAKERS,
        'generation_mode': 'mixed_overlap',
        'total_samples': len(metadata_list),
        'main_overlap_samples': n_main_overlap,
        'non_main_samples': n_non_main,
        'main_overlap_percentage': MAIN_SPEAKER_OVERLAP_PERCENTAGE,
        'total_main_utterances': total_main_utterances,
        'main_speaker_sample_counts': dict(main_speaker_sample_counts),
        'unique_main_utterances_used': len(main_utterance_usage)
    }
    speaker_config_path = output_dir / 'speaker_config.json'
    with open(speaker_config_path, 'w') as f:
        json.dump(speaker_config, f, indent=2)
    
    print(f"\n✓ Generated {len(metadata_list)} samples")
    print(f"\n📁 Output files:")
    print(f"  Audio: {audio_dir}")
    print(f"  Labels: {label_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Speaker config: {speaker_config_path}")
    
    # Generate summary statistics
    print(f"\n📊 Summary Statistics:")
    durations = [m['duration'] for m in metadata_list]
    print(f"  Average duration: {np.mean(durations):.2f}s")
    print(f"  Min duration: {np.min(durations):.2f}s")
    print(f"  Max duration: {np.max(durations):.2f}s")
    
    # Count samples by type
    main_overlap_count = sum(1 for m in metadata_list if m.get('has_main_speaker', True))
    non_main_count = len(metadata_list) - main_overlap_count
    print(f"\n  Sample distribution:")
    print(f"    Main speaker overlap: {main_overlap_count} samples ({(main_overlap_count / len(metadata_list)) * 100:.1f}%)")
    print(f"    Non-main speaker only: {non_main_count} samples ({(non_main_count / len(metadata_list)) * 100:.1f}%)")
    
    # Main speaker usage statistics
    if main_utterance_usage:
        usage_counts = list(main_utterance_usage.values())
        print(f"\n  Main speaker utterance usage:")
        print(f"    Unique utterances used: {len(main_utterance_usage)} / {total_main_utterances}")
        print(f"    Min reuse count: {min(usage_counts)}")
        print(f"    Max reuse count: {max(usage_counts)}")
        print(f"    Avg reuse count: {np.mean(usage_counts):.2f}")
    
    print(f"\n  Main speaker sample distribution:")
    for main_spk in sorted(used_main_speakers):
        count = main_speaker_sample_counts[main_spk]
        pct = (count / main_overlap_count) * 100 if main_overlap_count > 0 else 0
        marker = " (TARGET)" if main_spk == TARGET_SPEAKER else ""
        
        # Count unique utterances for this speaker
        spk_utts = [utt for utt in main_utterance_usage.keys() if Path(utt).parts[-3] == main_spk]
        print(f"    Speaker {main_spk}{marker}: {count} samples ({pct:.1f}% of main overlap), {len(spk_utts)} unique utterances")
    
    print("\n" + "=" * 70)
    print(f"\n✅ GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated dataset with:")
    if TARGET_SPEAKER and MAIN_SPEAKERS is None:
        print(f"  • Target speaker: {TARGET_SPEAKER}")
        print(f"  • Additional main speakers: {len(used_main_speakers) - 1} random speakers")
    print(f"  • Total main speakers ({len(used_main_speakers)}): {', '.join(sorted(used_main_speakers))}")
    print(f"  • Other speakers: {len(available_other_speakers)}")
    print(f"  • Total samples: {len(metadata_list)}")
    print(f"    - Main speaker overlap: {main_overlap_count} ({MAIN_SPEAKER_OVERLAP_PERCENTAGE}%)")
    print(f"    - Non-main speaker only: {non_main_count} ({100 - MAIN_SPEAKER_OVERLAP_PERCENTAGE}%)")
    print(f"  • Main speaker utterances used: {len(main_utterance_usage)} / {total_main_utterances}")
    print(f"  • Overlap type: FULL (speakers talk simultaneously for entire duration)")
    print(f"  • ✓ Speaker separation: Main and other speakers are mutually exclusive")
    print(f"\nUse these samples to test:")
    print(f"  - Speaker separation with continuous overlap")
    print(f"  - Target speaker detection in multi-speaker scenarios")
    print(f"  - PersonalVAD performance with full overlap")


if __name__ == "__main__":
    generate_dataset()
