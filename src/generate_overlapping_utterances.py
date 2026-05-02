#!/usr/bin/env python3
"""
Generate simple overlapping utterances for PersonalVAD training.
Simple approach:
1. Choose a target speaker utterance
2. Choose a non-target speaker utterance  
3. Overlap them based on overlap percentage
4. Pad with silence
5. Label each frame according to speech/silence from alignments
"""

import argparse as ap
import math
import os
import random
import re
import sys
import numpy as np
import soundfile as sf

# Constants
OVERLAP_PERCENTAGE = 50.0  # Default overlap %
OVERLAP_AMPLITUDE_RATIO = 1.0  # Amplitude ratio for overlap speaker
FILES_PER_DIR = 500
FLAC = True

def get_speech_ranges(aligned_text, stamps):
    """
    Parse alignment data to get speech ranges.
    
    LibriSpeech format: ",WORD1,,WORD2," with timestamps "t1,t2,t3"
    - Commas separate segments
    - Empty string (between commas) = silence
    - Word = speech
    - Timestamps define segment boundaries (need to prepend 0.0)
    
    Args:
        aligned_text: Comma-separated aligned text (e.g., ",GO,,DO,YOU,HEAR,")
        stamps: List of timestamp strings marking segment boundaries
        
    Returns:
        List of (start, end) tuples for speech segments
    """
    segments = aligned_text.split(',')
    speech_ranges = []
    
    # Timestamps need 0.0 prepended for the start
    timestamps = [0.0] + [float(t.strip().strip('"')) for t in stamps]
    
    # Each segment[i] describes the region between timestamps[i] and timestamps[i+1]
    for i, segment in enumerate(segments):
        if i + 1 < len(timestamps):
            if segment.strip():  # Non-empty = speech
                start_time = timestamps[i]
                end_time = timestamps[i + 1]
                if end_time > start_time:
                    speech_ranges.append((start_time, end_time))
    
    return speech_ranges


def load_single_speaker_dataset(dataset_dir, speaker_id, max_utterances=None):
    """
    Load utterances from a single speaker dataset (created by extract_single_speaker.py).
    
    The dataset should have structure:
        dataset_dir/
            wav.scp
            text (with alignment info)
            audio/*.flac
            
    Args:
        dataset_dir: Path to the single speaker dataset directory
        speaker_id: Expected speaker ID
        max_utterances: Optional maximum number of utterances to load (randomly sampled)
        
    Returns:
        List of utterances in the same format as load_speaker_utterances
        (audio_path, utt_id, aligned_text, stamps, transcript)
    """
    utterances = []
    
    wav_scp_path = os.path.join(dataset_dir, 'wav.scp')
    text_path = os.path.join(dataset_dir, 'text')
    
    if not os.path.exists(wav_scp_path) or not os.path.exists(text_path):
        print(f"  Warning: Single speaker dataset not found in {dataset_dir}")
        return utterances
    
    # Load wav.scp
    audio_paths = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id = parts[0]
                audio_path = parts[1]
                audio_paths[utt_id] = audio_path
    
    # Load text file with alignments
    # Format: utt_id [aligned_text] [timestamps]
    with open(text_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
                
            utt_id = parts[0]
            rest = parts[1]
            
            # Parse [aligned_text] [timestamps]
            # Example: [WW  W] [0.00,0.10,0.20,0.30,0.40,0.50]
            if '[' not in rest or ']' not in rest:
                continue
                
            # Extract aligned_text and timestamps
            start_align = rest.index('[')
            end_align = rest.index(']', start_align)
            aligned_text = rest[start_align+1:end_align]
            
            remainder = rest[end_align+1:].strip()
            if '[' in remainder and ']' in remainder:
                start_stamps = remainder.index('[')
                end_stamps = remainder.index(']', start_stamps)
                stamps_str = remainder[start_stamps+1:end_stamps]
            else:
                continue
            
            if utt_id in audio_paths:
                # Convert aligned_text to comma-separated format for consistency
                # Our format: "WWW " -> ",W,W,W,,"
                comma_separated = ',' + ','.join(aligned_text) + ','
                
                utterances.append((
                    audio_paths[utt_id],
                    utt_id,
                    comma_separated,
                    stamps_str,
                    ""  # No transcript text needed
                ))
    
    print(f"  Loaded {len(utterances)} utterances from single speaker dataset")
    
    # Randomly sample if max_utterances is specified
    if max_utterances and max_utterances > 0 and len(utterances) > max_utterances:
        print(f"  Randomly sampling {max_utterances} utterances from {len(utterances)} total")
        utterances = random.sample(utterances, max_utterances)
    
    return utterances


def load_speaker_utterances(libri_root, speaker_id, available_sets=None):
    """
    Load all utterances for a specific speaker across all available LibriSpeech sets.
    
    Args:
        libri_root: Path to LibriSpeech dataset
        speaker_id: Speaker ID to search for
        available_sets: Optional list of sets to search (default: search all)
        
    Returns:
        List of utterances for the speaker, or empty list if not found
    """
    if available_sets is None:
        # Auto-discover all LibriSpeech subsets
        available_sets = []
        if os.path.exists(libri_root):
            for item in os.listdir(libri_root):
                item_path = os.path.join(libri_root, item)
                # Look for directories like dev-clean, test-other, train-clean-100, etc.
                if os.path.isdir(item_path) and (item.startswith('dev-') or item.startswith('test-') or item.startswith('train-')):
                    available_sets.append(item)
        
        if available_sets:
            print(f"  Auto-discovered LibriSpeech subsets: {', '.join(sorted(available_sets))}")
        else:
            print(f"  Warning: No LibriSpeech subsets found in {libri_root}")
    
    utterances = []
    found_in_sets = []
    
    for libri_set in available_sets:
        speaker_path = os.path.join(libri_root, libri_set, speaker_id)
        if not os.path.exists(speaker_path):
            continue
            
        found_in_sets.append(libri_set)
        
        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)
            if not os.path.isdir(chapter_path):
                continue
                
            # Look for alignment file in the same directory as audio files
            alignment_path = os.path.join(chapter_path, f"{speaker_id}-{chapter}.alignment.txt")
            
            if not os.path.exists(alignment_path):
                # Try alternative locations (older structure with separate Alignments folder)
                alt_paths = [
                    os.path.join(libri_root, f"LibriSpeech-Alignments/{libri_set}/{speaker_id}/{chapter}.alignment.txt"),
                    os.path.join(os.path.dirname(libri_root), f"LibriSpeech-Alignments/{libri_set}/{speaker_id}/{chapter}.alignment.txt"),
                ]
                for path in alt_paths:
                    if os.path.exists(path):
                        alignment_path = path
                        break
                else:
                    # No alignment file found - skip this chapter
                    continue
                
            alignments = {}
            with open(alignment_path, 'r') as align_file:
                for line in align_file:
                    parts = line.strip().split(' ')
                    if len(parts) >= 3:
                        utt_id = parts[0]
                        aligned_text = parts[1]
                        stamps = ' '.join(parts[2:])
                        alignments[utt_id] = (aligned_text, stamps)
            
            # Load transcript
            transcript_path = os.path.join(chapter_path, f"{speaker_id}-{chapter}.trans.txt")
            transcripts = {}
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r') as trans_file:
                    for line in trans_file:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]
            
            # Load audio files
            for audio_file in os.listdir(chapter_path):
                if not audio_file.endswith('.flac'):
                    continue
                    
                utt_id = audio_file.replace('.flac', '')
                if utt_id not in alignments:
                    continue
                    
                audio_path = os.path.join(chapter_path, audio_file)
                aligned_text, stamps = alignments[utt_id]
                transcript = transcripts.get(utt_id, '')
                
                utterances.append((audio_path, utt_id, aligned_text, stamps, transcript))
    
    if utterances:
        print(f"  Found speaker {speaker_id} in: {', '.join(found_in_sets)} ({len(utterances)} utterances)")
    else:
        if found_in_sets:
            print(f"  Speaker {speaker_id} directory found in {', '.join(found_in_sets)} but no valid utterances (missing alignments?)")
    
    return utterances


def generate_overlapping_utterances(libri_root, dest, n, overlap_pct=50.0, 
                                   amplitude_ratio=1.0, wav_scp_prefix='', 
                                   sets=None, base_speakers=None, no_target_speaker=False,
                                   single_speaker_datasets=None, single_speaker_count=None):
    """
    Generate overlapping utterances with simple logic.
    
    Args:
        libri_root: Path to LibriSpeech dataset
        dest: Output directory
        n: Number of samples to generate
        overlap_pct: Percentage of target speech to overlap (0-100)
        amplitude_ratio: Amplitude multiplier for overlap speaker
        wav_scp_prefix: Prefix for wav.scp paths
        sets: List of LibriSpeech subsets to use (for non-target speakers)
        base_speakers: Optional list of specific speaker IDs to use as target
        no_target_speaker: If True, all speakers are non-target (no 'T' labels)
        single_speaker_datasets: List of paths to single speaker dataset directories
        single_speaker_count: Optional max number of utterances to use from EACH single speaker dataset
    """
    if sets is None:
        sets = ['train-clean-100']
    
    # Load dataset (for non-target speakers)
    print(f"Loading LibriSpeech from {libri_root}...")
    dataset = []
    
    for libri_set in sets:
        speakers_path = os.path.join(libri_root, libri_set)
        if not os.path.exists(speakers_path):
            print(f"Warning: Path not found: {speakers_path}")
            continue
            
        for speaker in os.listdir(speakers_path):
            speaker_path = os.path.join(speakers_path, speaker)
            if not os.path.isdir(speaker_path):
                continue
                
            utterances = []
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                if not os.path.isdir(chapter_path):
                    continue
                    
                # Read alignment file
                align_file = os.path.join(chapter_path, f"{speaker}-{chapter}.alignment.txt")
                if not os.path.exists(align_file):
                    continue
                    
                with open(align_file, 'r') as f:
                    for line in f:
                        # Format: utt_id "aligned_text" "timestamps"
                        parts = line.strip().split(' ', 1)
                        if len(parts) < 2:
                            continue
                        
                        utt_id = parts[0]
                        remainder = parts[1]
                        
                        # Extract aligned_text and timestamps from quoted strings
                        matches = re.findall(r'"([^"]*)"', remainder)
                        if len(matches) < 2:
                            continue
                        
                        aligned_text = matches[0]
                        timestamps = matches[1]
                        audio_path = os.path.join(chapter_path, f"{utt_id}.flac")
                        
                        if os.path.exists(audio_path):
                            # Read transcript
                            trans_file = os.path.join(chapter_path, f"{speaker}-{chapter}.trans.txt")
                            transcript = ""
                            if os.path.exists(trans_file):
                                with open(trans_file, 'r') as tf:
                                    for tline in tf:
                                        if tline.startswith(utt_id):
                                            transcript = tline.split(' ', 1)[1].strip()
                                            break
                            
                            utterances.append((audio_path, utt_id, aligned_text, timestamps, transcript))
            if utterances:
                dataset.append([f"{speaker}", utterances])
    
    print(f"Loaded {len(dataset)} speakers with utterances")
    
    if len(dataset) < 2:
        print("Error: Need at least 2 speakers in dataset")
        return set()
    
    # Setup single speaker dataset if specified (highest priority)
    single_speaker_data = None
    single_speaker_ids = []
    
    if single_speaker_datasets:
        print(f"\nLoading {len(single_speaker_datasets)} single speaker dataset(s)...")
        
        single_speaker_data = {}
        
        for dataset_path in single_speaker_datasets:
            # Extract speaker ID from utt2spk file
            speaker_id = None
            utt2spk_path = os.path.join(dataset_path, 'utt2spk')
            if os.path.exists(utt2spk_path):
                with open(utt2spk_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        speaker_id = first_line.split()[1]
            
            if speaker_id:
                utterances = load_single_speaker_dataset(dataset_path, speaker_id, single_speaker_count)
                if utterances:
                    if speaker_id not in single_speaker_data:
                        single_speaker_data[speaker_id] = {
                            'utterances': [],
                            'original_utterances': [],
                            'external': True
                        }
                    # Append to existing or create new entry
                    single_speaker_data[speaker_id]['utterances'].extend(utterances)
                    single_speaker_data[speaker_id]['original_utterances'].extend(list(utterances))
                    single_speaker_ids.append(speaker_id)
                    print(f"  ✓ Loaded speaker {speaker_id} from {dataset_path}: {len(utterances)} utterances")
                else:
                    print(f"  ⚠ Warning: Failed to load utterances from {dataset_path}")
            else:
                print(f"  ⚠ Warning: Could not extract speaker ID from {dataset_path}")
    
    # Setup base speakers if specified (search ALL LibriSpeech sets)
    base_speakers_list = None
    base_speakers_data = {}
    
    if no_target_speaker:
        if base_speakers or single_speaker_datasets:
            speaker_info = ', '.join(single_speaker_ids) if single_speaker_datasets else base_speakers
            print(f"\nMode: NO TARGET SPEAKERS - All speech labeled as non-target (N)")
            print(f"Speaker(s) {speaker_info} listed in metadata but NOT present in audio")
        else:
            print(f"\nMode: NO TARGET SPEAKERS - All speech labeled as non-target (N)")
    elif single_speaker_data:
        # Single speaker dataset mode
        total_utterances = sum(len(data['utterances']) for data in single_speaker_data.values())
        unique_speakers = list(set(single_speaker_ids))
        print(f"\nUsing {len(unique_speakers)} speaker(s) from single speaker dataset(s): {', '.join(unique_speakers)}")
        print(f"  Total utterances loaded: {total_utterances}")
        base_speakers_data = single_speaker_data
    elif base_speakers:
        base_speakers_list = base_speakers.split(',')
        print(f"\nSearching for base speakers across ALL LibriSpeech datasets...")
        
        for speaker_id in base_speakers_list:
            # Try to find in already loaded dataset first
            found_in_dataset = False
            for idx, spk in enumerate(dataset):
                if spk[0] == speaker_id:
                    base_speakers_data[speaker_id] = {
                        'utts': list(spk[1]),
                        'idx': idx,
                        'external': False
                    }
                    found_in_dataset = True
                    print(f"  Speaker {speaker_id}: Found in loaded datasets")
                    dataset[idx] = [spk[0], list(spk[1])]
                    break
            
            # If not found, search all available LibriSpeech sets
            if not found_in_dataset:
                utterances = load_speaker_utterances(libri_root, speaker_id)
                if utterances:
                    # Add as external speaker (not in main dataset list)
                    base_speakers_data[speaker_id] = {
                        'external': True,
                        'utterances': list(utterances),
                        'original_utterances': list(utterances)  # Keep copy for reuse
                    }
                else:
                    print(f"  ⚠️  Speaker {speaker_id}: NOT FOUND in any LibriSpeech dataset")
                    print(f"      Skipping this speaker...")
        
        if not base_speakers_data:
            print("Error: None of the specified base speakers were found")
            return set()
        
        print(f"Using {len(base_speakers_data)} base speaker(s): {', '.join(base_speakers_data.keys())}")
    
    # Create output directory
    os.makedirs(dest, exist_ok=True)
    
    # Open output files
    wav_scp_path = os.path.join(dest, 'wav.scp')
    utt2spk_path = os.path.join(dest, 'utt2spk')
    text_path = os.path.join(dest, 'text')
    
    wav_scp = open(wav_scp_path, 'w')
    utt2spk = open(utt2spk_path, 'w')
    text = open(text_path, 'w')
    
    used_speakers = set()
    sr = 16000
    
    # Dataset-level statistics
    dataset_total_duration = 0.0
    dataset_target_duration = 0.0
    dataset_non_target_duration = 0.0
    dataset_silence_duration = 0.0
    dataset_overlap_duration = 0.0
    dataset_num_standalone_utts = 0
    
    # Generate samples
    for iteration in range(n):
        # Create subdirectory every FILES_PER_DIR samples
        if iteration % FILES_PER_DIR == 0:
            scp_path = f"{iteration // FILES_PER_DIR}_overlap/"
            cur_dir = os.path.join(dest, scp_path)
            os.makedirs(cur_dir, exist_ok=True)
        
        # Select speakers based on mode
        if no_target_speaker:
            # No target speaker mode: all speech labeled as non-target
            # If base speaker specified, use ONLY for metadata (not in audio)
            if base_speakers_list:
                # Use base speaker ID for metadata only
                current_speaker_id = base_speakers_list[iteration % len(base_speakers_list)]
                
                # Select 2 random speakers (EXCLUDING the base speaker)
                if len(dataset) < 2:
                    print(f"Error: Need at least 2 speakers but only have {len(dataset)}")
                    continue
                
                # Filter out the base speaker from available speakers
                available_speakers = [s for s in dataset if s[0] != current_speaker_id]
                if len(available_speakers) < 2:
                    print(f"Error: Need at least 2 speakers besides base speaker {current_speaker_id}")
                    continue
                
                speaker1 = random.choice(available_speakers)
                speaker2 = random.choice([s for s in available_speakers if s[0] != speaker1[0]])
                
                if not speaker1[1] or not speaker2[1]:
                    print(f"Skipping: One of the speakers has no utterances left")
                    continue
                
                target_utt = random.choice(speaker1[1])
                speaker1[1].remove(target_utt)
                
                overlap_utt = random.choice(speaker2[1])
                speaker2[1].remove(overlap_utt)
                overlap_speaker_id = speaker2[0]
            else:
                # No base speaker specified: select 2 random non-target speakers
                if len(dataset) < 2:
                    print(f"Error: Need at least 2 speakers but only have {len(dataset)}")
                    continue
                
                speaker1 = random.choice(dataset)
                speaker2 = random.choice([s for s in dataset if s[0] != speaker1[0]])
                
                if not speaker1[1] or not speaker2[1]:
                    print(f"Skipping: One of the speakers has no utterances left")
                    continue
                
                target_utt = random.choice(speaker1[1])
                speaker1[1].remove(target_utt)
                current_speaker_id = "nontarget"  # Generic ID when no base speaker
                
                overlap_utt = random.choice(speaker2[1])
                speaker2[1].remove(overlap_utt)
                overlap_speaker_id = speaker2[0]
        
        elif base_speakers_list or base_speakers_data:
            # Handle both base_speakers (comma-separated list) and single_speaker_dataset
            if base_speakers_list:
                current_speaker_id = base_speakers_list[iteration % len(base_speakers_list)]
            else:
                # Single speaker dataset mode - use the only speaker in base_speakers_data
                current_speaker_id = list(base_speakers_data.keys())[0]
            
            speaker_data = base_speakers_data[current_speaker_id]
            
            # Check if this is an external speaker (not in main dataset)
            if speaker_data.get('external', False):
                # Use external utterances directly
                if not speaker_data.get('utterances'):
                    # Reload utterances from original list
                    print(f"Reloading utterances for external speaker {current_speaker_id}...")
                    speaker_data['utterances'] = list(speaker_data['original_utterances'])
                    if not speaker_data.get('utterances'):
                        print(f"No utterances available for external speaker {current_speaker_id}, skipping...")
                        continue
                
                target_utt = random.choice(speaker_data['utterances'])
                speaker_data['utterances'].remove(target_utt)
                target_speaker = None  # Mark as external
            else:
                # Use speaker from main dataset
                speaker_idx = speaker_data['idx']
                target_speaker = dataset[speaker_idx]
                
                # Reload utterances if needed
                if not target_speaker[1]:
                    print(f"Reloading utterances for speaker {current_speaker_id}...")
                    target_speaker[1] = list(speaker_data['utts'])
                
                if not target_speaker[1]:
                    print(f"No utterances available for speaker {current_speaker_id}, skipping...")
                    continue
                
                # Select one target utterance
                target_utt = random.choice(target_speaker[1])
                target_speaker[1].remove(target_utt)
        else:
            target_speaker = random.choice(dataset)
            current_speaker_id = target_speaker[0]
            
            if not target_speaker[1]:
                print(f"No utterances left for speaker {current_speaker_id}, skipping...")
                continue
            
            # Select one target utterance
            target_utt = random.choice(target_speaker[1])
            target_speaker[1].remove(target_utt)
        
        target_audio_path, target_id, target_aligned_text, target_stamps, target_transcript = target_utt
        
        # Load target audio
        target_audio, sr = sf.read(target_audio_path)
        assert sr == 16000, f"Invalid sample rate {sr}"
        
        # Get target speech ranges
        target_stamps_list = target_stamps.split(',')
        target_speech_ranges = get_speech_ranges(target_aligned_text, target_stamps_list)
        
        # DEBUG: Print first few samples
        if iteration < 3:
            print(f"\n[DEBUG] Sample {iteration}:")
            print(f"  Target ID: {target_id}")
            print(f"  Target duration: {target_audio.size / sr:.2f}s")
            print(f"  Aligned text: {target_aligned_text[:100]}")
            print(f"  Speech ranges (before adjustment): {target_speech_ranges[:5]}")
        
        # Calculate total target speech duration
        target_speech_duration = sum(end - start for start, end in target_speech_ranges)
        
        if target_speech_duration == 0:
            print(f"Warning: Target utterance {target_id} has no speech, skipping...")
            continue
        
        # Select non-target speaker
        non_target_speakers = [s for s in dataset if s[0] != current_speaker_id]
        if not non_target_speakers:
            print("No non-target speakers available, skipping...")
            continue
        
        overlap_speaker = random.choice(non_target_speakers)
        if not overlap_speaker[1]:
            print("No utterances for overlap speaker, skipping...")
            continue
        
        overlap_utt = random.choice(overlap_speaker[1])
        overlap_speaker[1].remove(overlap_utt)
        
        overlap_audio_path, overlap_id, overlap_aligned_text, overlap_stamps, overlap_transcript = overlap_utt
        overlap_speaker_id = overlap_speaker[0]
        
        # Load overlap audio
        overlap_audio, sr_ov = sf.read(overlap_audio_path)
        assert sr_ov == 16000, f"Invalid sample rate {sr_ov}"
        
        # Get overlap speech ranges
        overlap_stamps_list = overlap_stamps.split(',')
        overlap_speech_ranges = get_speech_ranges(overlap_aligned_text, overlap_stamps_list)
        
        # Calculate total target speech duration
        target_duration = target_audio.size / sr
        overlap_duration = overlap_audio.size / sr
        
        # Add standalone non-target speech utterances to fill silence gaps
        # This ensures we have roughly equal amounts of overlapped and standalone non-target speech
        standalone_utts = []
        standalone_speech_ranges = []
        
        # Calculate how much standalone non-target speech we need
        # For 50% overlap, we want approximately equal non-target speech (overlapped + standalone)
        target_speech_duration_total = sum(end - start for start, end in target_speech_ranges)
        desired_standalone_duration = target_speech_duration_total * (overlap_pct / 100.0)
        
        # Collect standalone non-target utterances
        current_standalone_duration = 0.0
        while current_standalone_duration < desired_standalone_duration:
            if not non_target_speakers:
                break
                
            standalone_speaker = random.choice(non_target_speakers)
            if not standalone_speaker[1]:
                non_target_speakers.remove(standalone_speaker)
                continue
                
            standalone_utt = random.choice(standalone_speaker[1])
            standalone_speaker[1].remove(standalone_utt)
            
            sa_audio_path, sa_id, sa_aligned_text, sa_stamps, sa_transcript = standalone_utt
            sa_audio, sa_sr = sf.read(sa_audio_path)
            assert sa_sr == 16000, f"Invalid sample rate {sa_sr}"
            
            sa_duration = sa_audio.size / sr
            sa_stamps_list = sa_stamps.split(',')
            sa_speech_ranges = get_speech_ranges(sa_aligned_text, sa_stamps_list)
            
            standalone_utts.append((sa_audio, sa_speech_ranges, sa_duration, sa_id))
            current_standalone_duration += sum(end - start for start, end in sa_speech_ranges)
        
        # Position overlap utterance to achieve desired overlap percentage
        if overlap_pct == 0:
            # For 0% overlap: place in silence/gaps
            overlap_start_time = target_duration + 0.5
        elif target_speech_ranges:
            # Position to overlap with target speech
            target_speech_start = target_speech_ranges[0][0]
            target_speech_end = target_speech_ranges[-1][1]
            target_speech_span = target_speech_end - target_speech_start
            
            max_start = max(0, target_speech_span - overlap_duration * 0.3)
            overlap_start_time = target_speech_start + random.uniform(0, max(0.1, max_start))
        else:
            overlap_start_time = 0.0
        
        overlap_end_time = overlap_start_time + overlap_duration
        
        # Position standalone utterances in available gaps/silence
        # Find silence regions in target speech
        silence_regions = []
        if target_speech_ranges:
            if target_speech_ranges[0][0] > 0.5:
                silence_regions.append((0, target_speech_ranges[0][0]))
            
            for i in range(len(target_speech_ranges) - 1):
                gap_start = target_speech_ranges[i][1]
                gap_end = target_speech_ranges[i + 1][0]
                if gap_end - gap_start > 0.3:
                    silence_regions.append((gap_start, gap_end))
            
            silence_regions.append((target_speech_ranges[-1][1], target_duration + 10.0))
        else:
            silence_regions.append((0, target_duration + 10.0))
        
        # Place standalone utterances
        positioned_standalone = []
        for sa_audio, sa_ranges, sa_duration, sa_id in standalone_utts:
            # Find suitable gap
            placed = False
            random.shuffle(silence_regions)
            
            for gap_start, gap_end in silence_regions:
                if gap_end - gap_start >= sa_duration:
                    # Place in this gap
                    sa_start = gap_start + random.uniform(0, min(1.0, gap_end - gap_start - sa_duration))
                    positioned_standalone.append((sa_audio, sa_ranges, sa_start, sa_duration, sa_id))
                    placed = True
                    break
            
            if not placed:
                # Place after everything else
                last_time = max(target_duration, overlap_end_time)
                if positioned_standalone:
                    last_time = max(last_time, max(s[2] + s[3] for s in positioned_standalone))
                sa_start = last_time + random.uniform(0.1, 0.5)
                positioned_standalone.append((sa_audio, sa_ranges, sa_start, sa_duration, sa_id))
        
        # Calculate total duration needed
        max_end_time = target_duration
        if overlap_end_time > max_end_time:
            max_end_time = overlap_end_time
        for _, _, sa_start, sa_duration, _ in positioned_standalone:
            if sa_start + sa_duration > max_end_time:
                max_end_time = sa_start + sa_duration
        
        # Add padding to make chunks of 15 seconds
        chunk_duration = 15.0  # seconds
        
        # Calculate preliminary total duration to determine available space
        # First, assume target starts at time 0 to get content bounds
        preliminary_max_end = max(target_duration, overlap_end_time)
        for _, _, sa_start, sa_duration, _ in positioned_standalone:
            if sa_start + sa_duration > preliminary_max_end:
                preliminary_max_end = sa_start + sa_duration
        
        # Calculate how many chunks we need (with some buffer)
        n_chunks = int(np.ceil((preliminary_max_end + 2.0) / chunk_duration))
        total_duration = n_chunks * chunk_duration
        
        # Randomly choose target speaker start time within available space
        # Leave room for target audio to fit completely
        max_target_start = max(0.1, total_duration - target_duration - 0.5)
        target_start_time = random.uniform(0.1, max_target_start)
        target_end_time = target_start_time + target_duration
        
        # Build complete audio
        total_samples = int(total_duration * sr)
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        # Add target audio (at randomly chosen start time)
        target_start_sample = int(target_start_time * sr)
        target_end_sample = target_start_sample + target_audio.size
        final_audio[target_start_sample:target_end_sample] += target_audio
        
        # Add overlap audio (adjusted relative to target start time, not leading silence)
        # overlap_start_time is relative to target audio start (0), so adjust it
        overlap_start_time_adjusted = overlap_start_time + target_start_time
        overlap_end_time_adjusted = overlap_end_time + target_start_time
        overlap_start_sample = int(overlap_start_time_adjusted * sr)
        overlap_end_sample = min(overlap_start_sample + overlap_audio.size, total_samples)
        overlap_samples_to_add = overlap_end_sample - overlap_start_sample
        
        if overlap_samples_to_add > 0:
            final_audio[overlap_start_sample:overlap_end_sample] += (
                overlap_audio[:overlap_samples_to_add] * amplitude_ratio
            )
        
        # Add standalone utterances
        # These were positioned relative to target at time 0, so adjust to target's actual position
        positioned_standalone_adjusted = []
        for sa_audio, sa_ranges, sa_start, sa_duration, sa_id in positioned_standalone:
            sa_start_adjusted = sa_start + target_start_time
            sa_start_sample = int(sa_start_adjusted * sr)
            sa_end_sample = min(sa_start_sample + sa_audio.size, total_samples)
            sa_samples_to_add = sa_end_sample - sa_start_sample
            
            if sa_samples_to_add > 0:
                final_audio[sa_start_sample:sa_end_sample] += (
                    sa_audio[:sa_samples_to_add] * amplitude_ratio
                )
            
            # Store adjusted position for later use in labeling
            positioned_standalone_adjusted.append((sa_audio, sa_ranges, sa_start_adjusted, sa_duration, sa_id))
        
        # Replace positioned_standalone with adjusted version for labeling
        positioned_standalone = positioned_standalone_adjusted
        
        # Clip to prevent overflow
        final_audio = np.clip(final_audio, -1.0, 1.0)
        
        # Adjust target speech ranges to account for random target start time
        target_speech_ranges = [(start + target_start_time, end + target_start_time) 
                                for start, end in target_speech_ranges]
        
        # DEBUG: Print adjusted ranges
        if iteration < 3:
            print(f"  Target start time: {target_start_time:.2f}s")
            print(f"  Speech ranges (after adjustment): {target_speech_ranges[:5]}")
            print(f"  Total duration: {total_duration:.2f}s")
        
        # Generate labels using event-based state machine
        overlap_speech_adjusted = [
            (start + overlap_start_time_adjusted, end + overlap_start_time_adjusted)
            for start, end in overlap_speech_ranges
        ]
        
        # Build events - add standalone utterances to events list
        standalone_events_list = []
        for sa_audio, sa_ranges, sa_start, sa_duration, sa_id in positioned_standalone:
            # sa_start is now the adjusted absolute time position
            for start, end in sa_ranges:
                start_adj = sa_start + start
                end_adj = sa_start + end
                standalone_events_list.append((start_adj, 'NTSS_START'))
                standalone_events_list.append((end_adj, 'NTSS_END'))
        
        events = []
        if no_target_speaker:
            # In no-target mode, treat first speaker as non-target too
            for start, end in target_speech_ranges:
                events.append((start, 'NTSS_START'))
                events.append((end, 'NTSS_END'))
        else:
            for start, end in target_speech_ranges:
                events.append((start, 'TSS_START'))
                events.append((end, 'TSS_END'))
        for start, end in overlap_speech_adjusted:
            events.append((start, 'NTSS_START'))
            events.append((end, 'NTSS_END'))
        
        # Add standalone events
        events.extend(standalone_events_list)
        
        events.sort(key=lambda x: x[0])
        
        # DEBUG: Show first and last events
        if iteration < 3:
            print(f"  Events (first 10):")
            for i, (time, event_type) in enumerate(events[:10]):
                print(f"    {time:.2f}s: {event_type}")
            if len(events) > 10:
                print(f"    ... ({len(events) - 10} more events)")
            print(f"  Events (last 5):")
            for i, (time, event_type) in enumerate(events[-5:]):
                print(f"    {time:.2f}s: {event_type}")
        
        # State machine for labeling
        tss_count = 0
        ntss_count = 0
        segments = []
        prev_time = 0.0
        
        for time, event_type in events:
            # Save segment [prev_time, time] with CURRENT state (before updating)
            if time > prev_time:
                # Determine label based on current state
                if tss_count > 0:
                    segment_label = 'T'
                elif ntss_count > 0:
                    segment_label = 'N'
                else:
                    segment_label = ''
                
                segments.append((prev_time, time, segment_label))
                prev_time = time
            
            # Update state for next segment
            if event_type == 'TSS_START':
                tss_count += 1
            elif event_type == 'TSS_END':
                tss_count -= 1
            elif event_type == 'NTSS_START':
                ntss_count += 1
            elif event_type == 'NTSS_END':
                ntss_count -= 1
        
        # Add final segment to end of audio
        if tss_count > 0:
            final_label = 'T'
        elif ntss_count > 0:
            final_label = 'N'
        else:
            final_label = ''
        
        if total_duration > prev_time:
            segments.append((prev_time, total_duration, final_label))
        
        # DEBUG: Show label distribution
        if iteration < 3:
            print(f"  Label segments (first 10):")
            for i, (start, end, label) in enumerate(segments[:10]):
                label_name = 'TARGET' if label == 'T' else ('NON-TARGET' if label == 'N' else 'SILENCE')
                print(f"    {start:.2f}-{end:.2f}s: {label_name} ({label!r})")
            if len(segments) > 10:
                print(f"    ... ({len(segments) - 10} more segments)")
            print(f"  Label segments (last 10):")
            for i, (start, end, label) in enumerate(segments[-10:]):
                label_name = 'TARGET' if label == 'T' else ('NON-TARGET' if label == 'N' else 'SILENCE')
                print(f"    {start:.2f}-{end:.2f}s: {label_name} ({label!r})")
            
            # Count label types
            t_count = sum(1 for _, _, l in segments if l == 'T')
            n_count = sum(1 for _, _, l in segments if l == 'N')
            s_count = sum(1 for _, _, l in segments if l == '')
            print(f"  Total segments: {len(segments)} (T:{t_count}, N:{n_count}, silence:{s_count})")
        
        # Calculate statistics for metadata
        # Note: segments are non-overlapping and cover the entire duration
        # 'T' = target speaking (includes overlapped regions due to priority)
        # 'N' = only non-target speaking (target not speaking)
        # '' = silence
        
        total_target_duration = 0.0  # All target speech (including overlapped)
        total_non_target_only_duration = 0.0  # Non-target speech when target is NOT speaking
        total_silence_duration = 0.0
        
        # Calculate durations from segments (which are non-overlapping and sum to total_duration)
        for start, end, label in segments:
            duration = end - start
            if label == 'T':
                total_target_duration += duration
            elif label == 'N':
                total_non_target_only_duration += duration
            else:  # label == ''
                total_silence_duration += duration
        
        # Calculate total non-target speech by merging all non-target ranges (to avoid double-counting)
        # Collect all non-target speech ranges
        all_non_target_ranges = []
        
        # Add overlap utterance speech ranges
        for ov_start, ov_end in overlap_speech_adjusted:
            if ov_start < total_duration:
                all_non_target_ranges.append((ov_start, min(ov_end, total_duration)))
        
        # Add standalone utterance speech ranges
        for sa_audio, sa_ranges, sa_start, sa_duration, sa_id in positioned_standalone:
            # sa_start is now the adjusted absolute time position
            for sa_s, sa_e in sa_ranges:
                sa_s_adj = sa_start + sa_s
                sa_e_adj = sa_start + sa_e
                if sa_s_adj < total_duration:
                    all_non_target_ranges.append((sa_s_adj, min(sa_e_adj, total_duration)))
        
        # Merge overlapping non-target ranges to get total non-target speech duration
        if all_non_target_ranges:
            # Sort by start time
            all_non_target_ranges.sort(key=lambda x: x[0])
            
            # Merge overlapping ranges
            merged_nt_ranges = [all_non_target_ranges[0]]
            for current_start, current_end in all_non_target_ranges[1:]:
                last_start, last_end = merged_nt_ranges[-1]
                if current_start <= last_end:
                    # Overlapping or adjacent - merge
                    merged_nt_ranges[-1] = (last_start, max(last_end, current_end))
                else:
                    # Non-overlapping - add new range
                    merged_nt_ranges.append((current_start, current_end))
            
            # Calculate total non-target speech from merged ranges
            total_non_target_speech_raw = sum(end - start for start, end in merged_nt_ranges)
        else:
            merged_nt_ranges = []
            total_non_target_speech_raw = 0.0
        
        # Calculate overlap duration (intersection of target with non-target)
        total_overlap_duration = 0.0
        for t_start, t_end in target_speech_ranges:
            for nt_start, nt_end in merged_nt_ranges:
                overlap_start = max(t_start, nt_start)
                overlap_end = min(t_end, nt_end)
                if overlap_start < overlap_end:
                    total_overlap_duration += (overlap_end - overlap_start)
        
        # Total non-target duration = merged non-target speech
        total_non_target_duration = total_non_target_speech_raw
        
        # Verification: target + non_target_only + silence should equal total_duration
        # Where non_target_only is the portion that doesn't overlap with target
        # So: target_only + non_target_only + overlap + silence = total_duration
        
        # Convert to timestamp/label format for output
        timestamps_list = []
        labels_list = []
        
        for start, end, label in segments:
            timestamps_list.append(f'{start:.2f}')
            labels_list.append(label)
        
        # Add final timestamp (N labels need N+1 timestamps)
        if segments:
            timestamps_list.append(f'{segments[-1][1]:.2f}')
        
        # Filter very close timestamps while maintaining N labels + N+1 timestamps relationship
        filtered_timestamps = []
        filtered_labels = []
        
        # Process each segment
        prev_ts_float = -1.0  # Track the last kept timestamp
        for i in range(len(labels_list)):
            ts = timestamps_list[i]
            lbl = labels_list[i]
            ts_float = float(ts)
            
            # Keep segment if timestamp is far enough from the last kept one
            if prev_ts_float < 0 or ts_float - prev_ts_float >= 0.01:
                filtered_timestamps.append(ts)
                filtered_labels.append(lbl)
                prev_ts_float = ts_float
            # else: skip this segment (both timestamp and label)
        
        # Always add the final timestamp (this creates the N+1 timestamp)
        final_ts = timestamps_list[-1]
        final_ts_float = float(final_ts)
        
        # Only add if it's different from the last timestamp we kept
        if not filtered_timestamps or final_ts_float - float(filtered_timestamps[-1]) >= 0.01:
            filtered_timestamps.append(final_ts)
        elif final_ts_float > float(filtered_timestamps[-1]):
            # Update the last timestamp if the final is slightly later
            filtered_timestamps[-1] = final_ts
        
        # Ensure we have at least total_duration as final timestamp
        if float(filtered_timestamps[-1]) < total_duration:
            filtered_timestamps[-1] = f'{total_duration:.2f}'
        
        # Verify format: N labels should have N+1 timestamps
        if len(filtered_labels) != len(filtered_timestamps) - 1:
            print(f"WARNING: Label/timestamp mismatch for {target_id}_OV_{overlap_id}")
            print(f"  Labels: {len(filtered_labels)}, Timestamps: {len(filtered_timestamps)}")
            print(f"  Original segments: {len(segments)}")
            print(f"  First 5 filtered labels: {filtered_labels[:5]}")
            print(f"  First 6 filtered timestamps: {filtered_timestamps[:6]}")
            print(f"  Last 5 filtered labels: {filtered_labels[-5:]}")
            print(f"  Last 6 filtered timestamps: {filtered_timestamps[-6:]}")
            # Force fix: ensure N+1 timestamps
            if len(filtered_labels) == len(filtered_timestamps):
                # Missing final timestamp - add it
                filtered_timestamps.append(f'{total_duration:.2f}')
            elif len(filtered_labels) > len(filtered_timestamps):
                # Too many labels - shouldn't happen, but truncate
                filtered_labels = filtered_labels[:len(filtered_timestamps)-1]
        
        # Generate filename
        file_name = f"{target_id}_OV_{overlap_id}"
        
        # Save audio
        audio_file = os.path.join(cur_dir, f"{file_name}.flac")
        sf.write(audio_file, final_audio, sr)
        
        # Write to output files
        if FLAC:
            wav_scp.write(f"{file_name} flac -d -c -s {wav_scp_prefix}{scp_path}{file_name}.flac |\n")
        else:
            wav_scp.write(f"{file_name} sox {wav_scp_prefix}{scp_path}{file_name}.flac -b 16 -e signed -c 1 -t wav - |\n")
        
        utt2spk.write(f"{file_name} {current_speaker_id}\n")
        
        labels_str = ','.join(filtered_labels)
        timestamps_str = ' '.join(filtered_timestamps)
        text.write(f"{file_name} {labels_str} {timestamps_str}\n")
        
        # Save metadata with detailed statistics
        meta_file = os.path.join(cur_dir, f"{file_name}.overlap_meta")
        with open(meta_file, 'w') as meta:
            meta.write(f"sample_id: {file_name}\n")
            meta.write(f"target_utterance: {target_id} (speaker: {current_speaker_id})\n")
            meta.write(f"overlap_utterance: {overlap_id} (speaker: {overlap_speaker_id})\n")
            meta.write(f"target_position: {target_start_time:.2f}-{target_end_time:.2f}s\n")
            meta.write(f"overlap_position: {overlap_start_time_adjusted:.2f}-{overlap_end_time_adjusted:.2f}s\n")
            meta.write(f"overlap_percentage_target: {overlap_pct}%\n")
            meta.write(f"amplitude_ratio: {amplitude_ratio}\n")
            
            # Duration statistics
            target_standalone = total_target_duration - total_overlap_duration
            non_target_standalone = total_non_target_duration - total_overlap_duration
            
            meta.write(f"\n# Duration Statistics:\n")
            meta.write(f"total_duration: {total_duration:.3f}s (100.00%)\n")
            meta.write(f"\n## Target Speech Breakdown:\n")
            meta.write(f"target_speech_total: {total_target_duration:.3f}s ({total_target_duration/total_duration*100:.2f}%)\n")
            meta.write(f"  - target_standalone: {target_standalone:.3f}s ({target_standalone/total_duration*100:.2f}%) [target only]\n")
            meta.write(f"  - target_overlap: {total_overlap_duration:.3f}s ({total_overlap_duration/total_duration*100:.2f}%) [with non-target]\n")
            meta.write(f"\n## Non-Target Speech Breakdown:\n")
            meta.write(f"non_target_speech_total: {total_non_target_duration:.3f}s ({total_non_target_duration/total_duration*100:.2f}%)\n")
            meta.write(f"  - non_target_standalone: {non_target_standalone:.3f}s ({non_target_standalone/total_duration*100:.2f}%) [non-target only]\n")
            meta.write(f"  - non_target_overlap: {total_overlap_duration:.3f}s ({total_overlap_duration/total_duration*100:.2f}%) [with target]\n")
            meta.write(f"\n## Other:\n")
            meta.write(f"silence_duration: {total_silence_duration:.3f}s ({total_silence_duration/total_duration*100:.2f}%)\n")
            meta.write(f"overlap_duration: {total_overlap_duration:.3f}s ({total_overlap_duration/total_duration*100:.2f}%) [both speaking]\n")
            meta.write(f"\n# Verification: target_standalone + non_target_standalone + overlap + silence = {target_standalone + non_target_standalone + total_overlap_duration + total_silence_duration:.3f}s ({(target_standalone + non_target_standalone + total_overlap_duration + total_silence_duration)/total_duration*100:.2f}%)\n")
            
            # Add standalone utterance info
            meta.write(f"\n# Standalone non-target utterances ({len(positioned_standalone)}):\n")
            for i, (sa_audio, sa_ranges, sa_start, sa_duration, sa_id) in enumerate(positioned_standalone):
                # sa_start is now the adjusted absolute time position
                meta.write(f"standalone_{i}: {sa_id} "
                         f"at {sa_start:.2f}s "
                         f"duration {sa_duration:.2f}s\n")
            
            meta.write(f"\n# Labeling scheme:\n")
            meta.write(f"T (TSS/class 2): Target speaker SPEECH (not just presence)\n")
            meta.write(f"N (NTSS/class 1): Only non-target speaker SPEECH\n")
            meta.write(f"'' (NS/class 0): Silence/non-speech (including target speaker silence)\n")
        
        # Accumulate dataset statistics
        dataset_total_duration += total_duration
        dataset_target_duration += total_target_duration
        dataset_non_target_duration += total_non_target_duration
        dataset_silence_duration += total_silence_duration
        dataset_overlap_duration += total_overlap_duration
        dataset_num_standalone_utts += len(positioned_standalone)
        
        used_speakers.add(current_speaker_id)
        
        if (iteration + 1) % 50 == 0:
            print(f"Generated {iteration + 1}/{n} samples")
    
    # Close files
    wav_scp.close()
    utt2spk.close()
    text.close()
    
    # Write dataset-level metadata
    dataset_meta_path = os.path.join(dest, 'dataset_metadata.txt')
    with open(dataset_meta_path, 'w') as meta:
        dataset_target_standalone = dataset_target_duration - dataset_overlap_duration
        dataset_non_target_standalone = dataset_non_target_duration - dataset_overlap_duration
        
        meta.write(f"# Dataset-Level Statistics\n")
        meta.write(f"# Generated: {n} utterances\n")
        meta.write(f"# Target speakers used: {len(used_speakers)}\n")
        meta.write(f"# Overlap percentage: {overlap_pct}%\n\n")
        
        meta.write(f"## Total Duration Statistics:\n")
        meta.write(f"total_duration: {dataset_total_duration:.3f}s (100.00%)\n")
        meta.write(f"\n### Target Speech:\n")
        meta.write(f"target_speech_total: {dataset_target_duration:.3f}s ({dataset_target_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - target_standalone: {dataset_target_standalone:.3f}s ({dataset_target_standalone/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - target_overlap: {dataset_overlap_duration:.3f}s ({dataset_overlap_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"\n### Non-Target Speech:\n")
        meta.write(f"non_target_speech_total: {dataset_non_target_duration:.3f}s ({dataset_non_target_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - non_target_standalone: {dataset_non_target_standalone:.3f}s ({dataset_non_target_standalone/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - non_target_overlap: {dataset_overlap_duration:.3f}s ({dataset_overlap_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"\n### Other:\n")
        meta.write(f"silence_duration: {dataset_silence_duration:.3f}s ({dataset_silence_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"overlap_duration: {dataset_overlap_duration:.3f}s ({dataset_overlap_duration/dataset_total_duration*100:.2f}%) [both speaking]\n")
        
        meta.write(f"\n## Per-Utterance Averages:\n")
        meta.write(f"avg_duration_per_utt: {dataset_total_duration/n:.3f}s\n")
        meta.write(f"avg_target_total_per_utt: {dataset_target_duration/n:.3f}s ({dataset_target_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - avg_target_standalone: {dataset_target_standalone/n:.3f}s ({dataset_target_standalone/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - avg_target_overlap: {dataset_overlap_duration/n:.3f}s ({dataset_overlap_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"avg_non_target_total_per_utt: {dataset_non_target_duration/n:.3f}s ({dataset_non_target_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - avg_non_target_standalone: {dataset_non_target_standalone/n:.3f}s ({dataset_non_target_standalone/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"  - avg_non_target_overlap: {dataset_overlap_duration/n:.3f}s ({dataset_overlap_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"avg_silence_per_utt: {dataset_silence_duration/n:.3f}s ({dataset_silence_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"avg_overlap_per_utt: {dataset_overlap_duration/n:.3f}s ({dataset_overlap_duration/dataset_total_duration*100:.2f}%)\n")
        meta.write(f"avg_standalone_utts_per_sample: {dataset_num_standalone_utts/n:.2f}\n")
        
        meta.write(f"\n## Labeling Scheme:\n")
        meta.write(f"T (TSS/class 2): Target speaker SPEECH\n")
        meta.write(f"N (NTSS/class 1): Non-target speaker SPEECH (overlapped or standalone)\n")
        meta.write(f"'' (NS/class 0): Silence/non-speech\n")
        
        meta.write(f"\n## Verification:\n")
        meta.write(f"target_standalone + non_target_standalone + overlap + silence = {dataset_target_standalone + dataset_non_target_standalone + dataset_overlap_duration + dataset_silence_duration:.3f}s ({(dataset_target_standalone + dataset_non_target_standalone + dataset_overlap_duration + dataset_silence_duration)/dataset_total_duration*100:.2f}%)\n")
    
    print(f"\nGeneration complete! Created {n} samples.")
    print(f"Used {len(used_speakers)} target speakers")
    print(f"Dataset metadata saved to: {dataset_meta_path}")
    
    return used_speakers


if __name__ == '__main__':
    parser = ap.ArgumentParser(
        description="Generate simple overlapping utterances for PersonalVAD training"
    )
    parser.add_argument('--libri_root', type=str, required=True,
                       help="Path to LibriSpeech dataset")
    parser.add_argument('--concat_dir', type=str, required=True,
                       help="Output directory")
    parser.add_argument('--count', type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument('--overlap_pct', type=float, default=OVERLAP_PERCENTAGE,
                       help="Overlap percentage (0-100)")
    parser.add_argument('--amplitude_ratio', type=float, default=OVERLAP_AMPLITUDE_RATIO,
                       help="Amplitude ratio for overlap speaker")
    parser.add_argument('--scp_prefix', type=str, default='',
                       help="Prefix for wav.scp paths")
    parser.add_argument('--base-speaker', type=str, default=None,
                       help="Comma-separated speaker IDs to use as target (e.g., '84,174,251')")
    parser.add_argument('--single-speaker-dataset', type=str, action='append', dest='single_speaker_datasets',
                       help="Path to single speaker dataset directory (can be specified multiple times for multiple datasets)")
    parser.add_argument('--single-speaker-count', type=int, default=None,
                       help="Maximum number of utterances to use from EACH single speaker dataset (0=all)")
    parser.add_argument('--no-target-speaker', action='store_true',
                       help="Generate samples with NO target speakers (all speech labeled as non-target)")
    parser.add_argument('parts', type=str, nargs='*',
                       help="LibriSpeech subsets to use (default: train-clean-100)")
    
    args = parser.parse_args()
    
    sets = args.parts if args.parts else ['train-clean-100']
    
    generate_overlapping_utterances(
        libri_root=args.libri_root,
        dest=args.concat_dir,
        n=args.count,
        overlap_pct=args.overlap_pct,
        amplitude_ratio=args.amplitude_ratio,
        wav_scp_prefix=args.scp_prefix,
        sets=sets,
        base_speakers=args.base_speaker,
        no_target_speaker=args.no_target_speaker,
        single_speaker_datasets=args.single_speaker_datasets,
        single_speaker_count=args.single_speaker_count
    )
