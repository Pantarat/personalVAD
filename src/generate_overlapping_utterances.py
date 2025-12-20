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
    
    Args:
        aligned_text: Comma-separated aligned text (empty segments = silence)
        stamps: List of timestamp strings marking segment boundaries
        
    Returns:
        List of (start, end) tuples for speech segments
    """
    segments = aligned_text.split(',')
    speech_ranges = []
    
    for i, segment in enumerate(segments):
        if segment.strip() != '':  # Non-empty = speech
            start_time = 0.0 if i == 0 else float(stamps[i-1])
            end_time = float(stamps[i])
            if end_time > start_time:
                speech_ranges.append((start_time, end_time))
    
    return speech_ranges


def generate_overlapping_utterances(libri_root, dest, n, overlap_pct=50.0, 
                                   amplitude_ratio=1.0, wav_scp_prefix='', 
                                   sets=None, base_speakers=None):
    """
    Generate overlapping utterances with simple logic.
    
    Args:
        libri_root: Path to LibriSpeech dataset
        dest: Output directory
        n: Number of samples to generate
        overlap_pct: Percentage of target speech to overlap (0-100)
        amplitude_ratio: Amplitude multiplier for overlap speaker
        wav_scp_prefix: Prefix for wav.scp paths
        sets: List of LibriSpeech subsets to use
        base_speakers: Optional list of specific speaker IDs to use as target
    """
    if sets is None:
        sets = ['train-clean-100']
    
    # Load dataset
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
    
    # Setup base speakers if specified
    base_speakers_list = None
    base_speakers_data = {}
    
    if base_speakers:
        base_speakers_list = base_speakers.split(',')
        for speaker_id in base_speakers_list:
            for idx, spk in enumerate(dataset):
                if spk[0] == speaker_id:
                    base_speakers_data[speaker_id] = {
                        'utts': list(spk[1]),
                        'idx': idx
                    }
                    dataset[idx] = [spk[0], list(spk[1])]
                    break
        print(f"Using base speakers: {', '.join(base_speakers_list)}")
    
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
    
    # Generate samples
    for iteration in range(n):
        # Create subdirectory every FILES_PER_DIR samples
        if iteration % FILES_PER_DIR == 0:
            scp_path = f"{iteration // FILES_PER_DIR}_overlap/"
            cur_dir = os.path.join(dest, scp_path)
            os.makedirs(cur_dir, exist_ok=True)
        
        # Select target speaker
        if base_speakers_list:
            current_speaker_id = base_speakers_list[iteration % len(base_speakers_list)]
            speaker_idx = base_speakers_data[current_speaker_id]['idx']
            target_speaker = dataset[speaker_idx]
            
            # Reload utterances if needed
            if not target_speaker[1]:
                target_speaker[1] = list(base_speakers_data[current_speaker_id]['utts'])
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
        
        # Position overlap to achieve desired overlap percentage
        # Place overlap randomly within target speech region
        if target_speech_ranges:
            target_speech_start = target_speech_ranges[0][0]
            target_speech_end = target_speech_ranges[-1][1]
            target_speech_span = target_speech_end - target_speech_start
            
            overlap_duration = overlap_audio.size / sr
            
            # Random position allowing partial overlap
            max_start = max(0, target_speech_span - overlap_duration * 0.3)
            overlap_start_time = target_speech_start + random.uniform(0, max(0.1, max_start))
        else:
            overlap_start_time = 0.0
        
        overlap_end_time = overlap_start_time + (overlap_audio.size / sr)
        
        # Create final audio
        target_duration = target_audio.size / sr
        total_duration = max(target_duration, overlap_end_time)
        total_samples = int(total_duration * sr)
        
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        # Add target audio (starts at 0)
        final_audio[:target_audio.size] += target_audio
        
        # Add overlap audio
        overlap_start_sample = int(overlap_start_time * sr)
        overlap_end_sample = min(overlap_start_sample + overlap_audio.size, total_samples)
        overlap_samples_to_add = overlap_end_sample - overlap_start_sample
        
        final_audio[overlap_start_sample:overlap_end_sample] += (
            overlap_audio[:overlap_samples_to_add] * amplitude_ratio
        )
        
        # Clip to prevent overflow
        final_audio = np.clip(final_audio, -1.0, 1.0)
        
        # Generate labels using event-based state machine
        overlap_speech_adjusted = [
            (start + overlap_start_time, end + overlap_start_time)
            for start, end in overlap_speech_ranges
        ]
        
        # Build events
        events = []
        for start, end in target_speech_ranges:
            events.append((start, 'TSS_START'))
            events.append((end, 'TSS_END'))
        for start, end in overlap_speech_adjusted:
            events.append((start, 'NTSS_START'))
            events.append((end, 'NTSS_END'))
        
        events.sort(key=lambda x: x[0])
        
        # State machine for labeling
        tss_count = 0
        ntss_count = 0
        segments = []
        prev_time = 0.0
        
        # Determine initial label at time 0
        current_label = ''
        
        for time, event_type in events:
            # Determine label for segment [prev_time, time] based on CURRENT state
            if tss_count > 0:
                segment_label = 'T'
            elif ntss_count > 0:
                segment_label = 'N'
            else:
                segment_label = ''
            
            # Save segment if it has duration
            if time > prev_time:
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
        
        # Convert to timestamp/label format
        timestamps_list = []
        labels_list = []
        
        for start, end, label in segments:
            timestamps_list.append(f'{start:.2f}')
            labels_list.append(label)
        
        if segments:
            timestamps_list.append(f'{segments[-1][1]:.2f}')
        
        # Filter very close timestamps
        filtered_timestamps = []
        filtered_labels = []
        for i, (ts, lbl) in enumerate(zip(timestamps_list, labels_list)):
            ts_float = float(ts)
            if i == 0 or ts_float - float(filtered_timestamps[-1]) >= 0.01:
                filtered_timestamps.append(ts)
                filtered_labels.append(lbl)
        
        if not filtered_timestamps or float(filtered_timestamps[-1]) < total_duration:
            filtered_timestamps.append(f'{total_duration:.2f}')
        
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
        
        # Save metadata
        meta_file = os.path.join(cur_dir, f"{file_name}.overlap_meta")
        with open(meta_file, 'w') as meta:
            meta.write(f"sample_id: {file_name}\n")
            meta.write(f"target_utterance: {target_id} (speaker: {current_speaker_id})\n")
            meta.write(f"overlap_utterance: {overlap_id} (speaker: {overlap_speaker_id})\n")
            meta.write(f"overlap_position: {overlap_start_time:.2f}-{overlap_end_time:.2f}\n")
            meta.write(f"overlap_percentage_target: {overlap_pct}%\n")
            meta.write(f"amplitude_ratio: {amplitude_ratio}\n")
            meta.write(f"\n# Labeling scheme:\n")
            meta.write(f"T (TSS/class 2): Target speaker present\n")
            meta.write(f"N (NTSS/class 1): Only non-target speaker(s)\n")
            meta.write(f"'' (NS/class 0): Silence/non-speech\n")
        
        used_speakers.add(current_speaker_id)
        
        if (iteration + 1) % 50 == 0:
            print(f"Generated {iteration + 1}/{n} samples")
    
    # Close files
    wav_scp.close()
    utt2spk.close()
    text.close()
    
    print(f"\nGeneration complete! Created {n} samples.")
    print(f"Used {len(used_speakers)} target speakers")
    
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
        base_speakers=args.base_speaker
    )
