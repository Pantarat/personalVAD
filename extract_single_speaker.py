#!/usr/bin/env python3
"""
Extract all utterances from a single speaker in LibriSpeech dataset,
concatenate them, split into 5-second chunks with speech/non-speech labels,
and save in Kaldi format.

Usage:
    python extract_single_speaker.py
    
Or override defaults with command-line arguments:
    python extract_single_speaker.py --speaker_id 1221 --libri_root data/LibriSpeech \
        --output_dir data/speaker_1221 --test_ratio 0.2 --chunk_duration 5.0
"""

import argparse
import os
import random
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import soundfile as sf


# ================ CONFIGURATION - EDIT HERE =========================

# Speaker and data paths
SPEAKER_ID = '5105'  # Speaker ID to extract (e.g., '1089', '121', '237', '84')
LIBRI_ROOT = 'data/LibriSpeech'  # Path to LibriSpeech root directory
OUTPUT_DIR = 'data/speaker_5105'  # Output directory for train/test splits

# Chunking parameters
CHUNK_DURATION = 5.0  # Duration of each chunk in seconds
SAMPLE_RATE = 16000  # Audio sample rate in Hz

# Train/test split parameters
TEST_RATIO = 0.2  # Ratio of test data (0.2 = 20% test, 80% train)
NO_SPLIT = False  # If True, save all data without splitting into train/test
SPLIT_TRAIN_SUBSETS = True  # If True, split train set into subsets with specified utterances
UTTERANCES_PER_SUBSET = 3  # Number of utterances per subset (None = use test set size)
RANDOM_SEED = 42  # Random seed for reproducible train/test splits

# ====================================================================


def parse_alignment_file(alignment_path: str) -> Dict[str, Tuple[str, List[float]]]:
    """
    Parse LibriSpeech alignment file to get speech timestamps.
    
    Format: utt_id ",WORD1,,WORD2,WORD3," "t1,t2,t3,t4"
    - Commas separate segments
    - Empty string (between commas) = silence
    - Word = speech
    - Timestamps define segment boundaries (prepend 0.0 for start)
    
    Returns:
        Dict mapping utt_id to (aligned_text, timestamps)
        aligned_text: string where each char is 'W' (speech) or ' ' (silence)
        timestamps: list of frame boundaries in seconds (includes 0.0 at start)
    """
    alignments = {}
    with open(alignment_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 3:
                continue
            
            utt_id = parts[0]
            aligned_str = parts[1][1:-1]  # Remove quotes: ",GO,,DO," -> ,GO,,DO,
            tstamps_str = parts[2][1:-2]  # Remove quotes: "0.49,0.89" -> 0.49,0.89
            
            # Split by commas to get individual segments
            segments = aligned_str.split(',')
            
            # Convert segments to W (word/speech) or space (silence/empty)
            aligned_text = ''.join(['W' if seg.strip() else ' ' for seg in segments])
            
            # Parse timestamps and prepend 0.0 for the start
            timestamps = [0.0] + [float(t) for t in tstamps_str.split(',')]
            
            alignments[utt_id] = (aligned_text, timestamps)
    
    return alignments


def find_speaker_utterances(libri_root: str, speaker_id: str) -> List[Dict]:
    """
    Find all utterances for a given speaker with alignment information.
    
    Returns:
        List of dicts with keys: utt_id, wav_path, text, speaker_id, aligned_text, timestamps
    """
    libri_path = Path(libri_root)
    utterances = []
    
    # Search through all subsets in LibriSpeech
    for subset_dir in libri_path.iterdir():
        if not subset_dir.is_dir():
            continue
            
        # Look for the speaker directory
        speaker_path = subset_dir / speaker_id
        if not speaker_path.exists():
            continue
            
        print(f"Found speaker {speaker_id} in {subset_dir.name}")
        
        # Iterate through all chapters for this speaker
        for chapter_dir in speaker_path.iterdir():
            if not chapter_dir.is_dir():
                continue
                
            chapter_id = chapter_dir.name
            trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            align_file = chapter_dir / f"{speaker_id}-{chapter_id}.alignment.txt"
            
            if not trans_file.exists():
                continue
            
            # Parse alignments if available
            alignments = {}
            if align_file.exists():
                alignments = parse_alignment_file(str(align_file))
            
            # Read transcriptions
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) < 2:
                        continue
                    
                    utt_id = parts[0]
                    text = parts[1]
                    wav_path = chapter_dir / f"{utt_id}.flac"
                    
                    if wav_path.exists():
                        utt_dict = {
                            'utt_id': utt_id,
                            'wav_path': str(wav_path.absolute()),
                            'text': text,
                            'speaker_id': speaker_id,
                            'aligned_text': None,
                            'timestamps': None
                        }
                        
                        if utt_id in alignments:
                            utt_dict['aligned_text'] = alignments[utt_id][0]
                            utt_dict['timestamps'] = alignments[utt_id][1]
                        
                        utterances.append(utt_dict)
    
    return utterances


def concatenate_audio(utterances: List[Dict], sample_rate: int = 16000) -> Tuple[np.ndarray, List[Tuple[float, float, str]]]:
    """
    Concatenate all audio files and track speech/non-speech regions.
    
    Returns:
        audio: concatenated audio array
        speech_regions: list of (start_time, end_time, type) where type is 'speech' or 'silence'
    """
    all_audio = []
    speech_regions = []
    current_time = 0.0
    
    # Sort utterances by utt_id for consistency
    utterances = sorted(utterances, key=lambda x: x['utt_id'])
    
    for utt in utterances:
        # Load audio
        audio, sr = sf.read(utt['wav_path'])
        if sr != sample_rate:
            raise ValueError(f"Expected sample rate {sample_rate}, got {sr} for {utt['utt_id']}")
        
        duration = len(audio) / sr
        
        # If we have alignment info, use it
        if utt['timestamps'] is not None and utt['aligned_text'] is not None:
            timestamps = utt['timestamps']
            aligned_text = utt['aligned_text']
            
            # Parse aligned text to get speech/silence regions
            # Each character in aligned_text corresponds to a segment BETWEEN consecutive timestamps
            # i.e., aligned_text[i] describes the region from timestamps[i] to timestamps[i+1]
            for i in range(len(aligned_text)):
                if i + 1 < len(timestamps):
                    start = current_time + timestamps[i]
                    end = current_time + timestamps[i + 1]
                    
                    char = aligned_text[i]
                    region_type = 'speech' if char == 'W' else 'silence'
                    speech_regions.append((start, end, region_type))
        else:
            # No alignment info, assume entire utterance is speech
            speech_regions.append((current_time, current_time + duration, 'speech'))
        
        all_audio.append(audio)
        current_time += duration
    
    # Concatenate all audio
    concatenated = np.concatenate(all_audio) if all_audio else np.array([])
    
    return concatenated, speech_regions


def split_into_chunks(audio: np.ndarray, speech_regions: List[Tuple[float, float, str]], 
                     chunk_duration: float, sample_rate: int = 16000) -> List[Dict]:
    """
    Split audio into fixed-duration chunks and generate frame-level labels.
    Discard chunks with no speech and discard the last chunk if it's not the full duration.
    
    Returns:
        List of dicts with keys: chunk_id, audio, labels
        labels is a numpy array where 1=speech, 0=non-speech at 10ms frame rate
    """
    frame_shift = 0.01  # 10ms frames
    chunk_samples = int(chunk_duration * sample_rate)
    total_duration = len(audio) / sample_rate
    
    chunks = []
    chunk_idx = 0
    
    for start_time in np.arange(0, total_duration, chunk_duration):
        end_time = min(start_time + chunk_duration, total_duration)
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract audio chunk
        chunk_audio = audio[start_sample:end_sample]
        
        # Check if this is the last chunk and it's shorter than the target duration
        chunk_actual_duration = len(chunk_audio) / sample_rate
        if chunk_actual_duration < chunk_duration * 0.99:  # Allow 1% tolerance
            # This is the last incomplete chunk, discard it
            break
        
        # Generate frame-level labels (10ms frames)
        n_frames = int(chunk_actual_duration / frame_shift)
        labels = np.zeros(n_frames, dtype=np.int32)
        
        # Mark frames as speech or non-speech based on speech_regions
        for frame_idx in range(n_frames):
            frame_start = start_time + frame_idx * frame_shift
            frame_end = frame_start + frame_shift
            frame_center = (frame_start + frame_end) / 2
            
            # Check if frame center falls in a speech region
            for region_start, region_end, region_type in speech_regions:
                if region_start <= frame_center < region_end and region_type == 'speech':
                    labels[frame_idx] = 1
                    break
        
        # Check if chunk contains any speech
        if np.sum(labels) > 0:
            chunks.append({
                'chunk_id': f'chunk_{chunk_idx:05d}',
                'audio': chunk_audio,
                'labels': labels,
                'start_time': start_time,
                'end_time': end_time
            })
            chunk_idx += 1
    
    return chunks


def split_train_test(chunks: List[Dict], test_ratio: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split chunks into train and test sets.
    """
    random.seed(seed)
    shuffled = chunks.copy()
    random.shuffle(shuffled)
    
    n_test = int(len(shuffled) * test_ratio)
    test_set = shuffled[:n_test]
    train_set = shuffled[n_test:]
    
    return train_set, test_set


def split_train_into_subsets(train_chunks: List[Dict], subset_size: int, seed: int = 42) -> List[List[Dict]]:
    """
    Split train set into multiple subsets of the specified size.
    
    Args:
        train_chunks: List of training chunks
        subset_size: Size of each subset (typically same as test set size)
        seed: Random seed for shuffling
        
    Returns:
        List of subsets, each containing subset_size chunks
    """
    random.seed(seed)
    shuffled = train_chunks.copy()
    random.shuffle(shuffled)
    
    subsets = []
    for i in range(0, len(shuffled), subset_size):
        subset = shuffled[i:i + subset_size]
        if len(subset) == subset_size:  # Only include complete subsets
            subsets.append(subset)
    
    return subsets


def write_kaldi_files(chunks: List[Dict], output_dir: str, speaker_id: str):
    """
    Write Kaldi format files and audio chunks.
    Files: wav.scp, utt2spk, spk2utt, text (alignment info), labels.ark
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create audio directory
    audio_dir = output_path / 'audio'
    audio_dir.mkdir(exist_ok=True)
    
    # Create labels directory  
    labels_dir = output_path / 'labels'
    labels_dir.mkdir(exist_ok=True)
    
    # Sort chunks by chunk_id for consistency
    chunks = sorted(chunks, key=lambda x: x['chunk_id'])
    
    # Write wav.scp
    with open(output_path / 'wav.scp', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            utt_id = f"{speaker_id}_{chunk['chunk_id']}"
            audio_path = audio_dir / f"{utt_id}.flac"
            f.write(f"{utt_id} {audio_path.absolute()}\n")
    
    # Write utt2spk
    with open(output_path / 'utt2spk', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            utt_id = f"{speaker_id}_{chunk['chunk_id']}"
            f.write(f"{utt_id} {speaker_id}\n")
    
    # Write spk2utt
    with open(output_path / 'spk2utt', 'w', encoding='utf-8') as f:
        utt_ids = [f"{speaker_id}_{chunk['chunk_id']}" for chunk in chunks]
        f.write(f"{speaker_id} {' '.join(utt_ids)}\n")
    
    # Write text (with alignment information)
    with open(output_path / 'text', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            utt_id = f"{speaker_id}_{chunk['chunk_id']}"
            # Create simple alignment text: W for speech frames, space for non-speech
            labels = chunk['labels']
            alignment = ''.join(['W' if l == 1 else ' ' for l in labels])
            # Timestamps at 10ms intervals
            timestamps = ','.join([f"{i*0.01:.2f}" for i in range(len(labels) + 1)])
            f.write(f"{utt_id} [{alignment}] [{timestamps}]\n")
    
    # Write labels.scp (paths to label files)
    with open(output_path / 'labels.scp', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            utt_id = f"{speaker_id}_{chunk['chunk_id']}"
            label_path = labels_dir / f"{utt_id}.npy"
            f.write(f"{utt_id} {label_path.absolute()}\n")
    
    # Save audio files and labels
    for chunk in chunks:
        utt_id = f"{speaker_id}_{chunk['chunk_id']}"
        
        # Save audio as FLAC
        audio_path = audio_dir / f"{utt_id}.flac"
        sf.write(audio_path, chunk['audio'], 16000)
        
        # Save labels as numpy array
        label_path = labels_dir / f"{utt_id}.npy"
        np.save(label_path, chunk['labels'])
    
    # Generate statistics
    total_frames = sum(len(chunk['labels']) for chunk in chunks)
    speech_frames = sum(np.sum(chunk['labels']) for chunk in chunks)
    
    stats_path = output_path / 'stats.txt'
    with open(stats_path, 'w') as f:
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Total duration: {total_frames * 0.01:.2f}s\n")
        f.write(f"Speech frames: {speech_frames} ({speech_frames/total_frames*100:.1f}%)\n")
        f.write(f"Non-speech frames: {total_frames - speech_frames} ({(total_frames - speech_frames)/total_frames*100:.1f}%)\n")
    
    print(f"Wrote {len(chunks)} chunks to {output_dir}")
    print(f"  Total duration: {total_frames * 0.01:.2f}s")
    print(f"  Speech: {speech_frames/total_frames*100:.1f}%, Non-speech: {(total_frames - speech_frames)/total_frames*100:.1f}%")



def main():
    parser = argparse.ArgumentParser(
        description='Extract single speaker, concatenate audio, split into 5s chunks with speech labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--speaker_id', type=str, default=SPEAKER_ID,
                        help='Speaker ID to extract')
    parser.add_argument('--libri_root', type=str, default=LIBRI_ROOT,
                        help='Path to LibriSpeech root directory')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for train/test splits')
    parser.add_argument('--chunk_duration', type=float, default=CHUNK_DURATION,
                        help='Duration of each chunk in seconds')
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO,
                        help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for splitting')
    parser.add_argument('--no_split', action='store_true', default=NO_SPLIT,
                        help='Do not split, save all data to output_dir')
    parser.add_argument('--split_train_subsets', action='store_true', default=SPLIT_TRAIN_SUBSETS,
                        help='Split train set into subsets with specified utterances')
    parser.add_argument('--utterances_per_subset', type=int, default=UTTERANCES_PER_SUBSET,
                        help='Number of utterances (chunks) per subset (default: use test set size)')
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE,
                        help='Audio sample rate')
    
    args = parser.parse_args()
    
    # Find all utterances for the speaker
    print(f"Searching for speaker {args.speaker_id} in {args.libri_root}...")
    utterances = find_speaker_utterances(args.libri_root, args.speaker_id)
    
    if not utterances:
        print(f"ERROR: No utterances found for speaker {args.speaker_id}")
        print(f"Make sure the LibriSpeech data and alignment files are extracted in {args.libri_root}")
        return 1
    
    print(f"Found {len(utterances)} utterances for speaker {args.speaker_id}")
    
    # Check if we have alignment information
    has_alignments = any(utt['timestamps'] is not None for utt in utterances)
    if not has_alignments:
        print("WARNING: No alignment files found. Speech/non-speech labels will be approximate.")
        print("Download alignment files from: https://zenodo.org/record/2619474")
    
    # Concatenate audio
    print(f"Concatenating audio files...")
    audio, speech_regions = concatenate_audio(utterances, args.sample_rate)
    total_duration = len(audio) / args.sample_rate
    print(f"Total audio duration: {total_duration:.2f}s")
    
    # Split into chunks
    print(f"Splitting into {args.chunk_duration}s chunks...")
    chunks = split_into_chunks(audio, speech_regions, args.chunk_duration, args.sample_rate)
    
    if not chunks:
        print("ERROR: No valid chunks generated (all chunks had no speech)")
        return 1
    
    print(f"Generated {len(chunks)} chunks with speech content")
    
    if args.no_split:
        # Save all data without splitting
        write_kaldi_files(chunks, args.output_dir, args.speaker_id)
    else:
        # Split into train and test
        train_chunks, test_chunks = split_train_test(chunks, args.test_ratio, args.seed)
        
        print(f"Split: {len(train_chunks)} train chunks, {len(test_chunks)} test chunks")
        
        # Write test set
        test_dir = os.path.join(args.output_dir, 'test')
        write_kaldi_files(test_chunks, test_dir, args.speaker_id)
        
        # Split train set into subsets if requested
        if args.split_train_subsets:
            # Determine subset size: use utterances_per_subset if specified, otherwise use test set size
            subset_size = args.utterances_per_subset if args.utterances_per_subset is not None else len(test_chunks)
            train_subsets = split_train_into_subsets(train_chunks, subset_size, args.seed)
            
            print(f"Splitting train set into {len(train_subsets)} subsets of {subset_size} chunks each")
            
            # Write each train subset
            for i, subset in enumerate(train_subsets, start=1):
                train_dir = os.path.join(args.output_dir, f'train{i}')
                write_kaldi_files(subset, train_dir, args.speaker_id)
            
            # Also write the complete train set
            train_all_dir = os.path.join(args.output_dir, 'train_all')
            write_kaldi_files(train_chunks, train_all_dir, args.speaker_id)
            print(f"Complete train set also saved to train_all/")
        else:
            # Write single train set
            train_dir = os.path.join(args.output_dir, 'train')
            write_kaldi_files(train_chunks, train_dir, args.speaker_id)
    
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
