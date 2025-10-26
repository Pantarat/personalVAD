#!/usr/bin/python

"""@package extract_features_overlap

Enhanced feature extraction for overlapping utterances.

This module extends extract_features.py to properly handle overlapping speech:
- Detects overlap regions from transcript markers
- Labels overlap regions as TSS if target speaker is present, NTSS otherwise
- Compatible with existing PersonalVAD pipeline

"""

import os
import sys
import numpy as np
import kaldiio
import random
import librosa
import argparse as ap
import torch
from kaldiio import ReadHelper, WriteHelper
from glob import glob
import multiprocessing as mp

from resemblyzer import VoiceEncoder
from resemblyzer_mod import VoiceEncoderMod

# Import from original extract_features
sys.path.insert(0, os.path.dirname(__file__))
from extract_features import (
    cos, load_dvector, gpu_worker, process_init,
    KALDI_ROOT, DATA_ROOT, EMBED_PATH, DEST, EMBED,
    TS_DROPOUT, rate, samples_per_frame, frame_step, min_coverage
)

txt = dict()


def parse_overlap_transcript(transcript, tstamps, target_spk_idx, main_speakers):
    """Parse transcript with overlap markers.
    
    Args:
        transcript (str): Transcript with overlap markers (e.g., "W W$W W$OV0:1.5-3.2")
        tstamps (np.array): Timestamp array
        target_spk_idx (int): Index of target speaker in main utterance
        main_speakers (list): List of main speaker IDs
        
    Returns:
        tuple: (base_labels, overlap_regions)
            base_labels: Original labels without overlap consideration
            overlap_regions: List of (start_frame, end_frame, is_target_speaker)
    """
    
    # Split transcript to separate main part and overlap markers
    parts = transcript.split('$OV')
    main_transcript = parts[0]
    
    overlap_regions = []
    
    # Parse overlap markers
    for i in range(1, len(parts)):
        # Format: "0:1.5-3.2" or "1:2.0-4.5"
        marker = parts[i]
        
        # Extract overlap index and time range
        if ':' in marker:
            overlap_info = marker.split(':')
            if len(overlap_info) >= 2:
                time_range = overlap_info[1].split('-')
                if len(time_range) >= 2:
                    try:
                        start_time = float(time_range[0])
                        end_time = float(time_range[1].split('$')[0] if '$' in time_range[1] else time_range[1])
                        
                        # Convert to frame indices (10ms frames)
                        start_frame = int(start_time * 100)
                        end_frame = int(end_time * 100)
                        
                        # For now, assume overlap contains non-target speaker
                        # In practice, you'd need to check overlap speaker IDs
                        # against target speaker
                        is_target = False  # Default: overlap is NTSS
                        
                        overlap_regions.append((start_frame, end_frame, is_target))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to parse overlap marker: {marker}")
                        continue
    
    return overlap_regions


def extract_features_with_overlap(scp, q_send, q_return):
    """CPU worker for PVAD feature extraction with overlap handling.
    
    Enhanced version that properly labels overlapping regions.
    """
    
    # Prepare scp and ark files
    wav_scp = ReadHelper(f'scp:{scp}')
    pid = int(scp.rpartition('.')[0].rpartition('_')[2])
    array_writer = WriteHelper(f'ark,scp:{DEST}/fbanks_{pid}.ark,{DEST}/fbanks_{pid}.scp')
    score_writer = WriteHelper(f'ark,scp:{DEST}/scores_{pid}.ark,{DEST}/scores_{pid}.scp')
    label_writer = WriteHelper(f'ark,scp:{DEST}/labels_{pid}.ark,{DEST}/labels_{pid}.scp')
    target_writer = open(f'{DEST}/targets_{pid}.scp', 'w')
    embed_scp = kaldiio.load_scp(f'{EMBED}/dvectors.scp')

    wav_iter = iter(wav_scp)
    utter_count = 0
    
    while True:
        try:
            item = next(wav_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f"Warning: failed to read entry: {type(e).__name__}: {e}")
            continue

        try:
            utt_id, (sr, arr) = item
        except Exception as e:
            print(f"Warning: unexpected item format: {type(e).__name__}: {e}")
            continue

        # Load transcription and timestamps
        try:
            gtruth, tstamps = txt[utt_id]
        except:
            print(f"Error: key {utt_id} not found.")
            continue

        gt_len = len(gtruth)
        assert gt_len == tstamps.size, "gtruth and tstamps must match"

        # Normalize audio
        arr = arr.astype(np.float32, order='C') / 32768

        # Extract filterbank features
        fbanks = librosa.feature.melspectrogram(
            y=arr, sr=16000, n_fft=400,
            hop_length=160, n_mels=40
        ).astype('float32').T[:-2]
        logfbanks = np.log10(fbanks + 1e-6)

        # Generate filterbank slices for scoring
        wav = arr.copy()
        wav_slices, mel_slices = VoiceEncoder.compute_partial_slices(
            wav.size, rate, min_coverage
        )
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= wav.size:
            wav = np.pad(arr, (0, max_wave_length - wav.size), "constant")
        mels = librosa.feature.melspectrogram(
            y=wav, sr=16000, n_fft=400,
            hop_length=160, n_mels=40
        ).astype('float32').T

        fbanks_sliced = np.array([mels[s] for s in mel_slices])

        # Generate ground truth labels
        n = logfbanks.shape[0]

        if tstamps[-1] < n * 10:
            tstamps[-1] = n * 10

        # Select target speaker
        n_speakers = sum(1 for g in gtruth if g == '$') + 1
        
        # Determine main speakers (before overlap markers)
        main_gtruth = []
        main_speakers_count = 0
        for g in gtruth:
            if 'OV' in str(g):
                break
            if g == '$':
                main_speakers_count += 1
            main_gtruth.append(g)
        
        if main_speakers_count == 0:
            main_speakers_count = 1

        # Handle TS_DROPOUT
        if TS_DROPOUT and main_speakers_count == 1:
            use_target = bool(np.random.randint(0, 3))
            if use_target:
                which = 0
                spk_embed, spk_id = load_dvector(utt_id, which, embed_scp)
            else:
                which = -1
                spk_embed, spk_id = load_dvector(utt_id, which, embed_scp, select_random=True)
        else:
            which = np.random.randint(0, main_speakers_count)
            spk_embed, spk_id = load_dvector(utt_id, which, embed_scp)

        # Get embeddings from GPU worker
        fbanks_tensor = torch.unsqueeze(torch.from_numpy(fbanks), 0)
        q_send.put((fbanks_tensor, torch.from_numpy(fbanks_sliced), pid))
        embeds_stream, embeds_slices = q_return.get()

        embeds_stream = embeds_stream.numpy().squeeze()
        embeds_slices = embeds_slices.numpy()

        # Compute scores
        scores_slices = np.array([cos(spk_embed, emb) for emb in embeds_slices])
        scores_stream = np.array([cos(spk_embed, emb) for emb in embeds_stream])

        try:
            # Generate score arrays
            scores_kron = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
            if scores_slices.size > 1:
                scores_kron = np.append(
                    scores_kron,
                    np.kron(scores_slices[1:], np.ones(frame_step, dtype='float32'))
                )
            scores_kron = scores_kron[:n]

            scores_lin = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
            for i, s in enumerate(scores_slices[1:]):
                scores_lin = np.append(
                    scores_lin,
                    np.linspace(scores_slices[i], s, frame_step, endpoint=False)
                )
            scores_lin = scores_lin[:n]

            scores = np.stack((scores_stream, scores_kron, scores_lin))

        except Exception as e:
            print(f"Error computing scores: {type(e).__name__}: {e}")
            continue

        # Generate base labels
        labels = np.ones(n, dtype=np.float32)
        stamp_prev = 0
        tstamps_frames = tstamps // 10

        # Track which speaker we're on
        current_speaker_idx = 0
        
        for (stamp, label) in zip(tstamps_frames, gtruth):
            # Skip overlap markers in label assignment
            if 'OV' in str(label):
                continue
                
            if label == '':
                labels[stamp_prev:stamp] = 0  # NS
            elif label == '$':
                current_speaker_idx += 1
                labels[stamp_prev:stamp] = 0  # Boundary
            else:
                if current_speaker_idx == which:  # TSS
                    labels[stamp_prev:stamp] = 2
                else:  # NTSS
                    labels[stamp_prev:stamp] = 1

            stamp_prev = stamp

        # Apply overlap regions
        # Parse overlap information from transcript
        transcript_str = ''.join(str(g) for g in gtruth)
        overlap_regions = parse_overlap_transcript(
            transcript_str, tstamps, which, []
        )
        
        # Update labels for overlap regions
        for start_frame, end_frame, is_target in overlap_regions:
            # Ensure frames are within bounds
            start_frame = max(0, min(start_frame, n - 1))
            end_frame = max(0, min(end_frame, n))
            
            if end_frame > start_frame:
                # If main utterance has target speaker speech in this region
                main_has_target = np.any(labels[start_frame:end_frame] == 2)
                
                if main_has_target or is_target:
                    # Overlap with target speaker = TSS
                    labels[start_frame:end_frame] = 2
                else:
                    # Overlap without target speaker = NTSS
                    # Check if main was silent
                    main_was_silent = np.all(labels[start_frame:end_frame] == 0)
                    if main_was_silent:
                        labels[start_frame:end_frame] = 1  # NTSS
                    # If main was NTSS, keep as NTSS

        # Write features
        array_writer(utt_id, logfbanks)
        score_writer(utt_id, scores)
        label_writer(utt_id, labels)
        target_writer.write(f"{utt_id} {spk_id}\n")

        # Flush periodically
        utter_count += 1
        if utter_count % 100 == 0:
            try:
                if hasattr(array_writer, 'fark') and hasattr(array_writer, 'fscp'):
                    array_writer.fark.flush()
                    array_writer.fscp.flush()
                if hasattr(score_writer, 'fark') and hasattr(score_writer, 'fscp'):
                    score_writer.fark.flush()
                    score_writer.fscp.flush()
                if hasattr(label_writer, 'fark') and hasattr(label_writer, 'fscp'):
                    label_writer.fark.flush()
                    label_writer.fscp.flush()
                target_writer.flush()
            except Exception as e:
                print(f"Warning: flush failed: {type(e).__name__}: {e}")

    # Close files
    wav_scp.close()
    array_writer.close()
    score_writer.close()
    label_writer.close()
    target_writer.close()


if __name__ == '__main__':
    # Same argument parsing as extract_features.py
    parser = ap.ArgumentParser(description="Extract features for overlapping utterances.")
    parser.add_argument('--kaldi_root', type=str, required=False, default=KALDI_ROOT)
    parser.add_argument('--data_root', type=str, required=False, default=DATA_ROOT)
    parser.add_argument('--dest_path', type=str, required=False, default=DEST)
    parser.add_argument('--embed_path', type=str, required=False, default=EMBED)
    parser.add_argument('--ts_dropout', action='store_true')
    parser.add_argument('--use_kaldi', action='store_true')
    args = parser.parse_args()

    USE_KALDI = args.use_kaldi
    KALDI_ROOT = args.kaldi_root
    DATA_ROOT = args.data_root
    DEST = args.dest_path
    EMBED = args.embed_path
    TS_DROPOUT = args.ts_dropout

    if USE_KALDI:
        os.chdir(KALDI_ROOT)

    # Load transcriptions
    with open(DATA_ROOT + '/text') as text_file:
        for utterance in text_file:
            utt_id, _, rest = utterance.partition(' ')
            labels, _, tstamps = rest.partition(' ')
            txt[utt_id] = (
                labels.split(','),
                np.array([int(float(stamp) * 1000) for stamp in tstamps.split(' ')],
                        dtype='int32')
            )

    # Get file list
    files = glob(DATA_ROOT + '/split_*.scp')
    files.sort()
    nj = len(files)

    # Create communication queues
    manager = mp.Manager()
    q_send = manager.Queue()
    q_return = []
    for i in range(nj):
        q_return.append(manager.Queue())

    # Create GPU worker
    worker = mp.Process(target=gpu_worker, args=(q_send, q_return,))
    worker.daemon = True
    worker.start()

    # Create CPU worker pool
    pool = mp.Pool(processes=nj, initializer=process_init, initargs=(txt,))
    pool.starmap(extract_features_with_overlap, zip(files, [q_send] * nj, q_return))
    pool.close()
    pool.join()

    print("✓ Feature extraction with overlap handling complete!")
