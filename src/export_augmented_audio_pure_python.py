#!/usr/bin/env python3
"""
Export augmented audio files from Kaldi wav.scp WITHOUT requiring Kaldi binaries.

This script reads Kaldi wav.scp files and recreates the augmented audio
using pure Python libraries (numpy, soundfile, scipy) instead of Kaldi tools.
"""

import subprocess
import os
import sys
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("❌ Error: soundfile library not found")
    print("   Install with: pip install soundfile")
    sys.exit(1)

try:
    from scipy.signal import fftconvolve
except ImportError:
    print("❌ Error: scipy library not found")
    print("   Install with: pip install scipy")
    sys.exit(1)


def parse_wav_scp(scp_file):
    """Parse wav.scp file to extract utterance IDs and commands."""
    utterances = []
    with open(scp_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, command = parts
                utterances.append((utt_id, command))
    return utterances


def load_flac_audio(flac_path):
    """
    Load FLAC audio file.
    
    Returns:
        audio: numpy array of audio samples
        sr: sample rate
    """
    if not Path(flac_path).exists():
        raise FileNotFoundError(f"FLAC file not found: {flac_path}")
    
    audio, sr = sf.read(flac_path, dtype='float32')
    return audio, sr


def mix_audio_with_snr(clean_audio, noise_audio, snr_db):
    """
    Mix clean audio with noise at specified SNR.
    
    Args:
        clean_audio: numpy array of clean audio
        noise_audio: numpy array of noise audio (will be truncated/looped to match length)
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        mixed_audio: numpy array of mixed audio
    """
    # Ensure noise is same length as clean audio
    clean_len = len(clean_audio)
    noise_len = len(noise_audio)
    
    if noise_len < clean_len:
        # Loop noise to match clean audio length
        repeats = int(np.ceil(clean_len / noise_len))
        noise_audio = np.tile(noise_audio, repeats)[:clean_len]
    else:
        # Truncate noise to match clean audio length
        noise_audio = noise_audio[:clean_len]
    
    # Calculate RMS
    clean_rms = np.sqrt(np.mean(clean_audio ** 2))
    noise_rms = np.sqrt(np.mean(noise_audio ** 2))
    
    # Avoid division by zero
    if noise_rms < 1e-10:
        return clean_audio
    
    # Calculate scaling factor for noise based on SNR
    # SNR = 20 * log10(signal_rms / noise_rms)
    # scaling_factor = signal_rms / (noise_rms * 10^(SNR/20))
    snr_linear = 10 ** (snr_db / 20)
    scaling_factor = clean_rms / (noise_rms * snr_linear)
    
    # Mix audio
    mixed_audio = clean_audio + scaling_factor * noise_audio
    
    # Normalize to prevent clipping
    max_val = np.abs(mixed_audio).max()
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val * 0.99
    
    return mixed_audio


def apply_reverb(clean_audio, impulse_response):
    """
    Apply reverberation to audio using an impulse response.
    
    Args:
        clean_audio: numpy array of clean audio
        impulse_response: numpy array of impulse response
    
    Returns:
        reverb_audio: numpy array of reverberated audio
    """
    # Normalize impulse response to prevent energy loss
    # Room impulse responses should preserve overall energy
    ir_energy = np.sqrt(np.sum(impulse_response ** 2))
    if ir_energy > 0:
        impulse_response = impulse_response / ir_energy
    
    # Use FFT-based convolution for efficiency
    reverb_audio = fftconvolve(clean_audio, impulse_response, mode='full')
    
    # Truncate to original length
    reverb_audio = reverb_audio[:len(clean_audio)]
    
    # Scale to match original RMS energy (preserve loudness)
    clean_rms = np.sqrt(np.mean(clean_audio ** 2))
    reverb_rms = np.sqrt(np.mean(reverb_audio ** 2))
    
    if reverb_rms > 1e-10:
        # Scale reverb audio to match clean audio RMS
        reverb_audio = reverb_audio * (clean_rms / reverb_rms)
    
    # Only normalize if clipping would occur
    max_val = np.abs(reverb_audio).max()
    if max_val > 1.0:
        reverb_audio = reverb_audio / max_val * 0.99
    
    return reverb_audio


def parse_additive_signals(additive_str):
    """
    Parse the --additive-signals parameter from Kaldi command.
    
    Example:
        'wav-reverberate --duration=10.31 "musan/speech/librivox/speech-librivox-0137.wav" - |,
         wav-reverberate --duration=10.31 "musan/speech/us-gov/speech-us-gov-0220.wav" - |'
    
    Returns:
        List of tuples: [(duration, file_path), ...]
    """
    signals = []
    # Split by comma but not within quotes
    parts = re.split(r',(?=wav-reverberate)', additive_str)
    
    for part in parts:
        # Extract duration and file path
        duration_match = re.search(r'--duration=([\d.]+)', part)
        file_match = re.search(r'"([^"]+)"', part)
        
        if duration_match and file_match:
            duration = float(duration_match.group(1))
            file_path = file_match.group(1)
            signals.append((duration, file_path))
    
    return signals


def parse_impulse_response(ir_str):
    """
    Parse the --impulse-response parameter from Kaldi command.
    
    Example:
        'sox RIRS_NOISES/simulated_rirs/smallroom/Room001/Room001-00007.wav -r 16000 -t wav - |'
    
    Returns:
        file_path: path to impulse response file
    """
    # Extract file path (first argument after sox)
    match = re.search(r'sox\s+([^\s]+\.wav)', ir_str)
    if match:
        return match.group(1)
    return None


def parse_kaldi_command(command, base_dir):
    """
    Parse Kaldi command to extract clean file, noise files/impulse response, and SNRs.
    
    Handles two types of augmentation:
    1. Additive noise (babble, music, noise) - uses --additive-signals and --snrs
    2. Reverberation - uses --impulse-response
    
    Returns:
        clean_file: path to clean FLAC file
        noise_files: list of noise file paths (empty for reverb)
        snrs: list of SNR values in dB (empty for reverb)
        impulse_response: path to impulse response file (None for additive noise)
    """
    # Extract clean FLAC file
    clean_match = re.search(r'flac -d -c -s (.+?\.flac)', command)
    if not clean_match:
        raise ValueError("Could not find clean FLAC file in command")
    clean_file = clean_match.group(1)
    
    # Check if this is reverb (impulse response) or additive noise
    impulse_match = re.search(r'--impulse-response="([^"]+)"', command)
    
    if impulse_match:
        # This is a reverb command
        ir_str = impulse_match.group(1)
        impulse_file = parse_impulse_response(ir_str)
        
        if not impulse_file:
            raise ValueError("Could not extract impulse response file from command")
        
        clean_file = Path(base_dir) / clean_file
        impulse_file = Path(base_dir) / impulse_file
        
        return clean_file, [], [], impulse_file
    
    else:
        # This is an additive noise command
        additive_match = re.search(r"--additive-signals='([^']+)'", command)
        if not additive_match:
            raise ValueError("Could not find additive signals in command")
        additive_str = additive_match.group(1)
        
        signals = parse_additive_signals(additive_str)
        noise_files = [file_path for _, file_path in signals]
        
        # Extract SNRs
        snr_match = re.search(r"--snrs='([^']+)'", command)
        if not snr_match:
            raise ValueError("Could not find SNRs in command")
        snrs = [float(x) for x in snr_match.group(1).split(',')]
        
        # Convert to absolute paths
        clean_file = Path(base_dir) / clean_file
        noise_files = [Path(base_dir) / f for f in noise_files]
        
        return clean_file, noise_files, snrs, None


def create_augmented_audio(clean_file, noise_files, snrs, impulse_response_file=None):
    """
    Create augmented audio by mixing clean audio with noise or applying reverberation.
    
    Args:
        clean_file: path to clean audio file
        noise_files: list of noise file paths (empty for reverb)
        snrs: list of SNR values in dB (empty for reverb)
        impulse_response_file: path to impulse response file (None for additive noise)
    
    Returns:
        augmented_audio: numpy array of mixed/reverberated audio
        sr: sample rate
    """
    # Load clean audio
    clean_audio, sr = load_flac_audio(clean_file)
    
    # Check if this is reverb or additive noise
    if impulse_response_file is not None:
        # Apply reverberation
        try:
            impulse_response, ir_sr = sf.read(impulse_response_file, dtype='float32')
            
            # Resample if needed
            if ir_sr != sr:
                print(f"  Warning: Sample rate mismatch for impulse response ({ir_sr} vs {sr})")
                # For now, skip resampling and just return clean audio
                return clean_audio, sr
            
            # Apply reverb
            augmented_audio = apply_reverb(clean_audio, impulse_response)
            
        except FileNotFoundError:
            print(f"  Warning: Impulse response file not found: {impulse_response_file}")
            augmented_audio = clean_audio
    
    else:
        # Mix with additive noise
        mixed_audio = clean_audio.copy()
        
        for noise_file, snr_db in zip(noise_files, snrs):
            try:
                noise_audio, noise_sr = load_flac_audio(noise_file)
                
                # Resample if needed (though LibriSpeech and MUSAN are both 16kHz)
                if noise_sr != sr:
                    print(f"  Warning: Sample rate mismatch for {noise_file.name} ({noise_sr} vs {sr})")
                    continue
                
                # Mix with current audio
                mixed_audio = mix_audio_with_snr(mixed_audio, noise_audio, snr_db)
            
            except FileNotFoundError:
                print(f"  Warning: Noise file not found: {noise_file}")
                continue
        
        augmented_audio = mixed_audio
    
    return augmented_audio, sr


def export_augmented_audio(scp_file, output_dir, format='wav', max_files=None, verbose=False, base_dir=None):
    """
    Export augmented audio from wav.scp to actual audio files using pure Python.
    
    Args:
        scp_file: Path to Kaldi wav.scp file
        output_dir: Directory to save audio files
        format: Output format ('wav' or 'flac')
        max_files: Maximum number of files to export (None = all)
        verbose: Print detailed error messages
        base_dir: Base directory for resolving relative paths in commands
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine base directory (directory containing the scp file)
    if base_dir is None:
        base_dir = Path(scp_file).parent.parent.parent
    else:
        base_dir = Path(base_dir)
    
    print(f"📁 Base directory: {base_dir}")
    
    # Parse wav.scp
    print(f"📖 Reading {scp_file}...")
    utterances = parse_wav_scp(scp_file)
    
    if max_files:
        utterances = utterances[:max_files]
    
    print(f"✓ Found {len(utterances)} utterances")
    print(f"💾 Exporting to: {output_dir}")
    print(f"🎵 Format: {format.upper()}")
    print("")
    
    # Export each utterance
    success_count = 0
    failed = []
    
    for utt_id, command in tqdm(utterances, desc="Exporting audio"):
        output_file = output_path / f"{utt_id}.{format}"
        
        # Skip if file already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            success_count += 1
            continue
        
        try:
            # Parse command to extract file paths and parameters
            clean_file, noise_files, snrs, impulse_response = parse_kaldi_command(command, base_dir)
            
            # Create augmented audio
            augmented_audio, sr = create_augmented_audio(clean_file, noise_files, snrs, impulse_response)
            
            # Save to file
            sf.write(output_file, augmented_audio, sr, format=format.upper())
            
            success_count += 1
        
        except Exception as e:
            failed.append((utt_id, str(e)))
            if verbose:
                print(f"\n❌ Failed: {utt_id}: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✓ Successfully exported: {success_count}/{len(utterances)} files")
    
    if failed:
        print(f"❌ Failed: {len(failed)} files")
        if len(failed) <= 10:
            print("\nFailed utterances:")
            for utt_id, error in failed:
                print(f"  - {utt_id}: {error}")
        else:
            print(f"  (Too many to list, first 5:)")
            for utt_id, error in failed[:5]:
                print(f"  - {utt_id}: {error}")
    
    print(f"{'='*70}\n")
    
    return success_count, failed


def main():
    parser = argparse.ArgumentParser(
        description='Export Kaldi augmented audio using pure Python (no Kaldi required)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export babble augmentation to WAV
  python export_augmented_audio_pure_python.py kaldi/egs/pvad/data/babble/wav.scp output/babble
  
  # Export music augmentation to FLAC
  python export_augmented_audio_pure_python.py kaldi/egs/pvad/data/music/wav.scp output/music --format flac
  
  # Export only first 10 files for testing
  python export_augmented_audio_pure_python.py kaldi/egs/pvad/data/noise/wav.scp output/noise --max-files 10
  
  # Specify base directory for resolving paths
  python export_augmented_audio_pure_python.py data/babble/wav.scp output/babble --base-dir kaldi/egs/pvad

Requirements:
  - soundfile: pip install soundfile
  - numpy: pip install numpy
  - scipy: pip install scipy
  - tqdm: pip install tqdm
        """
    )
    
    parser.add_argument('scp_file', help='Path to Kaldi wav.scp file')
    parser.add_argument('output_dir', help='Output directory for audio files')
    parser.add_argument('--format', choices=['wav', 'flac'], default='wav',
                       help='Output audio format (default: wav)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to export (default: all)')
    parser.add_argument('--base-dir', default=None,
                       help='Base directory for resolving relative paths (default: auto-detect from scp path)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed error messages')
    
    args = parser.parse_args()
    
    # Check if scp file exists
    if not Path(args.scp_file).exists():
        print(f"❌ Error: File not found: {args.scp_file}")
        sys.exit(1)
    
    # Export audio
    success_count, failed = export_augmented_audio(
        args.scp_file,
        args.output_dir,
        format=args.format,
        max_files=args.max_files,
        verbose=args.verbose,
        base_dir=args.base_dir
    )
    
    # Exit with error code if any files failed
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
