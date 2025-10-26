#!/usr/bin/python

"""@package play_overlap_sample

Play and visualize overlapping utterance samples.

This script loads and plays sample overlapping utterances, showing:
- Waveform visualization
- Overlap regions highlighted
- Speaker information
- Audio playback

"""

import argparse as ap
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio playback disabled.")


def load_overlap_metadata(meta_path):
    """Load overlap metadata from file."""
    metadata = {}
    overlaps = []
    
    if not os.path.exists(meta_path):
        return metadata, overlaps
    
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('main_speakers:'):
                metadata['main_speakers'] = line.split(':')[1].strip().split(',')
            elif line.startswith('overlap_count:'):
                metadata['overlap_count'] = int(line.split(':')[1].strip())
            elif line.startswith('overlap_'):
                # Parse overlap_N: start-end speakers=... amplitude=...
                parts = line.split(':')
                times = parts[1].strip().split()[0]
                start, end = times.split('-')
                
                # Extract speakers
                speakers_part = [p for p in parts[1].split() if p.startswith('speakers=')][0]
                speakers = speakers_part.split('=')[1].split(',')
                
                # Extract amplitude
                amp_part = [p for p in parts[1].split() if p.startswith('amplitude=')][0]
                amplitude = float(amp_part.split('=')[1])
                
                overlaps.append({
                    'start': float(start),
                    'end': float(end),
                    'speakers': speakers,
                    'amplitude': amplitude
                })
    
    return metadata, overlaps


def plot_waveform_with_overlaps(audio, sr, overlaps, save_path=None, show_plot=False):
    """Plot waveform with overlap regions highlighted."""
    
    time = np.arange(len(audio)) / sr
    duration = len(audio) / sr
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot waveform
    ax.plot(time, audio, linewidth=0.5, color='steelblue', alpha=0.7)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax.set_title('Overlapping Utterance Waveform', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight overlap regions
    colors = ['red', 'orange', 'purple', 'green']
    for idx, overlap in enumerate(overlaps):
        color = colors[idx % len(colors)]
        ax.axvspan(overlap['start'], overlap['end'], 
                   alpha=0.3, color=color, 
                   label=f"Overlap {idx+1}: {','.join(overlap['speakers'][:2])}")
        
        # Add text annotation
        mid_time = (overlap['start'] + overlap['end']) / 2
        ax.text(mid_time, ax.get_ylim()[1] * 0.9, 
               f"OV{idx+1}\n{overlap['amplitude']:.2f}x",
               horizontalalignment='center',
               verticalalignment='top',
               fontsize=9,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    if overlaps:
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved plot: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)


def play_audio(audio, sr):
    """Play audio using sounddevice."""
    if not AUDIO_AVAILABLE:
        print("Audio playback not available (sounddevice not installed)")
        return
    
    print(f"Playing audio... ({len(audio)/sr:.2f} seconds)")
    sd.play(audio, sr)
    sd.wait()
    print("Playback complete!")


def main():
    parser = ap.ArgumentParser(
        description="Play and visualize overlapping utterance samples"
    )
    parser.add_argument('--overlap_dir', type=str, required=True,
                       help="Directory containing overlapping utterances")
    parser.add_argument('--sample_idx', type=int, default=0,
                       help="Index of sample to play (default: 0 = random)")
    parser.add_argument('--num_samples', type=int, default=5,
                       help="Number of samples to visualize (default: 5)")
    parser.add_argument('--output_dir', type=str, default='overlap_samples',
                       help="Directory to save audio and plots (default: overlap_samples)")
    parser.add_argument('--no_save', action='store_true',
                       help="Don't save files, just show visualization")
    parser.add_argument('--show_plot', action='store_true',
                       help="Display plots interactively (default: save only)")
    parser.add_argument('--no_play', action='store_true',
                       help="Don't play audio, just show visualization")
    parser.add_argument('--save_plot', type=str, default=None,
                       help="Path to save the waveform plot (overrides output_dir)")
    args = parser.parse_args()
    
    overlap_dir = args.overlap_dir
    if overlap_dir[-1] != '/':
        overlap_dir += '/'
    
    # Find all flac files
    pattern = overlap_dir + '**/*.flac'
    audio_files = glob(pattern, recursive=True)
    
    if not audio_files:
        print(f"No audio files found in {overlap_dir}")
        return
    
    print(f"Found {len(audio_files)} overlapping utterances")
    
    # Create output directory
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nOutput directory: {args.output_dir}")
    
    # Determine which samples to process
    if args.sample_idx == 0:
        # Random samples
        num_samples = min(args.num_samples, len(audio_files))
        sample_indices = np.random.choice(len(audio_files), size=num_samples, replace=False)
        print(f"Randomly selected {num_samples} samples")
    else:
        # Single specific sample
        sample_indices = [min(args.sample_idx - 1, len(audio_files) - 1)]
        num_samples = 1
    
    # Process each sample
    for sample_num, sample_idx in enumerate(sample_indices, 1):
        audio_path = audio_files[sample_idx]
        print(f"\n{'='*60}")
        print(f"Sample {sample_num}/{num_samples} (index #{sample_idx})")
        print(f"{'='*60}")
        print(f"Loading: {os.path.basename(audio_path)}")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        print(f"Shape: {audio.shape}")
        
        # Load metadata
        meta_path = audio_path.replace('.flac', '.overlap_meta')
        metadata, overlaps = load_overlap_metadata(meta_path)
        
        if metadata:
            print(f"\n📊 Metadata:")
            print(f"  Main speakers: {', '.join(metadata.get('main_speakers', ['Unknown']))}")
            print(f"  Overlap count: {metadata.get('overlap_count', 0)}")
            
            for idx, overlap in enumerate(overlaps):
                print(f"\n  Overlap {idx+1}:")
                print(f"    Time: {overlap['start']:.2f}s - {overlap['end']:.2f}s "
                      f"(duration: {overlap['end']-overlap['start']:.2f}s)")
                print(f"    Speakers: {', '.join(overlap['speakers'])}")
                print(f"    Amplitude: {overlap['amplitude']:.2f}x")
        else:
            print("No metadata found")
        
        # Save audio copy to output directory
        if not args.no_save:
            output_audio_path = os.path.join(args.output_dir, f"sample_{sample_num}_{os.path.basename(audio_path)}")
            sf.write(output_audio_path, audio, sr)
            print(f"\n✅ Saved audio: {output_audio_path}")
            
            # Copy metadata file
            if os.path.exists(meta_path):
                output_meta_path = output_audio_path.replace('.flac', '.overlap_meta')
                with open(meta_path, 'r') as src, open(output_meta_path, 'w') as dst:
                    dst.write(src.read())
                print(f"✅ Saved metadata: {output_meta_path}")
        
        # Visualize
        print("\nGenerating waveform plot...")
        save_path = None
        if not args.no_save:
            if args.save_plot:
                # Use custom path if provided
                if num_samples > 1:
                    base, ext = os.path.splitext(args.save_plot)
                    save_path = f"{base}_{sample_num}{ext}"
                else:
                    save_path = args.save_plot
            else:
                # Default: save to output directory
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                save_path = os.path.join(args.output_dir, f"sample_{sample_num}_{base_name}.png")
        
        plot_waveform_with_overlaps(audio, sr, overlaps, save_path=save_path, show_plot=args.show_plot)
        
        # Play audio (only for single sample or if explicitly requested)
        if not args.no_play and num_samples == 1:
            response = input("\nPlay audio? (y/n): ")
            if response.lower() == 'y':
                play_audio(audio, sr)
    
    print(f"\n{'='*60}")
    print(f"Processed {num_samples} sample(s)")
    if not args.no_save:
        print(f"\n📁 All files saved to: {args.output_dir}/")
        print(f"   - {num_samples} audio files (.flac)")
        print(f"   - {num_samples} metadata files (.overlap_meta)")
        print(f"   - {num_samples} waveform plots (.png)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
