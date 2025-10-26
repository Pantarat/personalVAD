#!/usr/bin/python

"""
Quick example: Generate and visualize overlapping utterances

This script demonstrates the overlap generation pipeline with a small example.
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "="*70)
    print(f"📝 {description}")
    print("="*70)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error: Command failed with exit code {result.returncode}")
        return False
    print(f"\n✅ Success!")
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  PersonalVAD - Overlapping Utterances Generation Example       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

This example will:
  1. Generate 50 overlapping utterances (30% overlap, 0.7x amplitude)
  2. Visualize a sample
  3. Show metadata

Prerequisites:
  - LibriSpeech test-clean and test-other datasets downloaded
  - LibriSpeech alignments downloaded
  - Python dependencies installed (numpy, librosa, soundfile, matplotlib)
""")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Configuration
    repo_root = os.path.dirname(os.path.abspath(__file__))
    overlap_dir = os.path.join(repo_root, 'data', 'overlap_example')
    
    print(f"\n📁 Output directory: {overlap_dir}")
    
    # Step 1: Generate overlapping utterances
    cmd = f"""python src/generate_overlapping_utterances.py \
        --libri_root data/LibriSpeech \
        --concat_dir {overlap_dir} \
        --count 50 \
        --overlap_pct 30 \
        --amplitude_ratio 0.7 \
        test-clean test-other"""
    
    if not run_command(cmd, "Generating 50 overlapping utterances"):
        return
    
    # Step 2: Check output
    print("\n" + "="*70)
    print("📊 Generated Files:")
    print("="*70)
    
    wav_scp = os.path.join(overlap_dir, 'wav.scp')
    if os.path.exists(wav_scp):
        with open(wav_scp, 'r') as f:
            lines = f.readlines()
        print(f"  ✓ wav.scp: {len(lines)} entries")
    
    text_file = os.path.join(overlap_dir, 'text')
    if os.path.exists(text_file):
        with open(text_file, 'r') as f:
            lines = f.readlines()
        print(f"  ✓ text: {len(lines)} entries")
    
    # Count overlap metadata files
    import glob
    meta_files = glob.glob(os.path.join(overlap_dir, '**', '*.overlap_meta'), recursive=True)
    print(f"  ✓ Metadata files: {len(meta_files)}")
    
    # Step 3: Show sample metadata
    if meta_files:
        print("\n" + "="*70)
        print("📄 Sample Metadata (first file):")
        print("="*70)
        with open(meta_files[0], 'r') as f:
            print(f.read())
    
    # Step 4: Visualize sample
    print("\n" + "="*70)
    print("🎨 Visualizing Sample")
    print("="*70)
    
    cmd = f"python src/play_overlap_sample.py --overlap_dir {overlap_dir} --no_play"
    run_command(cmd, "Opening waveform visualization")
    
    # Step 5: Next steps
    print("\n" + "="*70)
    print("✅ Example Complete!")
    print("="*70)
    print(f"""
Next steps:

1. Listen to samples:
   python src/play_overlap_sample.py --overlap_dir {overlap_dir}

2. Generate full dataset with augmentation:
   bash prepare_overlap_dataset.sh 0

3. Extract features for training:
   bash prepare_overlap_dataset.sh 3

4. Train model:
   python src/personal_vad.py --train_data data/overlap_features

See doc/OVERLAP_GENERATION.md for full documentation.
""")


if __name__ == '__main__':
    main()
