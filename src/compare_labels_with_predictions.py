#!/usr/bin/env python3
"""
Compare ground truth labels with model predictions for PersonalVAD inference.

This script loads:
1. Audio file
2. Ground truth labels (from .overlap_meta or labels.scp if available)
3. Model predictions
4. Visualizes and analyzes the comparison

Author: PersonalVAD Demo Extension
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import soundfile as sf
from pathlib import Path
import re
from scipy.signal import medfilt

try:
    import kaldiio
    KALDIIO_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: kaldiio not available. Install with: pip install kaldiio")
    KALDIIO_AVAILABLE = False

# Import PersonalVAD modules
from personal_vad import PersonalVAD
from resemblyzer_mod import VoiceEncoderMod


class LabelComparator:
    """Compare ground truth labels with model predictions."""
    
    def __init__(self, sample_rate=16000, frame_duration=0.01):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.samples_per_frame = int(sample_rate * frame_duration)
        
        self.class_names = ['NS', 'NTSS', 'TSS']
        self.class_colors = ['gray', 'orange', 'green']
        
    def load_labels_from_ark(self, labels_scp_path, utterance_id):
        """
        Load ground truth labels from Kaldi ark file.
        
        Args:
            labels_scp_path: Path to labels.scp file
            utterance_id: Utterance ID to load labels for
            
        Returns:
            numpy.ndarray: Frame-level labels (0=NS, 1=NTSS, 2=TSS) or None if not found
        """
        if not KALDIIO_AVAILABLE:
            print("❌ kaldiio is required to load labels from ark files")
            print("   Install with: pip install kaldiio")
            return None
            
        if not Path(labels_scp_path).exists():
            print(f"⚠️ Labels file not found: {labels_scp_path}")
            return None
        
        try:
            # Load the labels using kaldiio
            labels_dict = kaldiio.load_scp(labels_scp_path)
            
            if utterance_id not in labels_dict:
                print(f"⚠️ Utterance '{utterance_id}' not found in labels file")
                print(f"   Available utterances (first 5): {list(labels_dict.keys())[:5]}")
                return None
            
            labels = labels_dict[utterance_id]
            
            # Convert to integer array (labels are stored as float32 in ark files)
            labels = labels.astype(np.int32)
            print(labels[-100:])
            
            print(f"✓ Loaded labels for '{utterance_id}': {len(labels)} frames")
            return labels
            
        except Exception as e:
            print(f"❌ Error loading labels: {e}")
            return None
    
    def parse_overlap_meta(self, meta_path):
        """
        Parse .overlap_meta file to extract ground truth label information.
        
        Returns:
            dict: Contains overlap segments and standalone NTSS segments with timestamps
        """
        if not Path(meta_path).exists():
            print(f"⚠️ Metadata file not found: {meta_path}")
            return None
            
        metadata = {
            'overlaps': [],
            'standalone_ntss': [],
            'main_speakers': [],
            'overlap_pct': 0
        }
        
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('main_speakers:'):
                    metadata['main_speakers'] = line.split(':')[1].strip().split(',')
                elif line.startswith('overlap_percentage_target:'):
                    metadata['overlap_pct'] = int(line.split(':')[1].strip().rstrip('%'))
                elif line.startswith('overlap_'):
                    # Parse: overlap_0: 3.53-10.42 speakers=8224-274384,1998-15444 amplitude=0.2 type=speech_overlap
                    match = re.search(r'(\d+\.\d+)-(\d+\.\d+)\s+speakers=([^\s]+)', line)
                    if match:
                        start, end, speakers = match.groups()
                        metadata['overlaps'].append({
                            'start': float(start),
                            'end': float(end),
                            'speakers': speakers.split(',')
                        })
                elif line.startswith('standalone_ntss_'):
                    # Parse: standalone_ntss_0: 0.00-0.51 speakers=1998-15444 amplitude=0.2 type=gap_ntss
                    match = re.search(r'(\d+\.\d+)-(\d+\.\d+)\s+speakers=([^\s]+)', line)
                    if match:
                        start, end, speakers = match.groups()
                        metadata['standalone_ntss'].append({
                            'start': float(start),
                            'end': float(end),
                            'speakers': speakers.split(',')
                        })
        
        return metadata
    
    def create_ground_truth_labels(self, audio_duration, metadata, target_speaker_id):
        """
        Create frame-level ground truth labels from metadata.
        
        Args:
            audio_duration: Duration in seconds
            metadata: Parsed metadata from .overlap_meta file
            target_speaker_id: Target speaker ID for the session
            
        Returns:
            numpy.ndarray: Frame-level labels (0=NS, 1=NTSS, 2=TSS)
        """
        num_frames = int(audio_duration / self.frame_duration)
        labels = np.zeros(num_frames, dtype=int)  # Start with all NS (0)
        
        # This is a simplified labeling - in reality we'd need alignment timestamps
        # from the original LibriSpeech files to know exactly when speech occurs
        
        # For now, we'll mark:
        # - Overlap regions as areas where predictions should show both TSS and NTSS
        # - Standalone NTSS regions as NTSS
        # - The rest would need proper alignment data
        
        print("\n📋 Creating ground truth labels from metadata:")
        print(f"  Audio duration: {audio_duration:.2f}s ({num_frames} frames)")
        print(f"  Target speaker: {target_speaker_id}")
        print(f"  Overlaps: {len(metadata['overlaps'])}")
        print(f"  Standalone NTSS: {len(metadata['standalone_ntss'])}")
        
        # Mark overlapping regions (these should ideally be TSS, but will have NTSS interference)
        for ov in metadata['overlaps']:
            start_frame = int(ov['start'] / self.frame_duration)
            end_frame = int(ov['end'] / self.frame_duration)
            # In overlap regions, we expect the target speaker (TSS=2) is present
            # but model might show mix of TSS and NTSS
            labels[start_frame:end_frame] = 2  # Mark as TSS (target present during overlap)
        
        # Mark standalone NTSS regions
        for ntss in metadata['standalone_ntss']:
            start_frame = int(ntss['start'] / self.frame_duration)
            end_frame = int(ntss['end'] / self.frame_duration)
            labels[start_frame:end_frame] = 1  # Mark as NTSS
        
        return labels
    
    def compute_metrics(self, ground_truth, predictions):
        """
        Compute comparison metrics between ground truth and predictions.
        
        Args:
            ground_truth: Frame-level ground truth labels
            predictions: Frame-level predictions from model
            
        Returns:
            dict: Metrics including accuracy, precision, recall, F1 per class
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        # Ensure same length
        min_len = min(len(ground_truth), len(predictions))
        gt = ground_truth[:min_len]
        pred = predictions[:min_len]
        
        # Overall accuracy
        accuracy = accuracy_score(gt, pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            gt, pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(gt, pred, labels=[0, 1, 2])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def visualize_comparison(self, audio, ground_truth, predictions, 
                           predictions_probs=None, save_path=None):
        """
        Create comprehensive visualization comparing ground truth with predictions.
        
        Args:
            audio: Audio samples
            ground_truth: Frame-level ground truth labels
            predictions: Frame-level predictions
            predictions_probs: Prediction probabilities (optional)
            save_path: Path to save figure (optional)
        """
        # Check for frame mismatch
        audio_duration_sec = len(audio) / self.sample_rate
        expected_frames = int(audio_duration_sec / self.frame_duration)
        
        if len(ground_truth) > expected_frames:
            print(f"\n⚠️ Frame mismatch detected:")
            print(f"   Audio duration: {audio_duration_sec:.3f}s ({expected_frames} frames expected)")
            print(f"   Ground truth: {len(ground_truth)} frames")
            print(f"   Extra frames: {len(ground_truth) - expected_frames} ({(len(ground_truth) - expected_frames) * self.frame_duration:.3f}s)")
            print(f"   Trimming ground truth to match audio duration...")
            ground_truth = ground_truth[:expected_frames]
        
        if len(predictions) > expected_frames:
            print(f"   Predictions: {len(predictions)} frames")
            print(f"   Trimming predictions to match audio duration...")
            predictions = predictions[:expected_frames]
            if predictions_probs is not None:
                predictions_probs = predictions_probs[:expected_frames]
        
        # Apply median filtering to predictions for smoother visualization
        predictions_smooth = medfilt(predictions, kernel_size=5)
        
        # Create time axis
        time_audio = np.arange(len(audio)) / self.sample_rate
        time_labels = np.arange(len(predictions)) * self.frame_duration
        time_labels_gt = np.arange(len(ground_truth)) * self.frame_duration
        
        # Create figure
        fig, axes = plt.subplots(5, 1, figsize=(16, 12))
        
        # 1. Audio waveform
        axes[0].plot(time_audio, audio, linewidth=0.5, color='blue', alpha=0.7)
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].set_title('Audio Waveform', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim([0, time_audio[-1]])
        
        # 2. Ground Truth Labels
        for i in range(len(ground_truth) - 1):
            axes[1].axvspan(time_labels_gt[i], time_labels_gt[i+1], 
                          facecolor=self.class_colors[ground_truth[i]], 
                          alpha=0.8, linewidth=0)
        axes[1].set_ylabel('Class', fontsize=10)
        axes[1].set_title('Ground Truth Labels', fontsize=12, fontweight='bold', color='darkblue')
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(self.class_names)
        axes[1].set_xlim([0, max(time_labels_gt[-1], time_audio[-1])])
        axes[1].grid(alpha=0.3, axis='x')
        
        # Add vertical line at audio end if labels extend beyond
        if time_labels_gt[-1] > time_audio[-1]:
            axes[1].axvline(time_audio[-1], color='red', linestyle='--', linewidth=2, 
                           label=f'Audio end ({time_audio[-1]:.2f}s)', alpha=0.7)
            axes[1].legend(loc='upper right')
        
        # 3. Model Predictions (Smoothed)
        for i in range(len(predictions_smooth) - 1):
            axes[2].axvspan(time_labels[i], time_labels[i+1], 
                          facecolor=self.class_colors[int(predictions_smooth[i])], 
                          alpha=0.8, linewidth=0)
        axes[2].set_ylabel('Class', fontsize=10)
        axes[2].set_title('Model Predictions (Smoothed)', fontsize=12, fontweight='bold', color='darkgreen')
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(self.class_names)
        axes[2].set_xlim([0, time_labels[-1]])
        axes[2].grid(alpha=0.3, axis='x')
        
        # 4. Prediction Probabilities (if available)
        if predictions_probs is not None:
            for i, name in enumerate(self.class_names):
                axes[3].plot(time_labels, predictions_probs[:, i], 
                           label=name, linewidth=2, color=self.class_colors[i], alpha=0.8)
            axes[3].set_ylabel('Probability', fontsize=10)
            axes[3].set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
            axes[3].legend(loc='upper right', fontsize=10)
            axes[3].grid(alpha=0.3)
            axes[3].set_xlim([0, time_labels[-1]])
            axes[3].set_ylim([0, 1])
        else:
            axes[3].text(0.5, 0.5, 'Probabilities not available', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        
        # 5. Difference Visualization (where predictions differ from ground truth)
        min_len = min(len(ground_truth), len(predictions_smooth))
        differences = (ground_truth[:min_len] != predictions_smooth[:min_len]).astype(int)
        
        for i in range(len(differences) - 1):
            color = 'red' if differences[i] == 1 else 'lightgreen'
            axes[4].axvspan(time_labels[i], time_labels[i+1], 
                          facecolor=color, alpha=0.6, linewidth=0)
        axes[4].set_ylabel('Match', fontsize=10)
        axes[4].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        axes[4].set_title('Prediction vs Ground Truth (Red = Mismatch, Green = Match)', 
                         fontsize=12, fontweight='bold', color='darkred')
        axes[4].set_yticks([0, 1])
        axes[4].set_yticklabels(['Match', 'Diff'])
        axes[4].set_xlim([0, time_labels[-1]])
        axes[4].grid(alpha=0.3, axis='x')
        
        # Add overall statistics as text
        mismatch_pct = (differences.sum() / len(differences)) * 100
        fig.text(0.99, 0.01, f'Frame-level Mismatch: {mismatch_pct:.1f}%', 
                ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n💾 Comparison visualization saved to: {save_path}")
        
        plt.show()
    
    def print_metrics(self, metrics):
        """Print detailed metrics in a nice format."""
        print("\n" + "=" * 70)
        print("📊 EVALUATION METRICS: Ground Truth vs Model Predictions")
        print("=" * 70)
        
        print(f"\n🎯 Overall Accuracy: {metrics['accuracy']:.2%}")
        
        print("\n📈 Per-Class Metrics:")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        for i, name in enumerate(self.class_names):
            print(f"{name:<10} {metrics['precision'][i]:>10.2%}  "
                  f"{metrics['recall'][i]:>10.2%}  "
                  f"{metrics['f1'][i]:>10.2%}  "
                  f"{int(metrics['support'][i]):>8}")
        
        print("\n🔢 Confusion Matrix:")
        print("         Predicted ->")
        print(f"         {'NS':<8} {'NTSS':<8} {'TSS':<8}")
        print("Actual ↓")
        for i, name in enumerate(self.class_names):
            print(f"{name:<8} ", end='')
            for j in range(3):
                print(f"{metrics['confusion_matrix'][i, j]:>6}  ", end='')
            print()
        
        print("\n" + "=" * 70)


def main():
    """Example usage of LabelComparator."""
    
    # Configuration
    AUDIO_PATH = 'path/to/audio.flac'
    META_PATH = 'path/to/audio.overlap_meta'
    MODEL_PATH = 'path/to/model.pt'
    TARGET_SPEAKER_ID = '8224-274384'
    
    # Initialize comparator
    comparator = LabelComparator()
    
    # Load audio
    print(f"Loading audio from: {AUDIO_PATH}")
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    audio_duration = len(audio) / 16000
    
    # Load metadata and create ground truth labels
    print(f"Loading metadata from: {META_PATH}")
    metadata = comparator.parse_overlap_meta(META_PATH)
    
    if metadata:
        ground_truth = comparator.create_ground_truth_labels(
            audio_duration, metadata, TARGET_SPEAKER_ID
        )
        
        # TODO: Run model inference to get predictions
        # predictions = run_model_inference(audio, model, target_embedding)
        
        # For now, create dummy predictions for demonstration
        predictions = np.random.randint(0, 3, len(ground_truth))
        predictions_probs = np.random.rand(len(ground_truth), 3)
        predictions_probs = predictions_probs / predictions_probs.sum(axis=1, keepdims=True)
        
        # Compute metrics
        metrics = comparator.compute_metrics(ground_truth, predictions)
        comparator.print_metrics(metrics)
        
        # Visualize comparison
        comparator.visualize_comparison(
            audio, ground_truth, predictions, predictions_probs,
            save_path='label_comparison.png'
        )
    else:
        print("⚠️ Could not load metadata. Skipping comparison.")


if __name__ == '__main__':
    main()
