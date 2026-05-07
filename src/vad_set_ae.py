"""@package vad_set_ae

Modified from vad_set.py to optionally use autoencoder-compressed d-vectors

This module implements the SET personal VAD architecture training loop with
optional autoencoder processing of the enrolled speaker d-vector.

The workflow supports three modes:

1. USE_AUTOENCODER=False (original behavior):
   - Use full enrolled speaker d-vector (256-dim)
   - Use pre-computed scores from scores.scp
   - Repeat full enrolled d-vector for all frames
   - Concatenate: [40 fbanks + 1 score + 256 enrolled d-vector] = 297-dim

2. USE_AUTOENCODER=True, USE_AE_RECONSTRUCTION=False (bottleneck mode):
   - Compress enrolled speaker d-vector with autoencoder (256->64 dim)
   - Use pre-computed scores from scores.scp
   - Repeat compressed enrolled d-vector for all frames
   - Concatenate: [40 fbanks + 1 score + 64 compressed enrolled d-vector] = 105-dim

3. USE_AUTOENCODER=True, USE_AE_RECONSTRUCTION=True (full reconstruction mode):
   - Keep enrolled speaker d-vector as original (256-dim, NOT processed)
   - Recompute scores on-the-fly using autoencoder-reconstructed stream d-vectors
   - Repeat original enrolled d-vector for all frames
   - Concatenate: [40 fbanks + 1 score + 256 original enrolled d-vector] = 297-dim
   - Note: Enrolled d-vectors are NOT processed, but stream d-vectors ARE reconstructed for scoring

"""

import warnings
# Suppress FutureWarning from resemblyzer library (librosa positional args deprecation)
warnings.filterwarnings('ignore', category=FutureWarning, module='resemblyzer')
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import kaldiio
import argparse as ap

from sklearn.metrics import average_precision_score

import numpy as np
import os
import sys
from pathlib import Path
import librosa

from personal_vad import PersonalVAD, WPL, pad_collate
from dataset_utils import sample_evenly_across_augmentations
from resemblyzer import VoiceEncoder
from resemblyzer_mod import VoiceEncoderMod

# Add AE_test directory to path for autoencoder import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AE_test'))
from autoencoder_utils import DvectorAutoencoder, load_autoencoder
import pickle


def pad_collate_with_metadata(batch):
    """Custom padding function that handles keys from dataset"""
    (xx, yy, keys) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    x_padded = pad_sequence(xx, batch_first=True, padding_value=0)
    y_padded = pad_sequence(yy, batch_first=True, padding_value=0)

    return x_padded, y_padded, x_lens, y_lens, keys

# model hyper parameters
num_epochs = 50
batch_size = 32  # Reduced from 64 to avoid CUDA OOM
batch_size_test = 32

# Autoencoder configuration
USE_AUTOENCODER = True  # Set to False to use full 256-dim d-vectors
USE_AE_RECONSTRUCTION = False  # True: use full enc-dec (256->64->256), False: use bottleneck only (256->64)
RECOMPUTE_SCORES = True  # True: recompute scores with AE-processed stream d-vectors, False: use pre-computed scores.scp

# Model architecture (will be set based on USE_AUTOENCODER and USE_AE_RECONSTRUCTION)
if USE_AUTOENCODER:
    if USE_AE_RECONSTRUCTION:
        # Full reconstruction: 256-dim reconstructed d-vectors
        input_dim = 297  # 40 fbanks + 256 reconstructed d-vector + 1 score
        dvector_dim = 256
    else:
        # Bottleneck only: 64-dim compressed d-vectors
        input_dim = 105  # 40 fbanks + 64 compressed d-vector + 1 score
        dvector_dim = 64
else:
    input_dim = 297  # 40 fbanks + 256 full d-vector + 1 score
    dvector_dim = 256

hidden_dim = 64
out_dim = 3
num_layers = 2
lr = 1e-3
SCHEDULER = True

# Early stopping and checkpointing
EARLY_STOPPING_PATIENCE = 2  # Stop if no improvement for N epochs
SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs
METRIC_FOR_BEST = 'mAP'  # 'mAP' or 'accuracy'

DATA_TRAIN = 'data/train'
DATA_TEST = 'data/test'
EMBED_PATH = 'embeddings'
# Default autoencoder model path - can be overridden via --ae_model_path argument
# IMPORTANT: Use identity-trained autoencoder, NOT denoising autoencoder!
# Denoising AE pushes all d-vectors toward main speaker -> false positives
AE_MODEL_PATH = 'src/AE_test/test_outputs/dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000'
MODEL_PATH = f'vad_set{"_ae" if USE_AUTOENCODER else ""}.pt'
SAVE_MODEL = True

USE_WPL = True
NUM_WORKERS = 0

# Selects which of the scoring methods should be used...
# legend: scores[0,:] -> baseline, 1 -> partially-constant, 2 -> linearly-interpolated
SCORE_TYPE = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WPL_WEIGHTS = torch.tensor([1.0, 0.1, 1.0]).to(device)
learning_rate = lr


# def load_autoencoder(model_dir, device=device):
#     """Load trained autoencoder model"""
#     model_dir = Path(model_dir)
    
#     # If path is relative, resolve it from script location
#     if not model_dir.is_absolute():
#         script_dir = Path(__file__).parent.parent  # Go up from src/ to repo root
#         model_dir = script_dir / model_dir
    
#     # Load config
#     config_path = model_dir / 'config.pkl'
#     if not config_path.exists():
#         raise FileNotFoundError(f"Config not found: {config_path}")
    
#     with open(config_path, 'rb') as f:
#         config = pickle.load(f)
    
#     # Create model
#     autoencoder = DvectorAutoencoder(
#         input_dim=config['input_dim'],
#         hidden_dims=config['hidden_dims'],
#         dropout_rate=config.get('dropout_rate', 0.2),
#         norm_type=config.get('norm_type', 'batchnorm'),
#         use_residual=config.get('use_residual', False),
#         residual_scale_init=config.get('residual_scale_init', 0.5),
#     )
    
#     # Load weights
#     model_path = model_dir / 'final_model.pth'
#     if not model_path.exists():
#         model_path = model_dir / 'best_model.pth'
#         if not model_path.exists():
#             raise FileNotFoundError(f"Model not found in: {model_dir}")
    
#     # Convert device to string for comparison if it's a torch.device object
#     device_str = str(device) if isinstance(device, torch.device) else device
    
#     if device_str == 'cuda' and torch.cuda.is_available():
#         autoencoder.load_state_dict(torch.load(model_path))
#         autoencoder = autoencoder.to(device)
#     else:
#         autoencoder.load_state_dict(torch.load(model_path, map_location='cpu'))
#         autoencoder = autoencoder.to(device)
    
#     autoencoder.eval()
    
#     print(f"✓ Loaded autoencoder from: {model_dir}")
#     print(f"  Architecture: {config['hidden_dims']}")
#     print(f"  Bottleneck dimension: {config['hidden_dims'][len(config['hidden_dims'])//2]}")
    
#     return autoencoder, config


class VadSETAEDataset(Dataset):
    """VadSET dataset with optional autoencoder-compressed d-vectors.
    
    Supports pre-computed scores (original) or on-the-fly recomputation:
    - If recompute_scores=False: Uses pre-computed scores from scores.scp
    - If recompute_scores=True: Extracts frame-level d-vectors and computes scores with autoencoder
    
    Supports two autoencoder modes:
    - Bottleneck only (use_ae_reconstruction=False): 256->64 compression
      Enrolled d-vectors are compressed and compared in compressed space
    - Full reconstruction (use_ae_reconstruction=True): 256->64->256 denoising
          Stream d-vectors are reconstructed and compared with enrolled d-vectors
          Enrolled vectors can be independently toggled for scoring anchor and model input
    """

    def __init__(
        self,
        root_dir,
        embed_path,
        score_type,
        autoencoder=None,
        use_autoencoder=True,
        use_ae_reconstruction=False,
        max_utterances=None,
        recompute_scores=False,
        audio_root=None,
        transform_enrolled_in_reconstruction=False,
        transform_enrolled_vadinput_in_reconstruction=False,
        collect_similarity_scores=False,
        similarity_score_sample_step=1,
        similarity_score_max_items=None,
    ):
        self.root_dir = root_dir
        self.embed_path = embed_path
        self.score_type = score_type
        self.use_autoencoder = use_autoencoder
        self.use_ae_reconstruction = use_ae_reconstruction
        self.transform_enrolled_in_reconstruction = transform_enrolled_in_reconstruction
        self.transform_enrolled_vadinput_in_reconstruction = transform_enrolled_vadinput_in_reconstruction
        self.max_utterances = max_utterances
        self.recompute_scores = recompute_scores
        self.audio_root = audio_root
        self.collect_similarity_scores = bool(collect_similarity_scores)
        self.similarity_score_sample_step = max(1, int(similarity_score_sample_step))
        self.similarity_score_max_items = similarity_score_max_items
        self.similarity_scores = {}
        self.similarity_scores_by_class = {0: [], 1: [], 2: []}
        self.similarity_score_num_utts = 0
        
        # Optional forced-inference controls (set externally by eval scripts)
        self.forced_inference_target = None
        self.original_targets = None
        self.forced_relabel_keys = set()
        
        # Determine device (use GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move autoencoder to GPU
        if autoencoder is not None:
            self.autoencoder = autoencoder.to(self.device)
        else:
            self.autoencoder = None

        # Initialize d-vector encoder if recomputing scores
        if self.recompute_scores:
            print("Initializing VoiceEncoderMod for on-the-fly score computation...")
            self.dvector_encoder = VoiceEncoderMod()
            self.dvector_encoder.eval()
            self.dvector_encoder = self.dvector_encoder.to(self.device)  # Move to GPU
            print(f"  VoiceEncoderMod loaded on: {self.device}")
            # Resemblyzer parameters (from extract_features.py)
            self.rate = 2.5
            self.samples_per_frame = 160
            self.frame_step = int(np.round((16000 / self.rate) / self.samples_per_frame))
            if audio_root is None:
                # Try to auto-detect audio path from wav.scp
                wav_scp_path = f'{self.root_dir}/wav.scp'
                if os.path.exists(wav_scp_path):
                    self.wavs = kaldiio.load_scp(wav_scp_path)
                    print(f"  Loaded audio from: {wav_scp_path}")
                else:
                    raise ValueError(f"recompute_scores=True requires audio_root or {wav_scp_path}")
            else:
                self.wavs = kaldiio.load_scp(f'{audio_root}/wav.scp')
                print(f"  Loaded audio from: {audio_root}/wav.scp")
        else:
            self.dvector_encoder = None
            self.wavs = None

        # load the scp files...
        self.fbanks = kaldiio.load_scp(f'{self.root_dir}/fbanks.scp')
        if not self.recompute_scores:
            self.scores = kaldiio.load_scp(f'{self.root_dir}/scores.scp')
        else:
            self.scores = None
            print("  Scores will be recomputed on-the-fly")
        self.labels = kaldiio.load_scp(f'{self.root_dir}/labels.scp')
        self.keys = np.array(list(self.fbanks)) # get all the keys
        
        # Filter out keys with missing/corrupted audio files if recomputing scores
        if self.recompute_scores:
            valid_keys = []
            invalid_count = 0
            for key in self.keys:
                if key in self.wavs:
                    try:
                        # Try to load audio to verify it's valid
                        sr, audio = self.wavs[key]
                        if isinstance(audio, np.ndarray) and len(audio) > 0:
                            valid_keys.append(key)
                        else:
                            invalid_count += 1
                    except (RuntimeError, ValueError, KeyError) as e:
                        # Skip corrupted/missing audio files
                        invalid_count += 1
                else:
                    invalid_count += 1
            
            self.keys = np.array(valid_keys)
            if invalid_count > 0:
                print(f"  ⚠️  Skipped {invalid_count} utterances with missing/corrupted audio files")
                print(f"  ✓ {len(self.keys)} utterances with valid audio")
        
        # Limit number of utterances if specified
        if max_utterances is not None and max_utterances > 0:
            self.keys = sample_evenly_across_augmentations(self.keys, max_utterances)
        
        self.embed = kaldiio.load_scp(f'{self.embed_path}/dvectors.scp')

        # load the target speaker ids
        self.targets = {}
        with open(f'{self.root_dir}/targets.scp') as targets:
            for line in targets:
                (utt_id, target) = line.split()
                self.targets[utt_id] = target
        
        # Pre-process enrolled d-vectors if using autoencoder
        if self.use_autoencoder and self.autoencoder is not None:
            if self.use_ae_reconstruction:
                if self.transform_enrolled_in_reconstruction or self.transform_enrolled_vadinput_in_reconstruction:
                    # Optional mode: reconstruct enrolled vectors too (still 256-dim output).
                    print("Full reconstruction mode: reconstructing enrolled d-vectors (256-dim → 256-dim)...")
                    self.processed_embed = {}
                    self.autoencoder.eval()
                    with torch.no_grad():
                        for target, dvector in self.embed.items():
                            dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(self.device)
                            reconstructed = self.autoencoder(dvector_tensor)
                            self.processed_embed[target] = reconstructed.cpu().numpy().squeeze()
                    print(f"✓ Reconstructed {len(self.processed_embed)} enrolled d-vectors (256-dim)")
                    print("  Stream/slice d-vectors will be reconstructed on-the-fly for scoring")
                else:
                    # Default reconstruction mode: keep enrolled vectors unchanged.
                    # Only stream/slice vectors are reconstructed when recomputing scores.
                    self.processed_embed = None
                    print("Full reconstruction mode: keeping enrolled d-vectors original (256-dim)")
                    print("  Stream/slice d-vectors will be reconstructed on-the-fly for scoring")
            else:
                # Bottleneck mode: compress enrolled d-vectors to 64-dim
                print(f"Bottleneck mode: compressing enrolled d-vectors (256-dim → 64-dim)...")
                self.processed_embed = {}
                self.autoencoder.eval()
                with torch.no_grad():
                    for target, dvector in self.embed.items():
                        dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(self.device)
                        compressed = self.autoencoder.encode(dvector_tensor)  # Bottleneck only
                        self.processed_embed[target] = compressed.cpu().numpy().squeeze()
                print(f"✓ Compressed {len(self.processed_embed)} enrolled d-vectors (64-dim)")
        else:
            print(f"Using full 256-dim d-vectors (no compression)")
            self.processed_embed = None

    def cos(self, a, b):
        """Compute the cosine similarity of two vectors"""
        return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys[idx]
        target = self.targets[key]
        x = self.fbanks[key]  # (n_frames, 40)
        embed = self.embed[target]  # (256,)
        y = self.labels[key]

        # Forced-target mode for evaluation-only relabeling:
        # remove target class entirely by mapping 2 -> 1 for all utterances.
        forced_mode = (self.forced_inference_target is not None) and (str(self.forced_inference_target).strip() != '')
        if forced_mode:
            y = y.copy()
            y[y == 2] = 1

        # Get or compute scores
        if self.recompute_scores:
            # Extract frame-level d-vectors and compute scores on-the-fly
            # IMPORTANT: This must match extract_features.py score computation exactly!
            sr, audio = self.wavs[key]
            audio = audio.astype(np.float32) / 32768.0
            
            # Extract fbanks from audio (matching extract_features.py)
            fbanks_audio = librosa.feature.melspectrogram(
                y=audio, sr=16000, n_fft=400, hop_length=160, n_mels=40
            ).astype('float32').T[:-2]
            
            # Generate windowed fbanks for slice-based scoring (matching extract_features.py)
            # Use resemblyzer parameters: rate=2.5, min_coverage=0.5, frame_step based on rate
            rate = 2.5
            min_coverage = 0.5
            samples_per_frame = 160
            frame_step = int(np.round((16000 / rate) / samples_per_frame))
            
            wav = audio.copy()
            wav_slices, mel_slices = VoiceEncoder.compute_partial_slices(wav.size, rate, min_coverage)
            max_wave_length = wav_slices[-1].stop
            if max_wave_length >= wav.size:
                wav = np.pad(audio, (0, max_wave_length - wav.size), "constant")
            mels = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400,
                hop_length=160, n_mels=40).astype('float32').T
            fbanks_sliced = np.array([mels[s] for s in mel_slices])
            
            # Extract frame-level d-vectors using VoiceEncoderMod
            fbanks_tensor = torch.unsqueeze(torch.from_numpy(fbanks_audio), 0).to(self.device)
            fbanks_sliced_tensor = torch.from_numpy(fbanks_sliced).to(self.device)
            
            with torch.no_grad():
                embeds_stream, _ = self.dvector_encoder.forward_stream(fbanks_tensor, None)
                embeds_stream = embeds_stream.cpu().numpy().squeeze()
                
                # Extract windowed embeddings for slice-based scoring
                embeds_slices = self.dvector_encoder(fbanks_sliced_tensor).cpu().numpy()
            
            # Determine which d-vector to use for scoring based on autoencoder mode
            if self.use_autoencoder and self.autoencoder is not None:
                if self.use_ae_reconstruction:
                    # Full reconstruction: process stream d-vectors, compare with selected enrolled anchor
                    # Process each frame's d-vector through autoencoder
                    embeds_stream_tensor = torch.FloatTensor(embeds_stream).to(self.device)
                    embeds_slices_tensor = torch.FloatTensor(embeds_slices).to(self.device)
                    with torch.no_grad():
                        embeds_stream_reconstructed = self.autoencoder(embeds_stream_tensor).cpu().numpy()
                        embeds_slices_reconstructed = self.autoencoder(embeds_slices_tensor).cpu().numpy()
                    if self.transform_enrolled_in_reconstruction and self.processed_embed is not None:
                        enrolled_dvec = self.processed_embed[target]
                    else:
                        enrolled_dvec = embed
                    # Compute scores: reconstructed stream/slices vs original enrolled
                    scores_stream = np.array([self.cos(enrolled_dvec, frame_dvec) for frame_dvec in embeds_stream_reconstructed])
                    scores_slices = np.array([self.cos(enrolled_dvec, slice_dvec) for slice_dvec in embeds_slices_reconstructed])
                else:
                    # Bottleneck mode: compress both, compare in compressed space
                    # Compress stream d-vectors and slice d-vectors
                    embeds_stream_tensor = torch.FloatTensor(embeds_stream).to(self.device)
                    embeds_slices_tensor = torch.FloatTensor(embeds_slices).to(self.device)
                    with torch.no_grad():
                        embeds_stream_compressed = self.autoencoder.encode(embeds_stream_tensor).cpu().numpy()
                        embeds_slices_compressed = self.autoencoder.encode(embeds_slices_tensor).cpu().numpy()
                    # Get compressed enrolled (from processed_embed)
                    enrolled_dvec = self.processed_embed[target]
                    # Compute scores in compressed space
                    scores_stream = np.array([self.cos(enrolled_dvec, frame_dvec) for frame_dvec in embeds_stream_compressed])
                    scores_slices = np.array([self.cos(enrolled_dvec, slice_dvec) for slice_dvec in embeds_slices_compressed])
            else:
                # No autoencoder: use original d-vectors
                scores_stream = np.array([self.cos(embed, frame_dvec) for frame_dvec in embeds_stream])
                scores_slices = np.array([self.cos(embed, slice_dvec) for slice_dvec in embeds_slices])

            # Generate all three score types (matching extract_features.py exactly)
            n = len(x)  # Number of frames

            if self.collect_similarity_scores:
                max_items = self.similarity_score_max_items
                if max_items is None or self.similarity_score_num_utts < max_items:
                    scores_frame = scores_stream[:n]
                    indices = np.arange(scores_frame.shape[0])[:: self.similarity_score_sample_step]
                    sampled_scores = scores_frame[indices].astype(np.float32)
                    sampled_labels = y[:n][indices]

                    self.similarity_scores[key] = sampled_scores
                    for cls in (0, 1, 2):
                        class_scores = sampled_scores[sampled_labels == cls]
                        if class_scores.size:
                            self.similarity_scores_by_class[cls].append(class_scores)
                    self.similarity_score_num_utts += 1
            
            
            # Type 0: Baseline - frame-level scores
            scores_type0 = scores_stream[:n]
            
            # Type 1: Partially-constant - repeat slice scores using np.kron
            scores_kron = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
            if scores_slices.size > 1:
                scores_kron = np.append(scores_kron,
                    np.kron(scores_slices[1:], np.ones(frame_step, dtype='float32')))
            scores_type1 = scores_kron[:n]  # Trim to match frame count
            
            # Type 2: Linearly-interpolated
            scores_lin = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
            for i, s in enumerate(scores_slices[1:]):
                scores_lin = np.append(scores_lin,
                    np.linspace(scores_slices[i], s, frame_step, endpoint=False))
            scores_type2 = scores_lin[:n]  # Trim to match frame count
            
            # Select score based on score_type (matching extract_features.py)
            # legend: 0 -> baseline (stream), 1 -> partially-constant (kron), 2 -> linearly-interpolated
            if self.score_type == 0:
                scores = scores_type0
            elif self.score_type == 1:
                scores = scores_type1
            elif self.score_type == 2:
                scores = scores_type2
            else:
                raise ValueError(f"Invalid score_type: {self.score_type}. Must be 0, 1, or 2.")
        else:
            # Use pre-computed scores from scores.scp
            scores = self.scores[key][self.score_type, :]  # (n_frames,)

        # add the speaker verification scores array to the feature vector
        x = np.hstack((x, np.expand_dims(scores, 1)))  # (n_frames, 41)
        
        if self.use_autoencoder and self.autoencoder is not None:
            if self.use_ae_reconstruction:
                # Full reconstruction mode: default to original enrolled d-vector in model input.
                # Optional input transform is kept separate from scoring transform.
                if self.transform_enrolled_vadinput_in_reconstruction and self.processed_embed is not None:
                    enrolled_for_input = self.processed_embed[target]
                else:
                    enrolled_for_input = embed
                x = np.hstack((x, np.full((x.shape[0], 256), enrolled_for_input)))  # (n_frames, 297)
            else:
                # Bottleneck mode: use compressed enrolled d-vector (64-dim)
                enrolled_compressed = self.processed_embed[target]
                x = np.hstack((x, np.full((x.shape[0], 64), enrolled_compressed)))  # (n_frames, 105)
        else:
            # Use full 256-dim enrolled d-vector repeated for all frames (original behavior)
            x = np.hstack((x, np.full((x.shape[0], 256), embed)))  # (n_frames, 297)

        x = torch.from_numpy(x.copy()).float()
        y = torch.from_numpy(y.copy()).long()
        
        return x, y, key


if __name__ == '__main__':
    """ Model training with optional autoencoder-compressed d-vectors """

    # program arguments
    parser = ap.ArgumentParser(description="Train the VAD SET model with optional AE-compressed d-vectors.")
    parser.add_argument('--train_dir', type=str, default=DATA_TRAIN,
                        help='Training data directory')
    parser.add_argument('--test_dir', type=str, default=DATA_TEST,
                        help='Test data directory')
    parser.add_argument('--embed_path', type=str, default=EMBED_PATH,
                        help='Path to speaker embeddings directory')
    parser.add_argument('--ae_model_path', type=str, default=AE_MODEL_PATH,
                        help='Path to trained autoencoder model directory (e.g., src/AE_test/test_outputs/dvector_ae_many_main_w_other_singles)')
    parser.add_argument('--score_type', type=int, default=SCORE_TYPE,
                        help='Scoring method: 0=baseline, 1=PC, 2=LI')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--use_kaldi', action='store_true')
    parser.add_argument('--use_wpl', action='store_true')
    parser.add_argument('--nuse_fc', action='store_false')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--nsave_model', action='store_false')
    parser.add_argument('--use_autoencoder', action='store_true', default=USE_AUTOENCODER,
                        help='Use autoencoder to compress d-vectors (64-dim) instead of full 256-dim')
    parser.add_argument('--use_ae_reconstruction', action='store_true', default=USE_AE_RECONSTRUCTION,
                        help='Use full autoencoder reconstruction (enc+dec: 256->64->256) instead of bottleneck only (256->64)')
    parser.add_argument('--recompute_scores', action='store_true', default=RECOMPUTE_SCORES,
                        help='Recompute speaker verification scores using autoencoder-processed d-vectors instead of using pre-computed scores.scp')
    parser.add_argument('--audio_root', type=str, default=None,
                        help='Path to audio root directory (for wav.scp when recomputing scores)')
    parser.add_argument('--max_train_utterances', type=int, default=None,
                        help='Maximum number of training utterances to use (default: use all)')
    parser.add_argument('--max_test_utterances', type=int, default=None,
                        help='Maximum number of test utterances to use (default: use all)')
    args = parser.parse_args()
    
    # Update global variables from args
    DATA_TRAIN = args.train_dir
    DATA_TEST = args.test_dir
    EMBED_PATH = args.embed_path
    AE_MODEL_PATH = args.ae_model_path
    SCORE_TYPE = args.score_type
    MODEL_PATH = args.model_path
    USE_KALDI = args.use_kaldi
    USE_WPL = args.use_wpl
    SAVE_MODEL = args.nsave_model
    USE_AUTOENCODER = args.use_autoencoder
    USE_AE_RECONSTRUCTION = args.use_ae_reconstruction
    RECOMPUTE_SCORES = args.recompute_scores
    linear = args.linear
    
    # Update input_dim and dvector_dim based on USE_AUTOENCODER and USE_AE_RECONSTRUCTION
    if USE_AUTOENCODER:
        if USE_AE_RECONSTRUCTION:
            input_dim = 297  # 40 fbanks + 256 reconstructed d-vec + 1 score
            dvector_dim = 256
        else:
            input_dim = 105  # 40 fbanks + 64 compressed d-vec + 1 score
            dvector_dim = 64
    else:
        input_dim = 297  # 40 fbanks + 256 full d-vec + 1 score
        dvector_dim = 256
    
    # Load autoencoder if needed
    if USE_AUTOENCODER:
        print(f"\n🔧 Loading autoencoder model...")
        print(f"   Model path: {AE_MODEL_PATH}")
        autoencoder, ae_config = load_autoencoder(AE_MODEL_PATH, device)
        autoencoder.eval()
        encoded_dim = ae_config['hidden_dims'][len(ae_config['hidden_dims'])//2]
        print(f"\n✅ Autoencoder loaded successfully!")
        print(f"   Model directory: {AE_MODEL_PATH}")
        print(f"   Architecture: {ae_config['hidden_dims']}")
        print(f"   Encoder: 256-dim → {encoded_dim}-dim (bottleneck)")
        print(f"   Decoder: {encoded_dim}-dim → 256-dim")
        print(f"   Compression ratio: {256/encoded_dim:.1f}x")
        if 'main_speakers' in ae_config:
            print(f"   Main speakers: {ae_config['main_speakers']}")
        if 'best_val_loss' in ae_config:
            print(f"   Validation loss: {ae_config['best_val_loss']:.6f}")
        if 'test_cosine_similarity_mean' in ae_config:
            print(f"   Test cosine similarity: {ae_config['test_cosine_similarity_mean']:.4f}")
    else:
        autoencoder = None
        print(f"\n⚠️  Autoencoder disabled - using full 256-dim d-vectors")
    
    # Create datasets
    print(f"\n📂 Loading datasets...")
    print(f"   Train: {DATA_TRAIN}")
    print(f"   Test: {DATA_TEST}")
    print(f"   Embeddings: {EMBED_PATH}")
    print(f"   Score type: {SCORE_TYPE}")
    if RECOMPUTE_SCORES:
        print(f"   Score recomputation: ENABLED (using autoencoder-processed d-vectors)")
    else:
        print(f"   Score recomputation: DISABLED (using pre-computed scores.scp)")
    
    train_data = VadSETAEDataset(
        DATA_TRAIN, EMBED_PATH, SCORE_TYPE,
        autoencoder=autoencoder, use_autoencoder=USE_AUTOENCODER,
        use_ae_reconstruction=USE_AE_RECONSTRUCTION,
        max_utterances=args.max_train_utterances,
        recompute_scores=RECOMPUTE_SCORES,
        audio_root=args.audio_root
    )
    test_data = VadSETAEDataset(
        DATA_TEST, EMBED_PATH, SCORE_TYPE,
        autoencoder=autoencoder, use_autoencoder=USE_AUTOENCODER,
        use_ae_reconstruction=USE_AE_RECONSTRUCTION,
        max_utterances=args.max_test_utterances,
        recompute_scores=RECOMPUTE_SCORES,
        audio_root=args.audio_root
    )
    
    print(f"✅ Train samples: {len(train_data)}")
    print(f"✅ Test samples: {len(test_data)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate_with_metadata
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate_with_metadata
    )
    
    # Create model
    model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=args.nuse_fc, linear=linear).to(device)

    print(f"\n🏗️  VAD Model:")
    if USE_AUTOENCODER:
        if USE_AE_RECONSTRUCTION:
            print(f"  Input: {input_dim}-dim (40 fbanks + 256 reconstructed d-vec + 1 score)")
        else:
            print(f"  Input: {input_dim}-dim (40 fbanks + 64 compressed d-vec + 1 score)")
    else:
        print(f"  Input: {input_dim}-dim (40 fbanks + 256 full d-vec + 1 score)")
    print(f"  Hidden: {hidden_dim}-dim")
    print(f"  Layers: {num_layers}")
    print(f"  Output: {out_dim} classes")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    if USE_AUTOENCODER:
        if USE_AE_RECONSTRUCTION:
            print(f"\n💡 Mode: Full autoencoder reconstruction (256→64→256)")
            print(f"   Enrolled d-vectors stay original (256-dim)")
            print(f"   Stream d-vectors are reconstructed for score recomputation")
        else:
            print(f"\n💡 Mode: Bottleneck compression only (256→64)")
            print(f"   Enrolled d-vectors use compressed bottleneck representation")
    else:
        print(f"\n💡 Note: Using full 256-dim d-vectors for speaker verification")

    # Initialize training components
    if USE_WPL:
        criterion = WPL(WPL_WEIGHTS)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    else:
        scheduler = None
    
    softmax = nn.Softmax(dim=1)
    
    # Early stopping tracking
    best_metric = 0.0  # Best validation metric (mAP or accuracy)
    epochs_without_improvement = 0
    best_epoch = 0
    
    # Create checkpoint directory
    checkpoint_dir = MODEL_PATH.rpartition('.')[0] + '_checkpoints'
    if SAVE_MODEL and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Train!!! hype!!!
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Early stopping: enabled (patience={EARLY_STOPPING_PATIENCE}, metric={METRIC_FOR_BEST})")
    print(f"Periodic saving: every {SAVE_EVERY_N_EPOCHS} epoch(s)")
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        print(f"\n====== Starting epoch {epoch} ======")
        for batch, (x_padded, y_padded, x_lens, y_lens, keys) in enumerate(train_loader):
            try:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = model(x_padded.to(device), x_lens, None)

                # compute the loss
                loss = 0
                for j in range(out_padded.size(0)):
                    loss += criterion(out_padded[j][:y_lens[j]], y_padded[j][:y_lens[j]])

                loss /= out_padded.size(0)  # normalize by actual batch size
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if batch % 10 == 0:
                    print(f'Batch: {batch}, loss = {loss:.4f}')
                    
            except RuntimeError as e:
                if 'CUDNN_STATUS_EXECUTION_FAILED' in str(e) or 'out of memory' in str(e):
                    print(f"\n⚠️  CUDA error at batch {batch}: {e}")
                    print("Clearing CUDA cache and continuing...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

        if SCHEDULER and epoch < 2:
            scheduler.step() # learning rate adjust
            if epoch == 1:
                optimizer.param_groups[0]['lr'] = 5e-5
        if SCHEDULER and epoch == 7:
            optimizer.param_groups[0]['lr'] = 1e-5
        
        # Clear CUDA cache to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Test the model after each epoch
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            print("testing...")
            n_correct = 0
            n_samples = 0
            targets = []
            outputs = []
            for x_padded, y_padded, x_lens, y_lens, keys in test_loader:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = model(x_padded.to(device), x_lens, None)

                # value, index
                for j in range(out_padded.size(0)):
                    classes = torch.argmax(out_padded[j][:y_lens[j]], dim=1)
                    n_samples += y_lens[j]
                    n_correct += torch.sum(classes == y_padded[j][:y_lens[j]]).item()

                    # average precision
                    p = softmax(out_padded[j][:y_lens[j]])
                    outputs.append(p.cpu().numpy())
                    targets.append(y_padded[j][:y_lens[j]].cpu().numpy())

            acc = 100.0 * n_correct / n_samples
            print(f"accuracy = {acc:.2f}")

            # and run the AP
            targets = np.concatenate(targets)
            outputs = np.concatenate(outputs)
            targets_oh = np.eye(3)[targets]
            out_AP = average_precision_score(targets_oh, outputs, average=None)
            mAP = average_precision_score(targets_oh, outputs, average='micro')

            print(out_AP)
            print(f"mAP: {mAP}")

        # Determine current metric for early stopping
        current_metric = mAP if METRIC_FOR_BEST == 'mAP' else acc
        
        # Check for improvement
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            if SAVE_MODEL:
                # if necessary, create the destination path for the model...
                path_seg = MODEL_PATH.split('/')[:-1]
                if path_seg != []:
                    if not os.path.exists(MODEL_PATH.rpartition('/')[0]):
                        os.makedirs('/'.join(path_seg))
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"✅ New best {METRIC_FOR_BEST}: {best_metric:.4f} - Model saved!")
        else:
            epochs_without_improvement += 1
            print(f"⚠️  No improvement for {epochs_without_improvement} epoch(s) (best {METRIC_FOR_BEST}: {best_metric:.4f} at epoch {best_epoch})")
        
        # Periodic checkpoint saving
        if SAVE_MODEL and (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_mAP_{mAP:.4f}_acc_{acc:.2f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
                'accuracy': acc,
                'best_metric': best_metric,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Early stopping check
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            print(f"   Best {METRIC_FOR_BEST}: {best_metric:.4f} (epoch {best_epoch})")
            break

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n💾 Best model saved to: {MODEL_PATH}")
    print(f"📊 Best {METRIC_FOR_BEST}: {best_metric:.4f} (epoch {best_epoch})")
    print(f"📊 Final accuracy: {acc:.2f}%")
    print(f"📊 Final mAP: {mAP:.4f}")
    if SAVE_MODEL:
        print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    
    if USE_AUTOENCODER:
        if USE_AE_RECONSTRUCTION:
            print(f"\n💡 This model uses autoencoder reconstruction:")
            print(f"   - Mode: Full enc-dec (256→64→256)")
            print(f"   - Enrolled d-vectors: Original (unchanged)")
            print(f"   - Stream d-vectors: Reconstructed for scoring")
            print(f"   - Purpose: Denoising and speaker extraction")
            print(f"   - Input size: {input_dim}-dim (same as original)")
        else:
            print(f"\n💡 This model uses compressed d-vectors:")
            print(f"   - Mode: Bottleneck only (256→64)")
            print(f"   - Enrolled d-vectors: 256-dim → 64-dim (autoencoder)")
            print(f"   - Purpose: Dimensionality reduction")
            print(f"   - Input size reduced: 297-dim → {input_dim}-dim ({297/input_dim:.2f}x smaller)")
    else:
        print(f"\n💡 This model uses full 256-dim d-vectors:")
        print(f"   - Frame d-vectors: 256-dim (no compression)")
        print(f"   - Enrolled d-vector: 256-dim (no compression)")
        print(f"   - Scores: Cosine similarity of full d-vectors")
        print(f"   - Input size: {input_dim}-dim")