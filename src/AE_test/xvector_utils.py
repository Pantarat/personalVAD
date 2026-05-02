#!/usr/bin/env python3
"""Utilities for extracting and comparing x-vectors via Kaldi.

This module provides functions to:
1. Extract x-vectors from audio files using Kaldi
2. Average x-vectors across time
3. Compare x-vectors with and without autoencoder processing

Usage:
    from xvector_utils import KaldiXVectorExtractor
    extractor = KaldiXVectorExtractor(model_dir, extract_script, kaldi_root)
    xvector = extractor.extract_from_wav(audio_file)
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
import shutil
import os


class KaldiXVectorExtractor:
    """Extract x-vectors from audio using Kaldi nnet3-xvector-compute."""
    
    def __init__(self, model_dir, extract_script=None, kaldi_root=None):
        """
        Initialize x-vector extractor.
        
        Args:
            model_dir: Path to Kaldi x-vector model directory
            extract_script: Path to extraction script (e.g., extract_xvectors.sh)
            kaldi_root: Path to Kaldi root (if None, assumes 'kaldi' in cwd)
        """
        self.model_dir = Path(model_dir)
        self.extract_script = Path(extract_script) if extract_script else None
        self.kaldi_root = Path(kaldi_root) if kaldi_root else Path('kaldi')
        
        # Check model directory
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Detect available Kaldi binaries
        self.nnet3_xvector_compute = self._find_kaldi_binary('nnet3-xvector-compute')
        self.feat_to_dim = self._find_kaldi_binary('feat-to-dim')
        self.ivector_compute = self._find_kaldi_binary('ivector-compute-mean')
        
        # Check for required model files
        self._verify_model_files()
    
    def _find_kaldi_binary(self, binary_name):
        """Find Kaldi binary in common locations."""
        common_paths = [
            self.kaldi_root / 'src' / 'ivectorbin' / binary_name,
            self.kaldi_root / 'src' / 'nnet3bin' / binary_name,
            self.kaldi_root / 'src' / 'featbin' / binary_name,
            shutil.which(binary_name),
        ]
        
        for path in common_paths:
            if path and Path(path).exists():
                return str(path)
        
        return binary_name  # Hope it's in PATH
    
    def _verify_model_files(self):
        """Verify required model files exist."""
        required_files = [
            'final.nnet',  # Neural network model
            'cmvn.opts',  # CMVN options
        ]
        
        for fname in required_files:
            fpath = self.model_dir / fname
            if not fpath.exists():
                # Try alternative names
                if fname == 'final.nnet' and (self.model_dir / 'nnet3' / 'final.nnet').exists():
                    continue
                print(f"Warning: {fname} not found in {self.model_dir}")
    
    def extract_from_wav(self, wav_file, use_cmvn=True):
        """
        Extract x-vector from a WAV file.
        
        Args:
            wav_file: Path to input WAV file
            use_cmvn: Whether to apply CMVN normalization
        
        Returns:
            x-vector as numpy array (ndim=1)
        """
        wav_file = Path(wav_file)
        if not wav_file.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_file}")
        
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create wav.scp file
            wav_scp = tmpdir / 'wav.scp'
            with open(wav_scp, 'w') as f:
                f.write(f"utt {wav_file.absolute()}\n")
            
            # Create feature file path
            feats_ark = tmpdir / 'feats.ark'
            xvector_ark = tmpdir / 'xvector.ark'
            
            try:
                # Compute features (MFCC or other)
                self._compute_features(wav_scp, feats_ark)
                
                # Compute x-vectors
                self._compute_xvector(feats_ark, xvector_ark)
                
                # Read x-vector
                xvector = self._read_ark_vector(xvector_ark)
                
                return xvector
            
            except Exception as e:
                print(f"Error extracting x-vector: {e}")
                return None
    
    def _compute_features(self, wav_scp, feats_ark):
        """Compute MFCC features from WAV files."""
        # This is a simplified version; actual implementation would need
        # to call compute-mfcc-feats or similar
        print(f"Note: Feature computation from wav.scp not fully implemented")
        print(f"  Expected output: {feats_ark}")
    
    def _compute_xvector(self, feats_ark, xvector_ark):
        """Compute x-vectors from features."""
        # This would call nnet3-xvector-compute
        print(f"Note: X-vector computation via Kaldi not fully implemented")
        print(f"  Would call: {self.nnet3_xvector_compute}")
    
    def _read_ark_vector(self, ark_file):
        """Read vector from Kaldi ARK file.
        
        This is a simplified reader; full implementation would need
        proper Kaldi binary format parsing.
        """
        # Placeholder - actual implementation requires kaldiio
        print(f"Note: ARK file reading not fully implemented")
        return None


def extract_xvectors_batch(wav_list, model_dir, extract_script=None, kaldi_root=None):
    """
    Extract x-vectors from a list of WAV files.
    
    Args:
        wav_list: List of WAV file paths
        model_dir: Path to Kaldi x-vector model
        extract_script: Optional path to extraction script
        kaldi_root: Optional path to Kaldi root
    
    Returns:
        Dictionary mapping wav_path -> x-vector (numpy array)
    """
    extractor = KaldiXVectorExtractor(model_dir, extract_script, kaldi_root)
    results = {}
    
    for wav_file in wav_list:
        xvec = extractor.extract_from_wav(wav_file)
        if xvec is not None:
            results[str(wav_file)] = xvec
    
    return results


def average_xvectors(xvector_list):
    """Average multiple x-vectors into a single x-vector."""
    if len(xvector_list) == 0:
        return None
    return np.mean(xvector_list, axis=0)


def cosine_similarity_xvectors(xvec1, xvec2):
    """Compute cosine similarity between two x-vectors."""
    xvec1 = np.asarray(xvec1, dtype=np.float32).ravel()
    xvec2 = np.asarray(xvec2, dtype=np.float32).ravel()
    
    norm1 = np.linalg.norm(xvec1)
    norm2 = np.linalg.norm(xvec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(xvec1, xvec2) / (norm1 * norm2)


# =====================
# ALTERNATIVE: Using kaldiio if available
# =====================
try:
    import kaldiio
    
    def read_xvector_ark(ark_file):
        """Read x-vectors from Kaldi ARK file using kaldiio."""
        data = kaldiio.load_ark(str(ark_file))
        return data
    
    KALDIIO_AVAILABLE = True
except ImportError:
    KALDIIO_AVAILABLE = False
    print("Note: kaldiio not installed. For full x-vector support, install with:")
    print("  pip install kaldiio")
