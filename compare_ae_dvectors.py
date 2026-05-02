"""
Compare d-vectors before and after autoencoder processing.

This script:
1. Loads enrolled d-vectors and some stream d-vectors
2. Processes them through the autoencoder
3. Visualizes the comparison in the same graph
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vad_set_ae import load_autoencoder
import kaldiio

# Configuration
IDENTITY_AE_PATH = 'src/AE_test/test_outputs/dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000'
EMBED_PATH = 'data/embeddings'
TEST_DIR = 'data/84_ov_test_ov50pct_main84_babble_100'
NUM_SAMPLES = 10  # Number of d-vectors to sample and compare

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def plot_comparison(original_dvecs, reconstructed_dvecs, title="D-Vector Comparison"):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Cosine Similarity Distribution
    ax1 = plt.subplot(2, 3, 1)
    cosine_sims = [compute_cosine_similarity(orig, recon) 
                   for orig, recon in zip(original_dvecs, reconstructed_dvecs)]
    ax1.hist(cosine_sims, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(cosine_sims), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(cosine_sims):.6f}')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Cosine Similarity Distribution\n(Original vs Reconstructed)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE Distribution
    ax2 = plt.subplot(2, 3, 2)
    mse_values = [np.mean((orig - recon) ** 2) 
                  for orig, recon in zip(original_dvecs, reconstructed_dvecs)]
    ax2.hist(mse_values, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.mean(mse_values), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(mse_values):.8f}')
    ax2.set_xlabel('Mean Squared Error', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('MSE Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. L2 Norm Comparison
    ax3 = plt.subplot(2, 3, 3)
    orig_norms = [np.linalg.norm(d) for d in original_dvecs]
    recon_norms = [np.linalg.norm(d) for d in reconstructed_dvecs]
    x_pos = np.arange(len(orig_norms))
    width = 0.35
    ax3.bar(x_pos - width/2, orig_norms, width, label='Original', alpha=0.8)
    ax3.bar(x_pos + width/2, recon_norms, width, label='Reconstructed', alpha=0.8)
    ax3.set_xlabel('D-Vector Index', fontsize=12)
    ax3.set_ylabel('L2 Norm', fontsize=12)
    ax3.set_title('L2 Norm Comparison', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Dimension-wise Error (first d-vector as example)
    ax4 = plt.subplot(2, 3, 4)
    if len(original_dvecs) > 0:
        dim_errors = np.abs(original_dvecs[0] - reconstructed_dvecs[0])
        ax4.plot(dim_errors, linewidth=1, alpha=0.7)
        ax4.fill_between(range(len(dim_errors)), dim_errors, alpha=0.3)
        ax4.set_xlabel('Dimension', fontsize=12)
        ax4.set_ylabel('Absolute Error', fontsize=12)
        ax4.set_title(f'Dimension-wise Error (Sample 0)\nMax: {np.max(dim_errors):.6f}, Mean: {np.mean(dim_errors):.6f}', 
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Scatter: Original vs Reconstructed (all dimensions)
    ax5 = plt.subplot(2, 3, 5)
    all_orig = np.concatenate(original_dvecs)
    all_recon = np.concatenate(reconstructed_dvecs)
    ax5.scatter(all_orig, all_recon, alpha=0.3, s=1)
    # Add diagonal line (perfect reconstruction)
    min_val, max_val = min(all_orig.min(), all_recon.min()), max(all_orig.max(), all_recon.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Reconstruction')
    ax5.set_xlabel('Original Value', fontsize=12)
    ax5.set_ylabel('Reconstructed Value', fontsize=12)
    ax5.set_title('Value Correspondence\n(All Dimensions, All Samples)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # 6. Per-sample Cosine Similarity
    ax6 = plt.subplot(2, 3, 6)
    sample_indices = range(len(cosine_sims))
    colors = ['green' if cs > 0.999 else 'orange' if cs > 0.99 else 'red' for cs in cosine_sims]
    ax6.bar(sample_indices, cosine_sims, color=colors, alpha=0.7, edgecolor='black')
    ax6.axhline(0.999, color='green', linestyle='--', linewidth=1, label='Excellent (>0.999)')
    ax6.axhline(0.99, color='orange', linestyle='--', linewidth=1, label='Good (>0.99)')
    ax6.set_xlabel('Sample Index', fontsize=12)
    ax6.set_ylabel('Cosine Similarity', fontsize=12)
    ax6.set_title('Per-Sample Cosine Similarity', fontsize=14, fontweight='bold')
    ax6.set_ylim([min(0.95, min(cosine_sims) - 0.01), 1.0])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig, cosine_sims, mse_values


def main():
    print("=" * 80)
    print("D-VECTOR COMPARISON: ORIGINAL vs AUTOENCODER-PROCESSED")
    print("=" * 80)
    
    # Load autoencoder
    print(f"\n1️⃣  Loading autoencoder...")
    print(f"   Path: {IDENTITY_AE_PATH}")
    
    autoencoder, ae_config = load_autoencoder(IDENTITY_AE_PATH, device)
    autoencoder.eval()
    autoencoder = autoencoder.to(device)
    
    print(f"   ✅ Loaded successfully")
    print(f"   Architecture: {ae_config['hidden_dims']}")
    
    if 'training_scheme' in ae_config:
        print(f"   ⚠️  Training scheme: {ae_config['training_scheme']} (EXTRACTION model)")
    else:
        print(f"   ✓ No training scheme (IDENTITY model)")
    
    if 'test_cosine_similarity_mean' in ae_config:
        print(f"   Test cos sim (training): {ae_config['test_cosine_similarity_mean']:.6f}")
    
    # Load enrolled d-vectors
    print(f"\n2️⃣  Loading enrolled d-vectors...")
    print(f"   Path: {EMBED_PATH}")
    
    dvectors_scp = kaldiio.load_scp(f'{EMBED_PATH}/dvectors.scp')
    all_speakers = list(dvectors_scp.keys())
    
    # Sample some d-vectors
    sampled_speakers = all_speakers[:NUM_SAMPLES] if len(all_speakers) >= NUM_SAMPLES else all_speakers
    print(f"   Total speakers: {len(all_speakers)}")
    print(f"   Sampling: {len(sampled_speakers)} speakers")
    print(f"   Speakers: {sampled_speakers}")
    
    # Collect original and reconstructed d-vectors
    original_dvecs = []
    reconstructed_dvecs = []
    
    print(f"\n3️⃣  Processing d-vectors through autoencoder...")
    
    with torch.no_grad():
        for speaker in sampled_speakers:
            original = dvectors_scp[speaker]
            original_tensor = torch.FloatTensor(original).unsqueeze(0).to(device)
            
            # Full reconstruction
            reconstructed_tensor = autoencoder(original_tensor)
            reconstructed = reconstructed_tensor.cpu().numpy().squeeze()
            
            original_dvecs.append(original)
            reconstructed_dvecs.append(reconstructed)
            
            # Compute metrics for this sample
            cos_sim = compute_cosine_similarity(original, reconstructed)
            mse = np.mean((original - reconstructed) ** 2)
            
            print(f"   {speaker}: cos_sim={cos_sim:.6f}, MSE={mse:.8f}")
    
    # Compute statistics
    print(f"\n4️⃣  Computing statistics...")
    
    all_cosine_sims = [compute_cosine_similarity(orig, recon) 
                       for orig, recon in zip(original_dvecs, reconstructed_dvecs)]
    all_mse = [np.mean((orig - recon) ** 2) 
               for orig, recon in zip(original_dvecs, reconstructed_dvecs)]
    
    print(f"\n   📊 Summary Statistics:")
    print(f"      Cosine Similarity:")
    print(f"         Mean: {np.mean(all_cosine_sims):.8f}")
    print(f"         Std:  {np.std(all_cosine_sims):.8f}")
    print(f"         Min:  {np.min(all_cosine_sims):.8f}")
    print(f"         Max:  {np.max(all_cosine_sims):.8f}")
    
    print(f"\n      Mean Squared Error:")
    print(f"         Mean: {np.mean(all_mse):.10f}")
    print(f"         Std:  {np.std(all_mse):.10f}")
    print(f"         Min:  {np.min(all_mse):.10f}")
    print(f"         Max:  {np.max(all_mse):.10f}")
    
    # Interpretation
    print(f"\n   🔍 Interpretation:")
    avg_cos_sim = np.mean(all_cosine_sims)
    if avg_cos_sim > 0.9999:
        print(f"      ✅ EXCELLENT: Near-perfect identity mapping (cos_sim > 0.9999)")
        print(f"         Autoencoder is almost perfectly preserving d-vectors")
    elif avg_cos_sim > 0.999:
        print(f"      ✅ VERY GOOD: High-quality identity mapping (cos_sim > 0.999)")
        print(f"         Autoencoder is preserving d-vectors very well")
    elif avg_cos_sim > 0.99:
        print(f"      ✓ GOOD: Decent identity mapping (cos_sim > 0.99)")
        print(f"         Some transformation is occurring")
    elif avg_cos_sim > 0.95:
        print(f"      ⚠️  MODERATE: Significant transformation (cos_sim > 0.95)")
        print(f"         Autoencoder is changing d-vectors noticeably")
    else:
        print(f"      ❌ POOR: Major transformation (cos_sim < 0.95)")
        print(f"         This is NOT behaving like an identity mapping!")
    
    # Create visualization
    print(f"\n5️⃣  Creating visualization...")
    
    fig, cosine_sims, mse_values = plot_comparison(
        original_dvecs, 
        reconstructed_dvecs,
        title=f"D-Vector Comparison: Identity Autoencoder\n" +
              f"Mean Cosine Similarity: {np.mean(all_cosine_sims):.6f} | Mean MSE: {np.mean(all_mse):.8f}"
    )
    
    # Save figure
    output_path = 'dvector_comparison_identity_ae.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved visualization to: {output_path}")
    
    # Show plot
    plt.show()
    
    # Final recommendation
    print(f"\n" + "=" * 80)
    print(f"📝 CONCLUSION:")
    print(f"=" * 80)
    
    if avg_cos_sim > 0.999:
        print(f"\n✅ The autoencoder is functioning as an identity mapping.")
        print(f"   Expected behavior: Evaluation results should be very similar to NO_AE baseline.")
        print(f"   Difference should be < 1-2% in accuracy.")
    else:
        print(f"\n⚠️  The autoencoder is transforming d-vectors significantly!")
        print(f"   This is NOT a true identity mapping.")
        print(f"   Possible causes:")
        print(f"   - Wrong model loaded (extraction instead of identity)")
        print(f"   - Identity training did not converge properly")
        print(f"   - Training data distribution mismatch")
        print(f"\n   Recommendation: Check model training logs and config")
    
    print(f"\n" + "=" * 80)


if __name__ == '__main__':
    main()
