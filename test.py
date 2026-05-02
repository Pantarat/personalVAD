# """
# t-SNE visualization comparing original vs autoencoder-processed d-vectors from test set.
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import pickle
# from pathlib import Path
# import sys
# import os

# # Add paths
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'AE_test'))

# # Configuration
# IDENTITY_AE_PATH = 'src/AE_test/test_outputs/dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000'
# TEST_DIR = 'data/84_ov_test_ov50pct_main84_babble_100'
# NUM_FILES = 5
# NUM_FRAMES_PER_FILE = 20
# PERPLEXITY = 30
# RANDOM_STATE = 42

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def load_autoencoder_minimal(model_dir, device):
#     """Load autoencoder"""
#     from train_dvector_autoencoder import DvectorAutoencoder
    
#     model_dir = Path(model_dir)
#     if not model_dir.is_absolute():
#         model_dir = Path(__file__).parent / model_dir
    
#     config_path = model_dir / 'config.pkl'
#     with open(config_path, 'rb') as f:
#         config = pickle.load(f)
    
#     autoencoder = DvectorAutoencoder(
#         input_dim=config['input_dim'],
#         hidden_dims=config['hidden_dims'],
#         dropout_rate=config['dropout_rate']
#     )
    
#     model_path = model_dir / 'final_model.pth'
#     if not model_path.exists():
#         model_path = model_dir / 'best_model.pth'
    
#     device_str = str(device) if isinstance(device, torch.device) else device
#     if device_str == 'cuda' and torch.cuda.is_available():
#         autoencoder.load_state_dict(torch.load(model_path))
#     else:
#         autoencoder.load_state_dict(torch.load(model_path, map_location='cpu'))
    
#     autoencoder = autoencoder.to(device)
#     autoencoder.eval()
    
#     return autoencoder, config


# def extract_stream_dvectors(test_dir, num_files=5, num_frames_per_file=20, device='cpu'):
#     """Extract frame-level d-vectors from test set"""
#     import kaldiio
#     import librosa
#     from resemblyzer_mod import VoiceEncoderMod
    
#     print(f"   Initializing VoiceEncoderMod...")
#     dvector_encoder = VoiceEncoderMod()
#     dvector_encoder.eval()
#     dvector_encoder = dvector_encoder.to(device)
    
#     wav_scp_path = f'{test_dir}/wav.scp'
#     wavs = kaldiio.load_scp(wav_scp_path)
#     all_keys = list(wavs.keys())
    
#     print(f"   Total files: {len(all_keys)}")
#     sampled_keys = all_keys[:num_files]
#     print(f"   Sampling: {len(sampled_keys)} files")
    
#     all_dvectors = []
#     all_labels = []
    
#     with torch.no_grad():
#         for key in sampled_keys:
#             try:
#                 sr, audio = wavs[key]
#                 audio = audio.astype(np.float32) / 32768.0
                
#                 fbanks_audio = librosa.feature.melspectrogram(
#                     y=audio, sr=16000, n_fft=400, hop_length=160, n_mels=40
#                 ).astype('float32').T[:-2]
                
#                 fbanks_tensor = torch.unsqueeze(torch.from_numpy(fbanks_audio), 0).to(device)
#                 embeds_stream, _ = dvector_encoder.forward_stream(fbanks_tensor, None)
#                 embeds_stream = embeds_stream.cpu().numpy().squeeze()
                
#                 total_frames = len(embeds_stream)
#                 if total_frames <= num_frames_per_file:
#                     frame_indices = range(total_frames)
#                 else:
#                     step = total_frames // num_frames_per_file
#                     frame_indices = range(0, total_frames, step)[:num_frames_per_file]
                
#                 for frame_idx in frame_indices:
#                     all_dvectors.append(embeds_stream[frame_idx])
#                     all_labels.append(f"{key}_f{frame_idx}")
                
#                 print(f"      {key}: {len(frame_indices)} frames")
#             except Exception as e:
#                 print(f"      Error {key}: {e}")
    
#     return np.array(all_dvectors), all_labels


# def compute_cosine_similarity(a, b):
#     """Compute cosine similarity"""
#     a = a.flatten()
#     b = b.flatten()
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# def main():
#     print("=" * 80)
#     print("t-SNE: Stream D-Vectors (Test Set) - Original vs Reconstructed")
#     print("=" * 80)
    
#     # Load autoencoder
#     print(f"\n1️⃣  Loading autoencoder: {IDENTITY_AE_PATH}")
#     autoencoder, ae_config = load_autoencoder_minimal(IDENTITY_AE_PATH, device)
#     print(f"   ✅ Loaded: {ae_config['hidden_dims']}")
#     if 'test_cosine_similarity_mean' in ae_config:
#         print(f"   Training cos_sim: {ae_config['test_cosine_similarity_mean']:.6f}")
    
#     # Extract d-vectors
#     print(f"\n2️⃣  Extracting stream d-vectors from: {TEST_DIR}")
#     stream_dvecs, frame_labels = extract_stream_dvectors(
#         TEST_DIR, NUM_FILES, NUM_FRAMES_PER_FILE, device
#     )
#     print(f"   ✅ Extracted {len(stream_dvecs)} d-vectors")
    
#     # Process through autoencoder
#     print(f"\n3️⃣  Processing through autoencoder...")
#     original_dvecs = stream_dvecs
#     reconstructed_dvecs = []
    
#     with torch.no_grad():
#         batch_size = 32
#         for i in range(0, len(original_dvecs), batch_size):
#             batch = original_dvecs[i:i+batch_size]
#             batch_tensor = torch.FloatTensor(batch).to(device)
#             reconstructed_batch = autoencoder(batch_tensor)
#             reconstructed_dvecs.append(reconstructed_batch.cpu().numpy())
#         reconstructed_dvecs = np.vstack(reconstructed_dvecs)
    
#     print(f"   ✅ Processed {len(reconstructed_dvecs)} d-vectors")
    
#     # Compute similarities
#     all_cosine_sims = [compute_cosine_similarity(orig, recon) 
#                        for orig, recon in zip(original_dvecs, reconstructed_dvecs)]
    
#     print(f"\n   📊 Cosine Similarity:")
#     print(f"      Mean: {np.mean(all_cosine_sims):.8f}")
#     print(f"      Std:  {np.std(all_cosine_sims):.8f}")
#     print(f"      Min:  {np.min(all_cosine_sims):.8f}")
#     print(f"      Max:  {np.max(all_cosine_sims):.8f}")
    
#     # Compute t-SNE
#     print(f"\n4️⃣  Computing t-SNE...")
#     all_dvecs = np.vstack([original_dvecs, reconstructed_dvecs])
#     tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=RANDOM_STATE, 
#                 n_iter=1000, verbose=1)
#     embeddings_2d = tsne.fit_transform(all_dvecs)
    
#     original_embeddings = embeddings_2d[:len(original_dvecs)]
#     reconstructed_embeddings = embeddings_2d[len(original_dvecs):]
    
#     # Visualize
#     print(f"\n5️⃣  Creating visualization...")
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
#     # Colors by file
#     num_colors = min(20, NUM_FILES)
#     colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
#     file_colors = []
#     for label in frame_labels:
#         file_part = label.split('_f')[0]
#         file_idx = hash(file_part) % num_colors
#         file_colors.append(colors[file_idx])
    
#     # Plot 1
#     for i in range(len(original_embeddings)):
#         ax1.scatter(original_embeddings[i, 0], original_embeddings[i, 1], 
#                    c=[file_colors[i]], s=80, marker='o', alpha=0.7, 
#                    edgecolors='black', linewidths=1)
#         ax1.scatter(reconstructed_embeddings[i, 0], reconstructed_embeddings[i, 1], 
#                    c=[file_colors[i]], s=80, marker='o', alpha=0.4, 
#                    edgecolors='black', linewidths=1.5, facecolors='none')
#         ax1.plot([original_embeddings[i, 0], reconstructed_embeddings[i, 0]], 
#                 [original_embeddings[i, 1], reconstructed_embeddings[i, 1]], 
#                 c=file_colors[i], alpha=0.2, linewidth=0.8, linestyle='--')
    
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
#                markersize=10, label='Original', markeredgecolor='black'),
#         Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
#                markersize=10, label='Reconstructed', markeredgecolor='black', markeredgewidth=1.5)
#     ]
#     ax1.legend(handles=legend_elements, loc='best', fontsize=12)
#     ax1.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
#     ax1.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
#     ax1.set_title(f'Stream D-Vectors: Original (Filled) vs Reconstructed (Hollow)\nCos Sim: {np.mean(all_cosine_sims):.6f}', 
#                   fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2
#     ax2.scatter(original_embeddings[:, 0], original_embeddings[:, 1], 
#                c='blue', s=150, marker='o', alpha=0.6, 
#                edgecolors='darkblue', linewidths=1.5, label='Original')
#     ax2.scatter(reconstructed_embeddings[:, 0], reconstructed_embeddings[:, 1], 
#                c='red', s=150, marker='^', alpha=0.6, 
#                edgecolors='darkred', linewidths=1.5, label='Reconstructed')
#     for i in range(len(original_embeddings)):
#         ax2.plot([original_embeddings[i, 0], reconstructed_embeddings[i, 0]], 
#                 [original_embeddings[i, 1], reconstructed_embeddings[i, 1]], 
#                 c='gray', alpha=0.15, linewidth=0.6, linestyle=':')
    
#     ax2.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
#     ax2.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
#     ax2.set_title('Group Comparison\nBlue=Original, Red=Reconstructed', 
#                   fontsize=14, fontweight='bold')
#     ax2.legend(loc='best', fontsize=12)
#     ax2.grid(True, alpha=0.3)
    
#     plt.suptitle(f'Identity AE: Stream D-Vectors (Test Set with Babble)\n' + 
#                  f'{len(original_dvecs)} Frames | Cos Sim: {np.mean(all_cosine_sims):.6f} ± {np.std(all_cosine_sims):.6f}',
#                  fontsize=16, fontweight='bold', y=0.98)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
    
#     output_path = 'tsne_ae_comparison_stream.png'
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     print(f"   ✅ Saved: {output_path}")
#     plt.show()
    
#     # Interpretation
#     print(f"\n" + "=" * 80)
#     avg_cos_sim = np.mean(all_cosine_sims)
#     if avg_cos_sim > 0.999:
#         print(f"✅ EXCELLENT identity mapping (>{0.999})")
#         print(f"   → Identity AE should match NO_AE baseline")
#     elif avg_cos_sim > 0.99:
#         print(f"✓ GOOD identity mapping (>{0.99})")
#     else:
#         print(f"⚠️  NOT identity-like (cos_sim={avg_cos_sim:.6f})")
#         print(f"   → Results will differ from NO_AE baseline")
#     print("=" * 80)


# if __name__ == '__main__':
#     main()

import torch

a = torch.tensor([[0.0083, 0.0587, 0.1158, 0.0000, 0.0000, 0.0712, 0.0000, 0.0000, 0.0615,
         0.0373, 0.1248, 0.2438, 0.0287, 0.0000, 0.0008, 0.0074, 0.1575, 0.0131,
         0.0043, 0.0381, 0.0287, 0.0898, 0.0000, 0.0000, 0.0000, 0.1687, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0417, 0.0236, 0.1439, 0.0000, 0.0989, 0.0045,
         0.0685, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0026,
         0.0407, 0.0000, 0.0042, 0.0000, 0.0153, 0.0000, 0.0141, 0.0000, 0.0000,
         0.0000, 0.0000, 0.1091, 0.0462, 0.1326, 0.0046, 0.1909, 0.0000, 0.0000,
         0.0000, 0.0067, 0.0000, 0.0726, 0.1508, 0.0000, 0.0000, 0.0373, 0.0007,
         0.0000, 0.0253, 0.0000, 0.0000, 0.0000, 0.0000, 0.0004, 0.1269, 0.0000,
         0.0000, 0.0703, 0.0227, 0.0000, 0.0000, 0.0000, 0.1512, 0.0000, 0.0401,
         0.0059, 0.0000, 0.0000, 0.0000, 0.0096, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0618, 0.0000, 0.0289, 0.1832, 0.0232, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0450, 0.0000, 0.0191, 0.0000, 0.0208, 0.0494, 0.0000, 0.0195,
         0.0000, 0.0000, 0.0090, 0.0000, 0.0581, 0.0046, 0.0681, 0.0000, 0.1394,
         0.0163, 0.0219, 0.1688, 0.0000, 0.1387, 0.0000, 0.0000, 0.0422, 0.1423,
         0.1591, 0.1953, 0.0000, 0.0375, 0.0137, 0.0024, 0.0000, 0.0429, 0.0000,
         0.0958, 0.0000, 0.0915, 0.0000, 0.1268, 0.0029, 0.1045, 0.0832, 0.0000,
         0.0000, 0.1241, 0.1477, 0.0000, 0.0277, 0.0551, 0.1257, 0.1836, 0.0000,
         0.0549, 0.1299, 0.0089, 0.0000, 0.0875, 0.0000, 0.0122, 0.0051, 0.0000,
         0.0000, 0.0035, 0.0266, 0.0000, 0.1410, 0.0032, 0.0696, 0.0000, 0.0017,
         0.1256, 0.0000, 0.0000, 0.0096, 0.0240, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.1121, 0.0000, 0.0000, 0.0734, 0.0000, 0.0813, 0.0035, 0.0023,
         0.0599, 0.0225, 0.1234, 0.0083, 0.0306, 0.0000, 0.0000, 0.0391, 0.0000,
         0.0122, 0.0000, 0.0943, 0.0161, 0.0000, 0.0759, 0.0937, 0.0000, 0.0976,
         0.0000, 0.0101, 0.1322, 0.0000, 0.0033, 0.0000, 0.0000, 0.0000, 0.1640,
         0.0681, 0.1191, 0.0000, 0.0000, 0.0000, 0.0086, 0.0000, 0.0025, 0.0000,
         0.0000, 0.0000, 0.0275, 0.0832, 0.0329, 0.0676, 0.0183, 0.0000, 0.0504,
         0.1882, 0.0165, 0.0435, 0.1329, 0.0000, 0.0307, 0.0656, 0.0000, 0.0448,
         0.0791, 0.0000, 0.0000, 0.0000]], device='cuda:0')       
b = torch.tensor([[0.0242, 0.0000, 0.0000, 0.0000, 0.0000, 0.0546, 0.0000, 0.0000, 0.0521,
         0.0466, 0.1002, 0.1617, 0.0678, 0.0000, 0.0000, 0.0000, 0.1152, 0.0127,
         0.0000, 0.0000, 0.0533, 0.0824, 0.0000, 0.0000, 0.0000, 0.1531, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0163, 0.0284, 0.1260, 0.0000, 0.0641, 0.0000,
         0.0704, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0398, 0.0000, 0.0045, 0.0161, 0.0337, 0.0000, 0.0325, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0808, 0.0000, 0.1163, 0.0222, 0.1495, 0.0000, 0.0127,
         0.0000, 0.0231, 0.0000, 0.0735, 0.0827, 0.0000, 0.0000, 0.0042, 0.0000,
         0.0000, 0.0403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0007, 0.0722, 0.0000,
         0.0000, 0.0769, 0.0297, 0.0000, 0.0000, 0.0000, 0.1104, 0.0000, 0.0555,
         0.0088, 0.0000, 0.0000, 0.0000, 0.0274, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0177, 0.1227, 0.0020, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0387, 0.0000, 0.0000, 0.0000, 0.0220, 0.0646, 0.0000, 0.0263,
         0.0000, 0.0000, 0.0690, 0.0000, 0.0457, 0.0065, 0.0378, 0.0000, 0.1077,
         0.0349, 0.0357, 0.0000, 0.0000, 0.1222, 0.0000, 0.0000, 0.0441, 0.1400,
         0.1041, 0.1482, 0.0000, 0.0203, 0.0446, 0.0000, 0.0000, 0.0195, 0.0000,
         0.0551, 0.0000, 0.0741, 0.0000, 0.1108, 0.0000, 0.0906, 0.0744, 0.0000,
         0.0000, 0.1094, 0.0977, 0.0000, 0.0049, 0.0603, 0.0000, 0.1805, 0.0000,
         0.0731, 0.0963, 0.0222, 0.0000, 0.0739, 0.0000, 0.0228, 0.0254, 0.0000,
         0.0000, 0.0000, 0.0428, 0.0000, 0.0896, 0.0276, 0.0554, 0.0000, 0.0000,
         0.1122, 0.0000, 0.0000, 0.0201, 0.0751, 0.0000, 0.0000, 0.0247, 0.0000,
         0.0000, 0.1002, 0.0000, 0.0000, 0.0734, 0.0000, 0.0887, 0.0000, 0.0000,
         0.0467, 0.0241, 0.1241, 0.0103, 0.0491, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0431, 0.0000, 0.0653, 0.0060, 0.0000, 0.0764, 0.0741, 0.0000, 0.1098,
         0.0000, 0.0000, 0.0839, 0.0000, 0.0195, 0.0000, 0.0000, 0.0000, 0.0946,
         0.0658, 0.0909, 0.0000, 0.0000, 0.0000, 0.0333, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0135, 0.0817, 0.0264, 0.0760, 0.0448, 0.0000, 0.0000,
         0.1471, 0.0309, 0.0374, 0.1174, 0.0000, 0.0388, 0.0515, 0.0000, 0.0475,
         0.0634, 0.0000, 0.0000, 0.0000]], device='cuda:0')  

import matplotlib.pyplot as plt
import numpy as np

t1 = a.detach().cpu().flatten()
t2 = b.detach().cpu().flatten()

x = np.arange(len(t1))
width = 0.4  # bar width

plt.figure(figsize=(18,6))

plt.bar(x - width/2, t1, width=width, label="Tensor 1")
plt.bar(x + width/2, t2, width=width, label="Tensor 2")

plt.xlabel("Dimension Index")
plt.ylabel("Value")
plt.title("Per-Dimension Bar Comparison")
plt.legend()

plt.tight_layout()
plt.show()