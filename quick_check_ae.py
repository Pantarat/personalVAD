"""Quick check of autoencoder config to identify extraction vs identity models"""

import pickle
from pathlib import Path

# Paths to check
AE_PATHS = [
    'src/AE_test/test_outputs/dvector_ae_identity_300Dev_5s_trainOther500_100pctAmp_40000',
    'src/AE_test/test_outputs/dvector_ae_many_main_w_other_singles_30000',
]

print("=" * 80)
print("AUTOENCODER CONFIG CHECKER")
print("=" * 80)

for ae_path_str in AE_PATHS:
    print(f"\n{'=' * 80}")
    print(f"Model: {ae_path_str}")
    print(f"{'=' * 80}")
    
    ae_path = Path(ae_path_str)
    config_path = ae_path / 'config.pkl'
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        continue
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    print(f"\n📋 Config keys: {list(config.keys())}")
    
    # Key fields
    print(f"\n🔑 Key Configuration:")
    print(f"   Architecture: {config.get('hidden_dims', 'N/A')}")
    print(f"   Bottleneck: {config['hidden_dims'][len(config['hidden_dims'])//2] if 'hidden_dims' in config else 'N/A'}")
    print(f"   Input dim: {config.get('input_dim', 'N/A')}")
    print(f"   Dropout: {config.get('dropout_rate', 'N/A')}")
    
    # Model type identification
    print(f"\n🏷️  Model Type:")
    if 'training_scheme' in config:
        print(f"   ⚠️  EXTRACTION MODEL")
        print(f"   Training scheme: {config['training_scheme']}")
        if 'main_speakers' in config:
            print(f"   Main speakers: {config['main_speakers']}")
    else:
        print(f"   ✅ IDENTITY MODEL (no training_scheme)")
    
    if 'include_noise_identity' in config:
        print(f"   Include noise identity: {config['include_noise_identity']}")
    
    # Training info
    print(f"\n📊 Training Info:")
    if 'best_val_loss' in config:
        print(f"   Best val loss: {config['best_val_loss']:.6f}")
    if 'test_cosine_similarity_mean' in config:
        print(f"   Test cos sim (mean): {config['test_cosine_similarity_mean']:.6f}")
    if 'test_cosine_similarity_std' in config:
        print(f"   Test cos sim (std): {config['test_cosine_similarity_std']:.6f}")
    
    # Key insight for identity models
    if 'training_scheme' not in config and 'test_cosine_similarity_mean' in config:
        cos_sim = config['test_cosine_similarity_mean']
        if cos_sim > 0.999:
            print(f"\n   ✅ EXCELLENT identity mapping (cos_sim > 0.999)")
        elif cos_sim > 0.99:
            print(f"\n   ✓ Good identity mapping (cos_sim > 0.99)")
        else:
            print(f"\n   ⚠️  Moderate identity mapping (cos_sim = {cos_sim:.6f})")
            print(f"   Expected > 0.999 for true identity")

print(f"\n" + "=" * 80)
print(f"✅ Check complete")
print(f"=" * 80)
