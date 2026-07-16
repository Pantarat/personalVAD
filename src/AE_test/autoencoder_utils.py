#!/usr/bin/env python3
"""
Shared Autoencoder Utilities
Universal autoencoder class and loading functions for d-vector compression/denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path


def _make_norm_layer(norm_type, dim):
    """Build normalization layer by type string."""
    norm = str(norm_type).lower()
    if norm == 'batchnorm':
        return nn.BatchNorm1d(dim)
    if norm == 'layernorm':
        return nn.LayerNorm(dim)
    if norm in ('none', 'identity'):
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _make_activation_layer(activation_type):
    activation = str(activation_type).lower()
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'relu':
        return nn.ReLU()
    raise ValueError(f"Unsupported activation_type: {activation_type}")


def _normalize_hidden_dims(hidden_dims, default_dims):
    """Normalize hidden-dim specs so callers can pass ints, tuples, or lists."""
    if hidden_dims is None:
        hidden_dims = default_dims
    elif isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    else:
        hidden_dims = list(hidden_dims)

    return [int(v) for v in hidden_dims]


def _normalize_stack_refinement_hidden_dims(stack_hidden_dims, n_refinement_blocks, default_dims):
    """Normalize deep-stacked refinement dims.

    Accepts either:
    - flat dims: [1024, 1024] (shared by all refinement blocks)
    - per-block dims: [[512, 384], [256, 256]] or [(512, 384), (256, 256)]
    """
    n_refinement_blocks = int(max(0, n_refinement_blocks))
    if n_refinement_blocks == 0:
        return []

    if stack_hidden_dims is None:
        shared = _normalize_hidden_dims(default_dims, default_dims)
        return [list(shared) for _ in range(n_refinement_blocks)]

    values = list(stack_hidden_dims)
    if len(values) == 0:
        shared = _normalize_hidden_dims(default_dims, default_dims)
        return [list(shared) for _ in range(n_refinement_blocks)]

    # Flat/shared spec (e.g. [1024, 1024]).
    if all(not isinstance(v, (list, tuple)) for v in values):
        shared = [int(v) for v in values]
        return [list(shared) for _ in range(n_refinement_blocks)]

    # Per-block spec (e.g. [(512, 384), (256, 256)]).
    per_block = []
    for idx, block_dims in enumerate(values):
        if not isinstance(block_dims, (list, tuple)):
            raise ValueError(
                f"Invalid stack_refinement_hidden_dims[{idx}]={block_dims!r}. "
                "Use flat dims [d1, d2, ...] or per-block dims [[...], [...]]."
            )
        per_block.append([int(v) for v in block_dims])

    if len(per_block) == 1 and n_refinement_blocks > 1:
        return [list(per_block[0]) for _ in range(n_refinement_blocks)]

    if len(per_block) != n_refinement_blocks:
        raise ValueError(
            "stack_refinement_hidden_dims block count mismatch: "
            f"got {len(per_block)} blocks, expected {n_refinement_blocks} "
            f"(from n_stacked_daes={n_refinement_blocks + 1})."
        )

    return per_block


class DvectorAutoencoder(nn.Module):
    """
    Autoencoder to map overlap d-vectors to clean d-vectors
    
    Architecture:
    - Encoder: d-vector (256) -> hidden layers -> bottleneck
    - Decoder: bottleneck -> hidden layers -> d-vector (256)
    
    Supports configurable normalization (BatchNorm/LayerNorm) and optional
    residual skip connection for faster optimization.
    """
    
    def __init__(
        self,
        input_dim=256,
        hidden_dims=None,
        dropout_rate=0.2,
        norm_type='batchnorm',
        activation_type='tanh',
        use_residual=False,
        residual_scale_init=0.5,
    ):
        super().__init__()

        hidden_dims = _normalize_hidden_dims(hidden_dims, [128, 64, 128])
        if len(hidden_dims) < 1:
            raise ValueError(f"hidden_dims must have length >= 1 (got: {hidden_dims})")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type.lower()
        self.activation_type = str(activation_type).lower()
        self.use_residual = bool(use_residual)

        if self.use_residual:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.register_buffer('residual_scale', torch.tensor(0.0), persistent=False)

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        # First half of hidden_dims for encoder
        n_encoder_layers = len(self.hidden_dims) // 2
        for i in range(n_encoder_layers):
            encoder_layers.append(nn.Linear(prev_dim, self.hidden_dims[i]))
            encoder_layers.append(_make_norm_layer(self.norm_type, self.hidden_dims[i]))
            encoder_layers.append(_make_activation_layer(self.activation_type))
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = self.hidden_dims[i]
        
        # Bottleneck
        bottleneck_dim = self.hidden_dims[n_encoder_layers]
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        encoder_layers.append(_make_norm_layer(self.norm_type, bottleneck_dim))
        encoder_layers.append(_make_activation_layer(self.activation_type))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = bottleneck_dim
        
        # Second half of hidden_dims for decoder
        for i in range(n_encoder_layers + 1, len(self.hidden_dims)):
            decoder_layers.append(nn.Linear(prev_dim, self.hidden_dims[i]))
            decoder_layers.append(_make_norm_layer(self.norm_type, self.hidden_dims[i]))
            decoder_layers.append(_make_activation_layer(self.activation_type))
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = self.hidden_dims[i]
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Full forward pass: input -> encoded -> decoded"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.use_residual:
            decoded = decoded + (self.residual_scale * x)
        return F.normalize(decoded, p=2, dim=-1, eps=1e-12)
    
    def encode(self, x):
        """Encode to bottleneck representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from bottleneck representation"""
        decoded = self.decoder(z)
        return F.normalize(decoded, p=2, dim=-1, eps=1e-12)


class _RefinementMLP(nn.Module):
    """Refinement block used after the first DAE in deep stacked mode."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=None,
        dropout_rate=0.2,
        norm_type='batchnorm',
        activation_type='tanh',
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 1024]

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = [int(v) for v in hidden_dims]
        self.dropout_rate = float(dropout_rate)
        self.norm_type = str(norm_type).lower()
        self.activation_type = str(activation_type).lower()

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(_make_norm_layer(self.norm_type, hidden_dim))
            layers.append(_make_activation_layer(self.activation_type))
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepStackedDvectorAutoencoder(nn.Module):
    """
    Deep Stacked autoencoder variant with configurable first block and refinement MLPs:
    - First block: MLP with configurable hidden layers (e.g., 256→256 or 256→128→256)
    - Refinement blocks: each takes concat(prev_out, noisy - prev_out) and outputs refined vector
    
    Example architecture with n_stacked_daes=3:
    1. First block: 256 → [optional hidden] → 256 (MLP)
    2. Refinement block 1: 512 → 512 → 256
    3. Refinement block 2: 512 → 512 → 256
    """

    def __init__(
        self,
        input_dim=256,
        hidden_dims=None,
        dropout_rate=0.2,
        norm_type='batchnorm',
        activation_type='tanh',
        use_residual=False,
        residual_scale_init=0.5,
        n_stacked_daes=2,
        stack_refinement_hidden_dims=None,
        first_block_hidden_dims=None,
    ):
        super().__init__()

        if n_stacked_daes < 1:
            raise ValueError(f"n_stacked_daes must be >= 1 (got {n_stacked_daes})")

        if stack_refinement_hidden_dims is None:
            stack_refinement_hidden_dims = [1024, 1024]

        self.input_dim = int(input_dim)
        self.hidden_dims = _normalize_hidden_dims(hidden_dims, [128, 64, 128])
        self.dropout_rate = float(dropout_rate)
        self.norm_type = str(norm_type).lower()
        self.activation_type = str(activation_type).lower()
        self.use_residual = bool(use_residual)
        self.residual_scale_init = float(residual_scale_init)
        self.n_stacked_daes = int(n_stacked_daes)
        
        n_refinement_blocks = max(0, self.n_stacked_daes - 1)
        self.stack_refinement_hidden_dims = _normalize_stack_refinement_hidden_dims(
            stack_refinement_hidden_dims,
            n_refinement_blocks=n_refinement_blocks,
            default_dims=[1024, 1024],
        )
        
        # First block is an MLP (not a full autoencoder)
        first_block_hidden_dims = _normalize_hidden_dims(first_block_hidden_dims, [])
        self.first_block_hidden_dims = list(first_block_hidden_dims)
        
        self.first_block = _RefinementMLP(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
            hidden_dims=self.first_block_hidden_dims,
            dropout_rate=self.dropout_rate,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
        )

        self.refinement_blocks = nn.ModuleList()
        for block_hidden_dims in self.stack_refinement_hidden_dims:
            self.refinement_blocks.append(
                _RefinementMLP(
                    input_dim=2 * self.input_dim,
                    output_dim=self.input_dim,
                    hidden_dims=block_hidden_dims,
                    dropout_rate=self.dropout_rate,
                    norm_type=self.norm_type,
                    activation_type=self.activation_type,
                )
            )

    def forward(self, x):
        noisy = x
        out = self.first_block(noisy)

        for block in self.refinement_blocks:
            residual = noisy - out
            block_input = torch.cat([out, residual], dim=-1)
            out = block(block_input)

        return out

    def encode(self, x):
        """Pass through first block (no bottleneck for MLP-based first block)."""
        return self.first_block(x)

    def decode(self, z):
        """Identity decode for MLP-based architecture (no decoder)."""
        return z


class GreedyLayerwiseStackedDvectorAutoencoder(nn.Module):
    """Stacked autoencoder built from greedy layer-wise pretraining specs."""

    def __init__(
        self,
        input_dim=256,
        greedy_hidden_dims=None,
        dropout_rate=0.1,
        norm_type='batchnorm',
        activation_type='tanh',
    ):
        super().__init__()

        greedy_hidden_dims = _normalize_hidden_dims(greedy_hidden_dims, [])
        if len(greedy_hidden_dims) == 0:
            raise ValueError("greedy_hidden_dims must be non-empty for greedy_layerwise_stacked")

        self.input_dim = int(input_dim)
        self.greedy_hidden_dims = list(greedy_hidden_dims)
        self.dropout_rate = float(dropout_rate)
        self.norm_type = str(norm_type).lower()
        self.activation_type = str(activation_type).lower()

        def _norm(dim):
            return _make_norm_layer(self.norm_type, dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_dim = self.input_dim
        for hidden_dim in self.greedy_hidden_dims:
            self.encoders.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    _norm(hidden_dim),
                    _make_activation_layer(self.activation_type),
                    nn.Dropout(self.dropout_rate),
                )
            )
            self.decoders.append(nn.Sequential(nn.Linear(hidden_dim, prev_dim)))
            prev_dim = hidden_dim

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        for decoder in reversed(self.decoders):
            out = decoder(out)
        return F.normalize(out, p=2, dim=-1, eps=1e-12)


def _extract_state_dict(checkpoint):
    """Get a state dict from either raw state_dict or checkpoint payload."""
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


def _infer_input_dim_from_state_dict(state_dict):
    """Infer input dim from encoder linear weights when config is missing it."""
    for key in ('encoder.0.weight', 'first_dae.encoder.0.weight'):
        preferred = state_dict.get(key)
        if isinstance(preferred, torch.Tensor) and preferred.ndim == 2:
            return int(preferred.shape[1])

    for key, tensor in state_dict.items():
        if (
            (key.startswith('encoder.') or key.startswith('first_dae.encoder.'))
            and key.endswith('.weight')
            and isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
        ):
            return int(tensor.shape[1])
    return None


def _infer_hidden_dims_from_state_dict(state_dict):
    """Infer hidden_dims list from sequential encoder/decoder linear layers."""
    encoder_linear = []
    decoder_linear = []

    for key, tensor in state_dict.items():
        if not (key.endswith('.weight') and isinstance(tensor, torch.Tensor) and tensor.ndim == 2):
            continue

        parts = key.split('.')
        if key.startswith('first_dae.'):
            if len(parts) < 4:
                continue
            module_name, module_idx = parts[1], parts[2]
        else:
            if len(parts) < 3:
                continue
            module_name, module_idx = parts[0], parts[1]

        if not module_idx.isdigit():
            continue

        idx = int(module_idx)
        out_dim = int(tensor.shape[0])

        if module_name == 'encoder':
            encoder_linear.append((idx, out_dim))
        elif module_name == 'decoder':
            decoder_linear.append((idx, out_dim))

    if not encoder_linear:
        return None

    encoder_linear.sort(key=lambda x: x[0])
    hidden_dims = [out_dim for _, out_dim in encoder_linear]

    if decoder_linear:
        decoder_linear.sort(key=lambda x: x[0])
        if len(decoder_linear) > 1:
            hidden_dims.extend([out_dim for _, out_dim in decoder_linear[:-1]])

    if not hidden_dims:
        return None
    return hidden_dims


def _infer_greedy_hidden_dims_from_state_dict(state_dict):
    """Infer greedy hidden dims from encoders.{i}.0.weight tensors."""
    encoders = []
    for key, tensor in state_dict.items():
        if not key.startswith('encoders.'):
            continue
        if not key.endswith('.weight'):
            continue
        parts = key.split('.')
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            continue
        encoders.append((idx, int(tensor.shape[0])))

    if not encoders:
        return []

    encoders.sort(key=lambda x: x[0])
    return [out_dim for _, out_dim in encoders]


def _resolve_arch_config(config, state_dict):
    """Resolve architecture keys, including intermediate-finetune nested config fallback."""
    resolved = dict(config) if isinstance(config, dict) else {}

    nested = resolved.get('base_model_config', {})
    if not isinstance(nested, dict):
        nested = {}

    def pick(key, default=None):
        if key in resolved and resolved[key] is not None:
            return resolved[key]
        if key in nested and nested[key] is not None:
            return nested[key]
        return default

    model_type = str(pick('model_type', 'dvector_autoencoder')).lower()
    if any(k.startswith('encoders.') for k in state_dict.keys()):
        model_type = 'greedy_layerwise_stacked'
    if model_type in ('deepstackeddae', 'deep_stacked', 'deep_stacked_dae'):
        model_type = 'deep_stacked_dae'
    elif model_type in ('standard', 'autoencoder', 'dvectorae'):
        model_type = 'dvector_autoencoder'
    elif model_type in ('greedy_layerwise', 'greedy_layerwise_stacked'):
        model_type = 'greedy_layerwise_stacked'

    input_dim = pick('input_dim')
    hidden_dims = pick('hidden_dims')
    dropout_rate = pick('dropout_rate', 0.2)
    norm_type = pick('norm_type', 'batchnorm')
    activation_type = pick('activation_type', 'tanh')
    use_residual = pick('use_residual', False)
    residual_scale_init = pick('residual_scale_init', 0.5)
    n_stacked_daes = pick('n_stacked_daes', 1)
    first_block_hidden_dims = pick('first_block_hidden_dims', [])
    stack_refinement_hidden_dims = pick('stack_refinement_hidden_dims', [1024, 1024])
    greedy_hidden_dims = pick('greedy_hidden_dims', [])

    if model_type == 'greedy_layerwise_stacked' and not greedy_hidden_dims:
        greedy_hidden_dims = _infer_greedy_hidden_dims_from_state_dict(state_dict)
        if not greedy_hidden_dims:
            greedy_hidden_dims = _infer_hidden_dims_from_state_dict(state_dict) or []

    if input_dim is None:
        input_dim = _infer_input_dim_from_state_dict(state_dict)
    if hidden_dims is None and model_type != 'greedy_layerwise_stacked':
        hidden_dims = _infer_hidden_dims_from_state_dict(state_dict)

    if input_dim is None:
        raise KeyError(
            "Missing 'input_dim' in config and unable to infer from checkpoint state_dict"
        )
    if hidden_dims is None and model_type != 'greedy_layerwise_stacked':
        raise KeyError(
            "Missing 'hidden_dims' in config and unable to infer from checkpoint state_dict"
        )

    resolved['input_dim'] = int(input_dim)
    if hidden_dims is not None:
        resolved['hidden_dims'] = list(hidden_dims)
    resolved['dropout_rate'] = float(dropout_rate)
    resolved['norm_type'] = str(norm_type)
    resolved['activation_type'] = str(activation_type)
    resolved['use_residual'] = bool(use_residual)
    resolved['residual_scale_init'] = float(residual_scale_init)
    resolved['model_type'] = model_type
    resolved['n_stacked_daes'] = int(n_stacked_daes)
    resolved['first_block_hidden_dims'] = _normalize_hidden_dims(first_block_hidden_dims, [])
    resolved['greedy_hidden_dims'] = _normalize_hidden_dims(greedy_hidden_dims, [])
    resolved['stack_refinement_hidden_dims'] = _normalize_stack_refinement_hidden_dims(
        stack_refinement_hidden_dims,
        n_refinement_blocks=max(0, resolved['n_stacked_daes'] - 1),
        default_dims=[1024, 1024],
    )

    return resolved


def _build_autoencoder_from_config(config):
    """Instantiate the model class selected by config['model_type']."""
    model_type = str(config.get('model_type', 'dvector_autoencoder')).lower()

    if model_type == 'deep_stacked_dae':
        return DeepStackedDvectorAutoencoder(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            dropout_rate=config.get('dropout_rate', 0.2),
            norm_type=config.get('norm_type', 'batchnorm'),
            activation_type=config.get('activation_type', 'tanh'),
            use_residual=config.get('use_residual', False),
            residual_scale_init=config.get('residual_scale_init', 0.5),
            n_stacked_daes=config.get('n_stacked_daes', 2),
            stack_refinement_hidden_dims=config.get('stack_refinement_hidden_dims', [1024, 1024]),
            first_block_hidden_dims=config.get('first_block_hidden_dims', []),
        )

    if model_type == 'dvector_autoencoder':
        return DvectorAutoencoder(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            dropout_rate=config.get('dropout_rate', 0.2),
            norm_type=config.get('norm_type', 'batchnorm'),
            activation_type=config.get('activation_type', 'tanh'),
            use_residual=config.get('use_residual', False),
            residual_scale_init=config.get('residual_scale_init', 0.5),
        )

    if model_type == 'greedy_layerwise_stacked':
        return GreedyLayerwiseStackedDvectorAutoencoder(
            input_dim=config['input_dim'],
            greedy_hidden_dims=config.get('greedy_hidden_dims', []),
            dropout_rate=config.get('dropout_rate', 0.1),
            norm_type=config.get('norm_type', 'batchnorm'),
            activation_type=config.get('activation_type', 'tanh'),
        )

    raise ValueError(f"Unsupported model_type in config: {model_type}")


def load_autoencoder(model_path, config_path=None, device='cuda'):
    """
    Load trained autoencoder model from checkpoint.

    Supports two call styles for backward compatibility:
    1) Explicit paths:
       load_autoencoder(model_path, config_path, device='cuda') -> model
    2) Model directory:
       load_autoencoder(model_dir, device) -> (model, config)

    For directory mode, this function expects:
    - config.pkl
    - final_model.pth (or best_model.pth fallback)
    """

    def _is_device_like(value):
        if isinstance(value, torch.device):
            return True
        if isinstance(value, str):
            v = value.lower()
            return v == 'cpu' or v.startswith('cuda') or v.startswith('mps')
        return False

    # Backward-compatible argument parsing.
    # - Explicit paths: load_autoencoder(model_path, config_path, device)
    # - Directory mode: load_autoencoder(model_dir, device)
    return_with_config = False
    if _is_device_like(config_path):
        device = config_path
        model_dir = Path(model_path)
        if model_dir.is_file():
            resolved_model_path = model_dir
            resolved_config_path = model_dir.parent / 'config.pkl'
        else:
            resolved_model_path = model_dir / 'final_model.pth'
            if not resolved_model_path.exists():
                resolved_model_path = model_dir / 'best_model.pth'
            resolved_config_path = model_dir / 'config.pkl'
        return_with_config = True
    else:
        resolved_model_path = Path(model_path)
        resolved_config_path = Path(config_path)

    print(f"\n🧠 Loading autoencoder...")
    print(f"  Model: {resolved_model_path}")
    print(f"  Config: {resolved_config_path}")
    
    # Load config
    with open(resolved_config_path, 'rb') as f:
        config = pickle.load(f)

    # Load checkpoint first to allow architecture fallback/inference when needed.
    checkpoint = torch.load(resolved_model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    config = _resolve_arch_config(config, state_dict)
    
    # Create model
    model = _build_autoencoder_from_config(config)
    
    # Extract model state dict (handle both checkpoint format and direct state dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format with metadata
        model.load_state_dict(state_dict)
        if 'epoch' in checkpoint:
            print(f"  Loaded from epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    else:
        # Direct state dict format
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Autoencoder loaded successfully")
    model_type = config.get('model_type', 'dvector_autoencoder')
    if model_type == 'deep_stacked_dae':
        first_block_hidden_dims = config.get('first_block_hidden_dims', [])
        if first_block_hidden_dims:
            first_block_str = f"{config['input_dim']} -> {first_block_hidden_dims} -> {config['input_dim']}"
        else:
            first_block_str = f"{config['input_dim']} -> {config['input_dim']}"
        print(
            f"  Architecture: deep_stacked_dae ({config.get('n_stacked_daes', 2)} DAEs)"
        )
        print(f"  First block: {first_block_str} (MLP)")
        print(f"  Refinement hidden dims: {config.get('stack_refinement_hidden_dims', [1024, 1024])}")
        print(f"  Processed dim: {config['input_dim']}-dim (linear output)")
    elif model_type == 'greedy_layerwise_stacked':
        print("  Architecture: greedy_layerwise_stacked")
        print(f"  Greedy hidden dims: {config.get('greedy_hidden_dims', [])}")
        print(f"  Processed dim: {config['input_dim']}-dim (linear output)")
    else:
        print(f"  Architecture: {config['input_dim']} -> {config['hidden_dims']} -> {config['input_dim']}")
    print(f"  Norm: {config.get('norm_type', 'batchnorm')}, Residual: {config.get('use_residual', False)}")
    print(f"  Activation: {config.get('activation_type', 'tanh')}")

    if return_with_config:
        return model, config
    return model


def load_autoencoder_with_config(model_path, config_path, device='cuda'):
    """
    Load trained autoencoder model and return both model and config
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        config_path: Path to config file (.pkl file)
        device: 'cuda' or 'cpu'
    
    Returns:
        tuple: (model, config)
    """
    print(f"\n🧠 Loading autoencoder...")
    print(f"  Model: {model_path}")
    print(f"  Config: {config_path}")
    
    # Load config
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Load checkpoint first to allow architecture fallback/inference when needed.
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    config = _resolve_arch_config(config, state_dict)
    
    # Create model
    model = _build_autoencoder_from_config(config)
    
    # Extract model state dict (handle both checkpoint format and direct state dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format with metadata
        model.load_state_dict(state_dict)
        if 'epoch' in checkpoint:
            print(f"  Loaded from epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    else:
        # Direct state dict format
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Autoencoder loaded successfully")
    model_type = config.get('model_type', 'dvector_autoencoder')
    if model_type == 'deep_stacked_dae':
        first_block_hidden_dims = config.get('first_block_hidden_dims', [])
        if first_block_hidden_dims:
            first_block_str = f"{config['input_dim']} -> {first_block_hidden_dims} -> {config['input_dim']}"
        else:
            first_block_str = f"{config['input_dim']} -> {config['input_dim']}"
        print(
            f"  Architecture: deep_stacked_dae ({config.get('n_stacked_daes', 2)} DAEs)"
        )
        print(f"  First block: {first_block_str} (MLP)")
        print(f"  Refinement hidden dims: {config.get('stack_refinement_hidden_dims', [1024, 1024])}")
        print(f"  Processed dim: {config['input_dim']}-dim (linear output)")
    elif model_type == 'greedy_layerwise_stacked':
        print("  Architecture: greedy_layerwise_stacked")
        print(f"  Greedy hidden dims: {config.get('greedy_hidden_dims', [])}")
        print(f"  Processed dim: {config['input_dim']}-dim (linear output)")
    else:
        print(f"  Architecture: {config['input_dim']} -> {config['hidden_dims']} -> {config['input_dim']}")
    print(f"  Norm: {config.get('norm_type', 'batchnorm')}, Residual: {config.get('use_residual', False)}")
    print(f"  Activation: {config.get('activation_type', 'tanh')}")
    
    return model, config


def apply_autoencoder_to_dvectors(dvector_data, autoencoder, device='cuda'):
    """
    Apply autoencoder to all d-vectors in the dataset
    
    Args:
        dvector_data: Dict with structure {key: {'dvector': array, ...}}
        autoencoder: Trained DvectorAutoencoder model
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict with same structure but dvector replaced with processed d-vector
        Original d-vectors are kept in 'original_dvector' key
    """
    print(f"\n🔄 Applying autoencoder to d-vectors...")
    
    processed_data = {}
    
    with torch.no_grad():
        for key, data in dvector_data.items():
            dvector = data['dvector']
            dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device)
            processed_dvector = autoencoder(dvector_tensor).squeeze(0).cpu().numpy()
            
            # Create new data dict with processed d-vector
            processed_data[key] = data.copy()
            processed_data[key]['dvector'] = processed_dvector
            processed_data[key]['original_dvector'] = dvector  # Keep original for reference
    
    print(f"✓ Processed {len(processed_data)} d-vectors through autoencoder")
    return processed_data


def extract_bottleneck_features(dvector_data, autoencoder, device='cuda'):
    """
    Extract bottleneck features from d-vectors using encoder only
    
    Args:
        dvector_data: Dict with structure {key: {'dvector': array, ...}}
        autoencoder: Trained DvectorAutoencoder model
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict with same structure but dvector replaced with bottleneck features
    """
    print(f"\n🔧 Extracting bottleneck features...")
    print(f"  Input samples: {len(dvector_data)}")
    
    autoencoder.eval()
    bottleneck_data = {}
    
    with torch.no_grad():
        for key, data in dvector_data.items():
            dvector = data['dvector']
            dvector_tensor = torch.FloatTensor(dvector).unsqueeze(0).to(device)
            
            # Extract bottleneck (encoded representation)
            bottleneck = autoencoder.encode(dvector_tensor).squeeze(0).cpu().numpy()
            
            # Create new data dict with bottleneck
            bottleneck_data[key] = data.copy()
            bottleneck_data[key]['dvector'] = bottleneck
            bottleneck_data[key]['original_dvector'] = dvector
    
    print(f"✓ Extracted bottleneck features from {len(bottleneck_data)} d-vectors")
    
    if len(bottleneck_data) > 0:
        sample_bottleneck = list(bottleneck_data.values())[0]['dvector']
        print(f"  Bottleneck dimension: {sample_bottleneck.shape[0]}")
    
    return bottleneck_data
