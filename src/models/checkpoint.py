"""Model Checkpoint Serialization and Restoration Utilities.

Saves and reconstructs MultiHeadRestorationNet complete with architecture config,
parameter counts, degradation metadata, and state_dict.
"""

import os
import torch
from typing import Tuple, Dict, Any, Optional, Union
from .restoration_net import MultiHeadRestorationNet


def save_checkpoint(
    model: MultiHeadRestorationNet,
    filepath: str,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Saves model checkpoint containing architecture config, parameters, and metadata.
    
    Args:
        model: Trained or initialized MultiHeadRestorationNet instance
        filepath: Destination file path (e.g. checkpoints/multi_head_model.pth)
        config: Optional model configuration dictionary
        metadata: Optional training metadata (epoch, loss, optimizer state, etc.)
        
    Returns:
        Absolute path to saved checkpoint file
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if config is None:
        config = {
            "in_channels": model.in_channels,
            "encoder_channels": model.encoder_channels,
            "num_blocks": model.num_blocks,
            "scale_factor": model.scale_factor,
            "num_degradation_params": model.num_degradation_params
        }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "degradation_param_names": [
            "poisson_scale", "gaussian_std", "blur_ksize", "downsample_scale"
        ],
        "parameter_count": model.count_parameters(),
        "architecture_metadata": {
            "backbone": "NAFNet-Style Gated CNN Encoder-Decoder",
            "heads": ["Restoration (PixelShuffle 2x)", "Degradation (MLP)", "Uncertainty (Log-Variance Map)"]
        },
        "metadata": metadata or {}
    }

    torch.save(checkpoint, filepath)
    return os.path.abspath(filepath)


def load_checkpoint(
    filepath: str,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[MultiHeadRestorationNet, Dict[str, Any]]:
    """Loads checkpoint and reconstructs MultiHeadRestorationNet model without manual code edits.
    
    Args:
        filepath: Path to saved .pth checkpoint file
        device: Torch device to load model onto ('cpu' or 'cuda')
        
    Returns:
        Tuple of (reconstructed_model, checkpoint_dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint.get("model_config", {})

    model = MultiHeadRestorationNet(
        in_channels=cfg.get("in_channels", 1),
        encoder_channels=cfg.get("encoder_channels", [32, 64, 128]),
        num_blocks=cfg.get("num_blocks", [2, 2, 4]),
        scale_factor=cfg.get("scale_factor", 2),
        num_degradation_params=cfg.get("num_degradation_params", 4),
        min_log_variance=cfg.get("min_log_variance", -10.0),
        max_log_variance=cfg.get("max_log_variance", 10.0)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint
