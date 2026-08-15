import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file safely with fallback defaults.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dict containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def get_device(device_str: str = "auto") -> str:
    """Resolves target compute device (cuda, cpu, mps).
    
    Args:
        device_str: "auto", "cuda", "cpu", or "mps"
        
    Returns:
        String indicating resolved device name.
    """
    if device_str == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    return device_str
