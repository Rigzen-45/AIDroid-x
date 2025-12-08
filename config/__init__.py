"""
Configuration management
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

__all__ = [
    'load_config',
    'save_config'
]